# Qwen3 finetuning on FlashAttention-3

LoRA finetuning of a Qwen3 decoder whose attention is the FlashAttention-3
Hopper kernel, bound through `Nx.block/4` and an XLA custom call by
[ex_flashattention3](../ex_flashattention3).

```elixir
config = Qwen3.Config.qwen3_8b()
{cos, sin} = Qwen3.Training.rope_table(config, 2048)
{loss, params, state} =
  Qwen3.Training.step(params, state, tokens,
    config: config, cos: cos, sin: sin, update: update)
```

## Why LoRA

Qwen3-8B is about 16 GB of BF16 weights. A full finetune also wants FP32 master
weights and two Adam moments, roughly 96 GB more, which does not fit the two
80 GB cards this targets. Rank-16 adapters on the four attention projections
are a few tens of megabytes, and they are the only tensors
`Qwen3.Training.step/4` moves.

## What the kernel constrains

The model is written for FA3's contract rather than around it.

- **BSHD throughout.** Q, K, and V go from projection to kernel without a
  transpose. The custom call pins row-major layouts, so a BHSD detour would
  materialize a copy of every projection in every layer.
- **`head_dim` is 128.** FA3 links one CUTLASS instantiation per head
  dimension and this build carries 128 and 256. Qwen3 uses 128, so nothing has
  to change; another model might not fit.
- **BF16 or FP16, CUDA only.** Anything else raises rather than substituting a
  score-matrix attention, which would change the memory complexity of the model
  that called it.
- **Prefill only.** FA3 here requires `seqlen_q == seqlen_k` under a causal
  mask, so this trains but does not generate. Decode needs a KV-cache entry
  point the kernel does not have.

Everything FA3 does not do stays in `defn`: RMS norm, Qwen3's per-head QK norm,
rotary embedding, SwiGLU, and the loss.

## Setup

Two checkouts, because both dependencies are unreleased:

```sh
export NX_MLIR_ATTRIBUTES_WORKTREE=/absolute/path/to/nx-worktree
export FLASH_ATTENTION3_PATH=/absolute/path/to/ex_flashattention3
mix deps.get
```

The Nx worktree must carry `EXLA.CustomCall.Spec.mlir_attributes`
([elixir-nx/nx#1824](https://github.com/elixir-nx/nx/pull/1824)).

On a GPU the application loads the kernel once at boot, before compiling
anything that contains an FA3 call:

```elixir
:ok = EXLA.NIF.load_dylib("/absolute/path/to/libfa3_xla.so")
```

## What is verified

`mix test` runs on CPU at `Qwen3.Config.tiny/0`, which keeps every dimension
the kernel constrains — `head_dim` 128, BF16 — and shrinks only the counts. It
covers logit shapes, a finite loss near `ln(vocab)` before training, the
rotary tables, that the 8B configuration is one FA3 admits, and that only
`lora_b` receives gradient at initialisation.

Those run under `Nx.Defn.Evaluator`, which never consults the protocol that
refuses non-CUDA clients, so they exercise the model and the gradient but
**not the kernel**. Nothing here has run on a GPU.

## Not implemented

- **Checkpoint loading.** `Qwen3.Model.init/3` produces random weights.
  Loading Qwen3-8B means mapping the published tensor names onto the parameter
  tree and keeping the adapters this produces.
- **Tokenization and data.** `step/4` takes token ids.
- **Tensor parallelism.** `FlashAttention3.TensorParallel` shards KV heads, and
  Qwen3-8B's 8 KV heads divide across 2, 4, or 8 ranks. The training step here
  is single-device.
