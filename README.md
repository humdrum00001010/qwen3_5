# Qwen3.5 on FlashAttention-3

A Qwen3.5 decoder as an `Axon` model, with attention supplied by the
FlashAttention-3 Hopper kernel through
[ex_flashattention3](https://github.com/humdrum00001010/ex_flashattention3).
One module, `Qwen3_5.Model`.

```elixir
model = Qwen3_5.Model.from_json("/checkpoints/Qwen3.5-8B/config.json")

{init, predict} = Axon.build(model, compiler: EXLA)
logits = predict.(state, tokens)

model
|> Axon.Loop.trainer(loss, Polaris.Optimizers.adamw(learning_rate: 1.0e-4))
|> Axon.Loop.run(data, state, compiler: EXLA)
```

Dimensions are read from the `config.json` beside a checkpoint's weights;
any key left out defaults to Qwen3-8B's, so `Qwen3_5.Model.new(%{})` is that
model and a toy is a few overridden counts. The model is an ordinary
`Axon` graph, so running and training it is Axon's job: `examples/train.exs`
is a complete training loop at a size that runs on CPU.

## LoRA

`Qwen3_5.Model.lora/2` rewrites the graph so the attention projections carry
rank-`r` adapters and everything else is frozen, so the same `Axon.Loop.trainer`
call trains only the adapters. They keep FP32 master weights and compute in
the base's dtype. Qwen3-8B is about 16 GB of BF16 weights; a full finetune
with FP32 master weights and two Adam moments wants roughly 96 GB more, and
rank-16 adapters on the four attention projections are a few tens of
megabytes.

```elixir
model = Qwen3_5.Model.new(config) |> Qwen3_5.Model.lora(rank: 16, alpha: 32)
```

## What the kernel constrains

The model is written for FA3's contract rather than around it.

- **BSHD throughout.** Q, K, and V go from projection to kernel without a
  transpose. The custom call pins row-major layouts, so a BHSD detour would
  copy every projection in every layer.
- **`head_dim` is 128.** FA3 links one CUTLASS instantiation per head
  dimension and this build carries 128 and 256. Qwen3.5 uses 128.
- **BF16 or FP16, CUDA only.** Compiled with EXLA, anything else raises rather
  than substituting a score-matrix attention that would change the model's
  memory complexity. Under `Nx.Defn.Evaluator` there is no gate and a dense
  fallback runs, which is what the tests use.
- **Prefill only.** FA3 here requires `seqlen_q == seqlen_k` under a causal
  mask, so this trains but does not generate.

Every block Axon has is Axon's: embedding, dense, RMS norm, SiLU, the head
split. One custom layer holds the rest, rotary embedding and the kernel call.
The loss is the caller's; cast logits to FP32
before it, since a normaliser summed over a 151_936-entry vocabulary in BF16
loses most of the distribution.

## Setup

```sh
mix deps.get
```

On a GPU the application loads the kernel once at boot, before compiling
anything that contains an FA3 call:

```elixir
:ok = EXLA.load_dylib("/absolute/path/to/libfa3_xla.so")
```

`mix test` runs on CPU under `Nx.Defn.Evaluator`, against metadata shaped like
a checkpoint's with `head_dim` 128 and BF16 kept and every count shrunk. It
covers logit shapes, parameter names and layout, an untrained loss near
`ln(vocab)`, a few `Axon.Loop.trainer` steps fitting a batch, and that
metadata this model does not implement is refused at the parse. It exercises
the model and its gradient but **not the kernel**.

## Not implemented

- **Checkpoint loading.** Parameters initialise at random. A loader maps a
  checkpoint's tensors onto the layer names, transposing each `{out, in}`
  projection into Axon's `{in, units}`, and fills `lm_head` from the embedding
  when the checkpoint ties them.
- **Tokenization and data.** The model takes token ids. `examples/prompts.exs`
  fetches the harmful and harmless instruction sets a refusal direction is
  found and chosen from, and JailbreakBench as the benchmark, through `hf_hub`
  into `data/hub`, and stores the prepared sets as `data/prompts/*.csv`; nothing
  tokenizes them yet.
- **Decoding.** The kernel has no KV-cache entry point.
