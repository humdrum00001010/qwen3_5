defmodule Qwen3_5.Model do
  @moduledoc """
  Qwen3.5 decoder as an `Axon` model, with attention supplied by
  FlashAttention-3.

  `new/1` builds the model from a checkpoint's `config.json`; a key it leaves
  out defaults to Qwen3-8B's. The result is an ordinary `Axon` graph:
  `Axon.build/2` runs it and `Axon.Loop.trainer/3` trains it.

  Every block Axon has is Axon's: embedding, dense projections, RMS norm
  (including Qwen3.5's per-head QK norm), SiLU, and the head split. One custom
  layer holds what Axon does not have, rotary embedding and the kernel call.
  Q, K, and V reach `FlashAttention3.attention/4` in BSHD
  `{batch, sequence, heads, dim}` without a transpose, since the custom call
  pins row-major layouts and a BHSD detour would copy each projection on every
  layer.

  Layers are named after the checkpoint's tensors — `embedding`,
  `layers.3.attention.q_proj`, `layers.3.attention.q_norm`,
  `layers.3.down_proj`, `norm`, `lm_head` — and a dense kernel is Axon's
  `{in, units}`, the transpose of a checkpoint's `{out, in}`. `lm_head` is
  always its own layer, since a graph cannot share a parameter between two
  layers; a checkpoint with `tie_word_embeddings` fills it from the embedding.
  """

  import Nx.Defn

  @doc """
  The model, from a checkpoint's decoded `config.json`. Any key left out
  defaults to Qwen3-8B's, so `new(%{})` is that model and a toy is a few
  overridden counts.

  One input, `"tokens"`, of shape `{batch, sequence}`. The output is logits of
  shape `{batch, sequence, vocab_size}` in the checkpoint's `torch_dtype`,
  which is also the dtype every parameter initialises in.

  A `hidden_act` other than `silu`, a `rope_scaling`, or a dtype the kernel
  refuses fail here rather than running and being wrong.
  """
  def new(%{} = config) do
    dims = dims!(config)

    embedded =
      Axon.input("tokens", shape: {nil, nil})
      |> Axon.embedding(dims.vocab, dims.hidden,
        name: "embedding",
        kernel_initializer: Axon.Initializers.normal(scale: dims.initializer_range)
      )

    hidden =
      Enum.reduce(0..(dims.layers - 1)//1, embedded, &layer(&2, dims, "layers.#{&1}"))

    hidden
    |> Axon.rms_norm(name: "norm", epsilon: dims.norm_eps)
    |> dense(dims.vocab, dims, "lm_head")
    |> Axon.MixedPrecision.apply_policy(Axon.MixedPrecision.create_policy(params: dims.type))
  end

  @doc "The model, from the path to a checkpoint's `config.json`."
  def from_json(path) when is_binary(path) do
    path |> File.read!() |> JSON.decode!() |> new()
  end

  @doc """
  The model with low-rank adapters on its projections, and only those
  trainable.

  Each dense layer named by `:sites` — by default the four attention
  projections of every layer — gains `x·A·B · alpha/rank` beside its output.
  `A` is `{in, rank}` with a fan-in normal initialiser and `B` is
  `{rank, units}` at zero, so the adapted model computes what the base did
  until something trains it. They are layers named after their site,
  `layers.3.attention.q_proj.lora_a` and `.lora_b`, each with a `kernel`.

  Every other parameter is frozen in the graph, so `Axon.Loop.trainer/3` moves
  the adapters and nothing else; `Axon.ModelState.unfreeze/2` on a state
  reopens any of it. The adapters keep FP32 master weights and compute in the
  base's dtype, so an update smaller than a BF16 step is not lost.

      Qwen3_5.Model.new(config) |> Qwen3_5.Model.lora(rank: 16, alpha: 32)

  """
  def lora(%Axon{} = model, opts \\ []) do
    opts = Keyword.validate!(opts, rank: 16, alpha: 32, sites: ~w(q_proj k_proj v_proj o_proj))
    rank = opts[:rank]
    scale = opts[:alpha] / rank

    # FP32 master weights, computed in the base's dtype. Applied over the whole
    # graph afterwards: `apply_policy/3` rebuilds the node map from what its
    # output reaches, and inside the rewrite that would drop the site itself.
    %{policy: %{params: type}} = model.nodes[model.output]
    precision = Axon.MixedPrecision.create_policy(params: {:f, 32}, compute: type, output: type)

    model
    |> Axon.rewrite_nodes(fn
      %Axon.Node{op: :dense, meta: %{units: units}} = node ->
        name = name(node)

        if site?(name, opts[:sites]) do
          fn [x], output -> Axon.add(output, adapter(x, name, rank, units, scale)) end
        else
          :skip
        end

      _node ->
        :skip
    end)
    |> Axon.MixedPrecision.apply_policy(precision, &adapter?/1)
    |> Axon.map_nodes(&freeze_base/1)
  end

  defp adapter(x, name, rank, units, scale) do
    x
    |> Axon.dense(rank,
      use_bias: false,
      name: "#{name}.lora_a",
      kernel_initializer: :lecun_normal
    )
    |> Axon.dense(units, use_bias: false, name: "#{name}.lora_b", kernel_initializer: :zeros)
    |> Axon.nx(&Nx.multiply(&1, scale))
  end

  # Frozen in the graph rather than by a mask over a state, so the freeze
  # travels with the model into every state built from it. `Axon.freeze/2`
  # does this and is deprecated in favour of the mask.
  defp freeze_base(%Axon.Node{parameters: parameters} = node) do
    if adapter?(node) do
      node
    else
      %{node | parameters: Enum.map(parameters, &%{&1 | frozen: true})}
    end
  end

  defp adapter?(node), do: String.ends_with?(name(node), [".lora_a", ".lora_b"])

  defp site?(name, sites) do
    Enum.any?(sites, &(name == &1 or String.ends_with?(name, ".#{&1}")))
  end

  # Exact for every layer this module names; Axon names the rest by position
  # at build time.
  defp name(%Axon.Node{name: name_fn, op_name: op_name}), do: name_fn.(op_name, %{})

  # Read with Qwen3-8B's values as defaults, so a key a config leaves out is
  # that model's rather than an error, and a toy is a few overridden counts.
  # `head_dim` defaults to the kernel's 128 rather than to
  # `hidden_size / num_attention_heads`.
  defp dims!(config) do
    if Map.get(config, "hidden_act", "silu") != "silu" do
      raise ArgumentError,
            "unsupported hidden_act #{inspect(config["hidden_act"])}: this model gates with silu"
    end

    if config["rope_scaling"] do
      raise ArgumentError,
            "unsupported rope_scaling #{inspect(config["rope_scaling"])}: " <>
              "this model builds unscaled positions"
    end

    %{
      layers: Map.get(config, "num_hidden_layers", 36),
      hidden: Map.get(config, "hidden_size", 4096),
      q_heads: Map.get(config, "num_attention_heads", 32),
      kv_heads: Map.get(config, "num_key_value_heads", 8),
      head_dim: Map.get(config, "head_dim", 128),
      intermediate: Map.get(config, "intermediate_size", 12288),
      vocab: Map.get(config, "vocab_size", 151_936),
      rope_theta: Map.get(config, "rope_theta", 1_000_000.0),
      norm_eps: Map.get(config, "rms_norm_eps", 1.0e-6),
      max_positions: Map.get(config, "max_position_embeddings", 40_960),
      initializer_range: Map.get(config, "initializer_range", 0.02),
      type: type!(Map.get(config, "torch_dtype", "bfloat16"))
    }
  end

  # Translated rather than filtered: the kernel is the authority on what it
  # accepts, and it refuses anything but BF16 and FP16.
  defp type!("bfloat16"), do: {:bf, 16}
  defp type!("float16"), do: {:f, 16}
  defp type!("float32"), do: {:f, 32}

  defp type!(other) do
    raise ArgumentError, "unrecognised torch_dtype #{inspect(other)}"
  end

  # ## The graph

  defp layer(hidden, dims, prefix) do
    attended =
      hidden
      |> Axon.rms_norm(name: "#{prefix}.input_norm", epsilon: dims.norm_eps)
      |> attention(dims, "#{prefix}.attention")

    residual = Axon.add(hidden, attended)

    feed_forward =
      residual
      |> Axon.rms_norm(name: "#{prefix}.post_norm", epsilon: dims.norm_eps)
      |> mlp(dims, prefix)

    Axon.add(residual, feed_forward, name: "#{prefix}.output")
  end

  defp attention(x, dims, prefix) do
    q =
      x
      |> heads(dims.q_heads, dims, "#{prefix}.q_proj")
      |> Axon.rms_norm(name: "#{prefix}.q_norm", epsilon: dims.norm_eps)

    k =
      x
      |> heads(dims.kv_heads, dims, "#{prefix}.k_proj")
      |> Axon.rms_norm(name: "#{prefix}.k_norm", epsilon: dims.norm_eps)

    v = heads(x, dims.kv_heads, dims, "#{prefix}.v_proj")

    Axon.layer(&attention_forward/4, [q, k, v],
      name: prefix,
      op_name: :attention,
      theta: dims.rope_theta,
      max_positions: dims.max_positions
    )
    |> Axon.reshape({:batch, :auto, dims.q_heads * dims.head_dim})
    |> dense(dims.hidden, dims, "#{prefix}.o_proj")
  end

  # A projection split into heads: `{batch, sequence, heads, head_dim}`.
  defp heads(x, heads, dims, name) do
    x
    |> dense(heads * dims.head_dim, dims, name)
    |> Axon.reshape({:batch, :auto, heads, dims.head_dim})
  end

  # SwiGLU: `silu(gate) * up`, projected back down.
  defp mlp(x, dims, prefix) do
    gate = x |> dense(dims.intermediate, dims, "#{prefix}.gate_proj") |> Axon.silu()
    up = dense(x, dims.intermediate, dims, "#{prefix}.up_proj")

    Axon.multiply(gate, up)
    |> dense(dims.hidden, dims, "#{prefix}.down_proj")
  end

  defp dense(x, units, dims, name) do
    Axon.dense(x, units,
      name: name,
      use_bias: false,
      kernel_initializer: Axon.Initializers.normal(scale: dims.initializer_range)
    )
  end

  # ## The custom layer
  #
  # Rotary embedding and the kernel. Q, K, and V arrive normalised and split
  # into heads; Axon calls the layer with its inputs, then its options with
  # `:mode` added.

  defnp attention_forward(q, k, v, opts) do
    opts = keyword!(opts, [:theta, :max_positions, mode: :inference])
    {cos, sin} = rope_table(q, opts[:theta], opts[:max_positions])

    FlashAttention3.attention(rope(q, cos, sin), rope(k, cos, sin), v, causal: true)
  end

  # Rotary embedding over a BSHD tensor; `cos` and `sin` are `{sequence, dim}`.
  defnp rope(x, cos, sin) do
    {_batch, sequence, _heads, head_dim} = Nx.shape(x)
    {left, right} = Nx.split(x, div(head_dim, 2), axis: 3)
    rotated = Nx.concatenate([-right, left], axis: 3)

    x * Nx.reshape(cos, {1, sequence, 1, head_dim}) +
      rotated * Nx.reshape(sin, {1, sequence, 1, head_dim})
  end

  # Built at trace time from the sequence the layer sees, so it is a constant
  # under a compiler. A sequence past the checkpoint's context is refused
  # rather than extrapolated into positions the model has never seen.
  deftransformp rope_table(q, theta, max_positions) do
    sequence = Nx.axis_size(q, 1)
    head_dim = Nx.axis_size(q, 3)

    if sequence > max_positions do
      raise ArgumentError,
            "sequence #{sequence} exceeds the checkpoint's max_position_embeddings " <>
              "(#{max_positions})"
    end

    half = div(head_dim, 2)
    exponent = Nx.multiply(Nx.iota({half}, type: {:f, 32}), 2.0 / head_dim)
    inv_freq = Nx.divide(1.0, Nx.pow(theta, exponent))

    angles =
      Nx.iota({sequence, 1}, type: {:f, 32})
      |> Nx.multiply(Nx.reshape(inv_freq, {1, half}))
      |> then(&Nx.concatenate([&1, &1], axis: 1))

    {Nx.as_type(Nx.cos(angles), Nx.type(q)), Nx.as_type(Nx.sin(angles), Nx.type(q))}
  end
end
