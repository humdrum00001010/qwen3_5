defmodule Qwen3.Model do
  @moduledoc """
  The decoder stack, and parameter initialisation for it.
  """

  import Nx.Defn

  alias Qwen3.{Attention, Config, Layers, LoRA}

  @doc """
  Logits for a batch of token ids.

  The rotary table is built once and shared by every layer, since it depends
  only on sequence length.
  """
  defn forward(tokens, params, opts \\ []) do
    opts = keyword!(opts, [:config, :cos, :sin])
    config = opts[:config]

    params.embedding
    |> Nx.take(tokens, axis: 0)
    |> decoder_stack(params, config, opts[:cos], opts[:sin])
    |> Layers.rms_norm(params.norm, eps: config.norm_eps)
    |> Nx.dot([2], params.lm_head, [1])
  end

  deftransformp decoder_stack(hidden, params, config, cos, sin) do
    opts = [
      cos: cos,
      sin: sin,
      q_heads: config.q_heads,
      kv_heads: config.kv_heads,
      head_dim: config.head_dim,
      eps: config.norm_eps
    ]

    params.layers
    |> Tuple.to_list()
    |> Enum.reduce(hidden, &layer_forward(&2, &1, opts))
  end

  defnp layer_forward(hidden, layer, opts \\ []) do
    opts = keyword!(opts, [:cos, :sin, :q_heads, :kv_heads, :head_dim, :eps])

    attended =
      hidden
      |> Layers.rms_norm(layer.input_norm, eps: opts[:eps])
      |> Attention.forward(layer.attention,
        cos: opts[:cos],
        sin: opts[:sin],
        q_heads: opts[:q_heads],
        kv_heads: opts[:kv_heads],
        head_dim: opts[:head_dim],
        eps: opts[:eps]
      )

    residual = Nx.add(hidden, attended)

    feed_forward =
      residual
      |> Layers.rms_norm(layer.post_norm, eps: opts[:eps])
      |> Layers.swiglu(layer.gate_proj, layer.up_proj, layer.down_proj)

    Nx.add(residual, feed_forward)
  end

  @doc """
  Randomly initialised parameters, for tests and for shape checking.

  A real run replaces `weight`, `q_norm`, and the rest from a checkpoint and
  keeps the adapters this produces.
  """
  def init(%Config{} = config, key, opts \\ []) do
    rank = Keyword.get(opts, :rank, 16)
    alpha = Keyword.get(opts, :alpha, 32)
    type = config.type
    head_width = config.q_heads * config.head_dim
    kv_width = config.kv_heads * config.head_dim

    {embedding, key} = normal(key, {config.vocab, config.hidden}, type)
    {lm_head, key} = normal(key, {config.vocab, config.hidden}, type)

    {layers, key} =
      Enum.map_reduce(1..config.layers, key, fn _index, key ->
        init_layer(config, key, rank, alpha, head_width, kv_width, type)
      end)

    params = %{
      embedding: embedding,
      # A tuple, not a list: `Nx.LazyContainer` covers tuples and maps but not
      # lists, so a list of layers cannot be a defn input.
      layers: List.to_tuple(layers),
      norm: Nx.broadcast(Nx.tensor(1, type: type), {config.hidden}),
      lm_head: lm_head
    }

    {params, key}
  end

  defp init_layer(config, key, rank, alpha, head_width, kv_width, type) do
    {q, key} = projection(key, {head_width, config.hidden}, type, rank, alpha)
    {k, key} = projection(key, {kv_width, config.hidden}, type, rank, alpha)
    {v, key} = projection(key, {kv_width, config.hidden}, type, rank, alpha)
    {o, key} = projection(key, {config.hidden, head_width}, type, rank, alpha)

    {gate, key} = normal(key, {config.intermediate, config.hidden}, type)
    {up, key} = normal(key, {config.intermediate, config.hidden}, type)
    {down, key} = normal(key, {config.hidden, config.intermediate}, type)

    ones = fn shape -> Nx.broadcast(Nx.tensor(1, type: type), shape) end

    layer = %{
      input_norm: ones.({config.hidden}),
      post_norm: ones.({config.hidden}),
      attention: %{
        q_proj: q,
        k_proj: k,
        v_proj: v,
        o_proj: o,
        q_norm: ones.({config.head_dim}),
        k_norm: ones.({config.head_dim})
      },
      gate_proj: gate,
      up_proj: up,
      down_proj: down
    }

    {layer, key}
  end

  defp projection(key, shape, type, rank, alpha) do
    {weight, key} = normal(key, shape, type)
    LoRA.init(weight, rank, alpha, key)
  end

  defp normal(key, shape, type) do
    {value, key} = Nx.Random.normal(key, 0.0, 0.02, shape: shape)
    {Nx.as_type(value, type), key}
  end
end
