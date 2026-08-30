defmodule Qwen3_5.Model do
  @moduledoc """
  The decoder stack, and parameter initialisation for it.

  The tree is described once, by `plan/2`, as shapes and the role each tensor
  is filled from. `init/3` and `template/2` are two walks over that one
  description, so they cannot disagree about what the tree holds.
  """

  import Nx.Defn

  alias Qwen3_5.{Attention, Config, Layers}

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
    |> unembed(params, config)
  end

  # A tied checkpoint has no `lm_head` tensor: the embedding matrix is the
  # output projection. The tree omits the key rather than holding a second copy
  # of it, so this reads the structure the config asked for.
  deftransformp unembed(hidden, params, config) do
    head = if config.tie_word_embeddings, do: params.embedding, else: params.lm_head
    Nx.dot(hidden, [2], head, [1])
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
  keeps the adapters this produces. `template/2` is the same tree without the
  values, which is what a loader wants.

  `:rank` and `:alpha` override `config :qwen3_5_finetune, :lora`. The
  environment is read only for what the caller leaves out, so supplying both
  makes the call self-contained.
  """
  def init(%Config{} = config, key, opts \\ []) do
    lora = lora_settings(opts)
    walk(plan(config, lora), key, &fill(&1, &2, &3, config, lora))
  end

  @doc """
  The parameter tree's shapes and dtypes, allocating nothing.

  Loading a checkpoint needs the tree, not values to overwrite: at 8B, `init/3`
  would draw 16 GB of random weights to be thrown away. Every tensor in it is
  an `Nx.template/2`, carrying shape and type and no data.
  """
  def template(%Config{} = config, opts \\ []) do
    {tree, _seed} =
      walk(plan(config, lora_settings(opts)), nil, fn _role, shape, seed ->
        {Nx.template(shape, config.type), seed}
      end)

    tree
  end

  # `:rank` and `:alpha` are the one part of the shape this project picks
  # rather than reads, so they are configured rather than parsed. `validate!`
  # rather than `Keyword.get`, because a misspelled option that silently falls
  # back to the configured value is the kind of thing that is found much later.
  defp lora_settings(opts) do
    opts = Keyword.validate!(opts, [:rank, :alpha])

    rank = Keyword.get_lazy(opts, :rank, fn -> configured(:rank) end)
    alpha = Keyword.get_lazy(opts, :alpha, fn -> configured(:alpha) end)

    unless is_integer(rank) and rank > 0 do
      raise ArgumentError, "LoRA :rank must be a positive integer, got: #{inspect(rank)}"
    end

    unless is_number(alpha) do
      raise ArgumentError, "LoRA :alpha must be a number, got: #{inspect(alpha)}"
    end

    %{rank: rank, alpha: alpha}
  end

  defp configured(key) do
    :qwen3_5_finetune
    |> Application.fetch_env!(:lora)
    |> Keyword.fetch!(key)
  end

  # The tree as shapes, with `{:tensor, role, shape}` wherever a tensor goes and
  # the role naming how it is filled. Maps and tuples are the structure around
  # them, so a projection is four of these rather than one.
  #
  # `lm_head` is absent when the config ties it, which is what makes the tree
  # match the tensors a tied checkpoint actually ships.
  defp plan(config, lora) do
    head_width = config.q_heads * config.head_dim
    kv_width = config.kv_heads * config.head_dim

    tree = %{
      embedding: {:tensor, :normal, {config.vocab, config.hidden}},
      layers: layer_plans(config, lora, head_width, kv_width),
      norm: {:tensor, :ones, {config.hidden}}
    }

    if config.tie_word_embeddings do
      tree
    else
      Map.put(tree, :lm_head, {:tensor, :normal, {config.vocab, config.hidden}})
    end
  end

  defp layer_plans(config, lora, head_width, kv_width) do
    layer = %{
      input_norm: {:tensor, :ones, {config.hidden}},
      post_norm: {:tensor, :ones, {config.hidden}},
      attention: %{
        q_proj: projection_plan({head_width, config.hidden}, lora),
        k_proj: projection_plan({kv_width, config.hidden}, lora),
        v_proj: projection_plan({kv_width, config.hidden}, lora),
        o_proj: projection_plan({config.hidden, head_width}, lora),
        q_norm: {:tensor, :ones, {config.head_dim}},
        k_norm: {:tensor, :ones, {config.head_dim}}
      },
      gate_proj: {:tensor, :normal, {config.intermediate, config.hidden}},
      up_proj: {:tensor, :normal, {config.intermediate, config.hidden}},
      down_proj: {:tensor, :normal, {config.hidden, config.intermediate}}
    }

    # A tuple, not a list: `Nx.LazyContainer` covers tuples and maps but not
    # lists, so a list of layers cannot be a defn input.
    layer |> List.duplicate(config.layers) |> List.to_tuple()
  end

  # Base weights are `{out, in}`, matching the checkpoint layout. `lora_b`
  # starts at zero, so the adapted model starts exactly at the base model and
  # no branch is needed for "no adapter yet".
  defp projection_plan({out, input} = shape, lora) do
    %{
      weight: {:tensor, :normal, shape},
      lora_a: {:tensor, :lora_a, {lora.rank, input}},
      lora_b: {:tensor, :zeros, {out, lora.rank}},
      scale: {:tensor, :scale, {}}
    }
  end

  # The `{:tensor, ...}` clause has to stay first. It is a three-tuple, so the
  # `is_tuple/1` clause would otherwise take a plan's terminal for a tuple of
  # layers and recurse into the role atom.
  #
  # `Enum.sort/1` because map iteration order is not guaranteed — it turns on
  # map size and OTP internals, neither of which this plan should depend on.
  # Sorting pins the order the accumulator moves in, so the same plan fills the
  # same way.
  defp walk({:tensor, role, shape}, seed, fun), do: fun.(role, shape, seed)

  defp walk(%{} = map, seed, fun) do
    {pairs, seed} =
      map
      |> Enum.sort()
      |> Enum.map_reduce(seed, fn {key, value}, seed ->
        {walked, seed} = walk(value, seed, fun)
        {{key, walked}, seed}
      end)

    {Map.new(pairs), seed}
  end

  defp walk(tuple, seed, fun) when is_tuple(tuple) do
    {walked, seed} = tuple |> Tuple.to_list() |> Enum.map_reduce(seed, &walk(&1, &2, fun))
    {List.to_tuple(walked), seed}
  end

  # One clause per role `plan/2` uses. All of them return `{tensor, key}`, so
  # `walk/3` can thread a single key through the whole tree.
  #
  # `:normal` and `:lora_a` draw, and pass on the key `Nx.Random` hands back.
  # The constants — `:ones`, `:zeros`, `:scale` — return the key they were
  # given, so they never advance the sequence. Adding one to `plan/2` leaves
  # every other tensor's values untouched.
  defp fill(:normal, shape, key, config, _lora) do
    normal(key, shape, config.initializer_range, config.type)
  end

  # `a` is scaled by `1/sqrt(in)` rather than drawn from the checkpoint's
  # initializer range, so the initial delta magnitude does not depend on the
  # projection's width.
  defp fill(:lora_a, {_rank, input} = shape, key, config, _lora) do
    normal(key, shape, 1.0 / :math.sqrt(input), config.type)
  end

  defp fill(:ones, shape, key, config, _lora), do: {constant(1, shape, config.type), key}
  defp fill(:zeros, shape, key, config, _lora), do: {constant(0, shape, config.type), key}

  # A tensor rather than a float because every value in a map handed to `defn`
  # has to be one.
  defp fill(:scale, _shape, key, config, lora) do
    {Nx.tensor(lora.alpha / lora.rank, type: config.type), key}
  end

  defp normal(key, shape, stddev, type) do
    {value, key} = Nx.Random.normal(key, 0.0, stddev, shape: shape)
    {Nx.as_type(value, type), key}
  end

  defp constant(value, shape, type) do
    Nx.broadcast(Nx.tensor(value, type: type), shape)
  end
end
