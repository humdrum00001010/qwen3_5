defmodule Qwen3_5.LoRA do
  @moduledoc """
  Low-rank adapters, and the only trainable parameters here.

  What this module owns is the adapter contract: the names `lora_a` and
  `lora_b`, `scale = alpha / rank`, that `lora_b` starts at zero so an adapted
  model computes exactly what it computed before, and that a rank has to factor
  the weight it sits on.

  On top of that contract sit two implementations, one per representation this
  project has — `Qwen3_5.Model`'s plan and an `Axon` graph. Each attaches
  adapters, computes them, and selects them for an optimizer. Only the contract
  is shared: a checkpoint kernel is `{out, in}` and Axon's is `{in, units}`, so
  the same formula contracts different axes over different trees.

  Adapters rather than a full finetune because Qwen3.5-8B is about 16 GB of
  BF16 weights, and FP32 master weights with two Adam moments would add roughly
  96 GB more than the two 80 GB cards this targets have.
  """

  import Nx.Defn

  @adapter_keys [:lora_a, :lora_b]

  # ## Computing — a checkpoint's `{out, in}` weights

  @doc """
  Applies a projection, with its adapter if it has one.

  An unadapted site is a bare `{out, in}` weight, so the two shapes are told
  apart by structure at trace time and no flag has to be threaded down from the
  plan.
  """
  deftransform project(x, projection) do
    if adapted?(projection) do
      adapted(x, projection)
    else
      Nx.dot(x, [2], projection, [1])
    end
  end

  # The contraction names axes explicitly rather than relying on `Nx.dot/2`'s
  # default, since the checkpoint layout puts `out` first.
  defnp adapted(x, projection) do
    base = Nx.dot(x, [2], projection.weight, [1])

    delta =
      x
      |> Nx.dot([2], projection.lora_a, [1])
      |> Nx.dot([2], projection.lora_b, [1])
      |> Nx.multiply(projection.scale)

    Nx.add(base, delta)
  end

  defp adapted?(%Nx.Tensor{}), do: false
  defp adapted?(%{} = projection), do: Map.has_key?(projection, :lora_a)
  defp adapted?(_other), do: false

  # ## Computing — Axon's `{in, units}` kernels

  @doc "A dense layer and its adapter, without a bias."
  defn lora_dense(x, kernel, lora_a, lora_b, opts) do
    opts = keyword!(opts, [:scale, mode: :inference])
    Axon.Layers.dense(x, kernel) + axon_delta(x, lora_a, lora_b, opts[:scale])
  end

  @doc "A dense layer and its adapter, with a bias."
  defn lora_dense(x, kernel, bias, lora_a, lora_b, opts) do
    opts = keyword!(opts, [:scale, mode: :inference])
    Axon.Layers.dense(x, kernel, bias) + axon_delta(x, lora_a, lora_b, opts[:scale])
  end

  # Two dense contractions rather than an explicit `Nx.dot/4`, so the adapter
  # contracts whatever axis Axon's dense does.
  defnp axon_delta(x, lora_a, lora_b, scale) do
    x
    |> Axon.Layers.dense(lora_a)
    |> Axon.Layers.dense(lora_b)
    |> Nx.multiply(scale)
  end

  # ## Attaching — the same rewrite over either representation

  @doc """
  Rewrites a model so the sites you pick carry adapters.

  Takes either representation. Given an `Axon` model and options, it rewrites
  the dense layers the graph is built from:

      model |> Qwen3_5.LoRA.adapt(rank: 16, alpha: 32)

  `:sites` is a predicate over the `%Axon.Node{}`, so a caller can adapt some
  dense layers and leave others alone. It defaults to all of them.

  Given a `Qwen3_5.Model` plan and a matcher, it rewrites the tensors the
  matcher picks:

      plan |> Qwen3_5.LoRA.adapt(Qwen3_5.LoRA.attention_projections(16, 32))

  `matcher` takes a path and returns `{:adapt, rank, alpha}` or `:skip`, the
  same shape as `Axon.rewrite_nodes/2`'s callback. A path is the keys and tuple
  indices down to a tensor, so `[:layers, 0, :attention, :q_proj]` names the
  first layer's query projection.

  A matcher is offered the model's projections and nothing else. Those are the
  tensors `project/2` runs, and so the only ones an adapter can reach: an
  adapter on an embedding would train every step and never be applied, since
  `Qwen3_5.Model.forward/3` gathers rows from it rather than projecting through
  it. Norms and embeddings are passed through without ever reaching `matcher`.
  """
  def adapt(%Axon{} = model, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:rank, :alpha, sites: fn _node -> true end])
    rank = Keyword.fetch!(opts, :rank)
    alpha = Keyword.fetch!(opts, :alpha)
    sites = Keyword.fetch!(opts, :sites)

    validate_rank!(rank)

    Axon.rewrite_nodes(model, fn
      %Axon.Node{op: :dense, meta: meta, name: name_fn} = node ->
        if sites.(node) do
          fn [%Axon{} = x], _output ->
            dense(x,
              units: meta[:units],
              use_bias: meta[:use_bias],
              name: name_fn,
              rank: rank,
              alpha: alpha
            )
          end
        else
          :skip
        end

      _node ->
        :skip
    end)
  end

  def adapt(plan, matcher) when is_function(matcher, 1) do
    rewrite(plan, [], matcher)
  end

  @doc """
  Matches the four attention projections in every layer.

  The default set, and the one the memory argument above is about: adapting the
  MLP as well would be most of the model's weights rather than a few tens of
  megabytes.
  """
  def attention_projections(rank, alpha) do
    fn
      [:layers, _index, :attention, name]
      when name in [:q_proj, :k_proj, :v_proj, :o_proj] ->
        {:adapt, rank, alpha}

      _path ->
        :skip
    end
  end

  @doc """
  An `Axon` dense layer carrying a rank-`:rank` adapter.

  Axon's kernels are `{in, units}`, so `lora_a` is `{in, rank}` and `lora_b` is
  `{rank, units}` and the adapter contracts the same axis the base does.
  """
  def dense(%Axon{} = x, opts) do
    opts =
      Keyword.validate!(opts, [
        :units,
        :name,
        :rank,
        :alpha,
        use_bias: true,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros
      ])

    units = Keyword.fetch!(opts, :units)
    rank = Keyword.fetch!(opts, :rank)
    alpha = Keyword.fetch!(opts, :alpha)

    kernel = Axon.param("kernel", [{:axis, -1}, units], initializer: opts[:kernel_initializer])
    lora_a = Axon.param("lora_a", [{:axis, -1}, rank], initializer: &fan_in_normal/3)
    lora_b = Axon.param("lora_b", [rank, units], initializer: :zeros)

    {inputs, op} =
      if opts[:use_bias] do
        bias = Axon.param("bias", [units], initializer: opts[:bias_initializer])
        {[x, kernel, bias, lora_a, lora_b], &lora_dense/6}
      else
        {[x, kernel, lora_a, lora_b], &lora_dense/5}
      end

    Axon.layer(op, inputs,
      name: opts[:name],
      meta: %{units: units, use_bias: opts[:use_bias], rank: rank, alpha: alpha},
      op_name: :dense,
      scale: alpha / rank
    )
  end

  @doc """
  Raises unless `rank` could describe an adapter at all.

  Shape only. Whether it is low enough to factor a particular weight depends on
  that weight, and is checked when the site is known.
  """
  def validate_rank!(rank) do
    unless is_integer(rank) and rank > 0 do
      raise ArgumentError, "LoRA :rank must be a positive integer, got: #{inspect(rank)}"
    end

    :ok
  end

  # The `{:tensor, ...}` clause stays first for the same reason it does in
  # `Model.walk/3`: it is a three-tuple, and the tuple clause would otherwise
  # take it for a tuple of layers.
  defp rewrite({:tensor, :projection, shape} = tensor, path, matcher) do
    case matcher.(path) do
      :skip ->
        tensor

      {:adapt, rank, alpha} ->
        adapter(shape, rank, alpha, path)

      other ->
        raise ArgumentError,
              "an :adapt matcher returns {:adapt, rank, alpha} or :skip, got: " <>
                "#{inspect(other)} for #{inspect(path)}"
    end
  end

  # Not a projection, so not a site an adapter could reach.
  defp rewrite({:tensor, _role, _shape} = tensor, _path, _matcher), do: tensor

  defp rewrite(%{} = map, path, matcher) do
    Map.new(map, fn {key, value} -> {key, rewrite(value, path ++ [key], matcher)} end)
  end

  defp rewrite(tuple, path, matcher) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.map(fn {value, index} -> rewrite(value, path ++ [index], matcher) end)
    |> List.to_tuple()
  end

  # `scale` is carried in the role rather than read back from settings, since a
  # matcher may give different sites different ranks.
  defp adapter({out, input} = shape, rank, alpha, path)
       when is_integer(out) and is_integer(input) do
    validate_rank!(rank)

    unless rank < min(out, input) do
      raise ArgumentError,
            "a rank has to be below #{min(out, input)} to factor the #{out}x#{input} " <>
              "weight at #{inspect(path)}, got: #{inspect(rank)}"
    end

    %{
      weight: {:tensor, :projection, shape},
      lora_a: {:tensor, :lora_a, {rank, input}},
      lora_b: {:tensor, :zeros, {out, rank}},
      scale: {:tensor, {:scale, alpha / rank}, {}}
    }
  end

  # ## Selecting — what an optimizer should move, from either tree

  @doc "The adapter tensors, which are what an optimizer should see."
  def trainable(%Axon.ModelState{data: data}) do
    names = Enum.map(@adapter_keys, &Atom.to_string/1)

    for {layer, params} <- data,
        {name, tensor} <- params,
        name in names,
        into: %{},
        do: {"#{layer}.#{name}", tensor}
  end

  def trainable(params) do
    params
    |> flatten([])
    |> Enum.filter(fn {path, _} -> List.last(path) in @adapter_keys end)
    |> Map.new(fn {path, tensor} -> {Enum.join(path, "."), tensor} end)
  end

  # `a` is scaled by `1/sqrt(in)`, so the initial delta does not depend on the
  # layer's width. `b` is zero, so the delta starts at zero regardless.
  defp fan_in_normal(shape, type, key) do
    {value, _key} =
      Nx.Random.normal(key, 0.0, 1.0 / :math.sqrt(elem(shape, 0)), shape: shape, type: type)

    value
  end

  defp flatten(%{} = map, prefix) when not is_struct(map) do
    Enum.flat_map(map, fn {key, value} -> flatten(value, prefix ++ [key]) end)
  end

  defp flatten(tuple, prefix) when is_tuple(tuple) and not is_struct(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, index} -> flatten(value, prefix ++ [index]) end)
  end

  defp flatten(%Nx.Tensor{} = tensor, prefix), do: [{prefix, tensor}]
  defp flatten(_other, _prefix), do: []
end
