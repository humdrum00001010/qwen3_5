defmodule Qwen3_5.LoRA do
  @moduledoc """
  Low-rank adapters, and the only trainable parameters here.

  Qwen3.5-8B in BF16 is about 16 GB of weights. A full finetune also needs FP32
  master weights and two Adam moments, roughly 96 GB more, which does not fit
  the two 80 GB cards this targets. Rank-16 adapters on the four attention
  projections are a few tens of megabytes.

  Adapters are a rewrite over `Qwen3_5.Model`'s plan rather than part of it:
  `adapt/2` replaces the tensors a matcher picks with
  `%{weight: , lora_a: , lora_b: , scale: }` and leaves the rest alone. So which
  sites carry an adapter is a caller's choice, and `project/2` handles a
  projection either way.

  `lora_b` starts at zero, so the adapted model starts exactly at the base
  model and no branch is needed for "no adapter yet".
  """

  import Nx.Defn

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

  # Base weights are `{out, in}`, matching the checkpoint layout, so the
  # contraction names axes explicitly rather than relying on `Nx.dot/2`'s
  # default.
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

  @doc """
  Rewrites the tensors `matcher` picks into adapted projections.

  `matcher` takes a path into the plan and returns `{:adapt, rank, alpha}` or
  `:skip`, the same shape as `Axon.rewrite_nodes/2`'s callback. A path is the
  keys and tuple indices down to a tensor, so
  `[:layers, 0, :attention, :q_proj]` names the first layer's query projection.
  """
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

  # The `{:tensor, ...}` clause stays first for the same reason it does in
  # `Model.walk/3`: it is a three-tuple, and the tuple clause would otherwise
  # take it for a tuple of layers.
  defp rewrite({:tensor, role, shape} = tensor, path, matcher) do
    case matcher.(path) do
      :skip -> tensor
      {:adapt, rank, alpha} -> adapter(role, shape, rank, alpha)
    end
  end

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
  defp adapter(role, {out, input} = shape, rank, alpha) do
    %{
      weight: {:tensor, role, shape},
      lora_a: {:tensor, :lora_a, {rank, input}},
      lora_b: {:tensor, :zeros, {out, rank}},
      scale: {:tensor, {:scale, alpha / rank}, {}}
    }
  end

  @doc "The adapter tensors, which are what an optimizer should see."
  def trainable(params) do
    params
    |> flatten([])
    |> Enum.filter(fn {path, _} -> List.last(path) in [:lora_a, :lora_b] end)
    |> Map.new(fn {path, tensor} -> {Enum.join(path, "."), tensor} end)
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
