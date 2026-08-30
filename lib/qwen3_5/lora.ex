defmodule Qwen3_5.LoRA do
  @moduledoc """
  Low-rank adapters, and the only trainable parameters here.

  Qwen3.5-8B in BF16 is about 16 GB of weights. A full finetune also needs FP32
  master weights and two Adam moments, roughly 96 GB more, which does not fit
  the two 80 GB cards this targets. Rank-16 adapters on the four attention
  projections are a few tens of megabytes.

  A projection is `%{weight: , lora_a: , lora_b: , scale: }`. `Qwen3_5.Model`
  builds it, since it owns the parameter tree; this module is what reads it.
  `lora_b` starts at zero, so the adapted model starts exactly at the base
  model and no branch is needed for "no adapter yet".
  """

  import Nx.Defn

  @doc """
  Applies a projection and its adapter.

  Base weights are `{out, in}`, matching the checkpoint layout, so the
  contraction names axes explicitly rather than relying on `Nx.dot/2`'s
  default.
  """
  defn project(x, projection) do
    base = Nx.dot(x, [2], projection.weight, [1])

    delta =
      x
      |> Nx.dot([2], projection.lora_a, [1])
      |> Nx.dot([2], projection.lora_b, [1])
      |> Nx.multiply(projection.scale)

    Nx.add(base, delta)
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
