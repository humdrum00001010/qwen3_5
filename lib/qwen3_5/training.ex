defmodule Qwen3_5.Training do
  @moduledoc """
  Causal language-model finetuning over the LoRA adapters.
  """

  import Nx.Defn

  alias Qwen3_5.Config
  alias Qwen3_5.Layers
  alias Qwen3_5.Model

  @doc """
  Mean cross-entropy of predicting each token from the ones before it.

  Reduced in FP32. The logits are BF16, and a normalizer summed over a 151936
  entry vocabulary in BF16 loses most of the distribution.
  """
  defn loss(params, tokens, opts \\ []) do
    opts = keyword!(opts, [:config, :cos, :sin])

    logits =
      Model.forward(tokens, params, config: opts[:config], cos: opts[:cos], sin: opts[:sin])

    predicted =
      logits
      |> Nx.slice_along_axis(0, Nx.axis_size(logits, 1) - 1, axis: 1)
      |> Nx.as_type({:f, 32})

    expected = Nx.slice_along_axis(tokens, 1, Nx.axis_size(tokens, 1) - 1, axis: 1)

    log_probabilities =
      Nx.subtract(predicted, Nx.logsumexp(predicted, axes: [-1], keep_axes: true))

    log_probabilities
    |> Nx.take_along_axis(Nx.new_axis(expected, -1), axis: -1)
    |> Nx.negate()
    |> Nx.mean()
  end

  @doc """
  One optimizer step over the adapters.

  The gradient is taken against the whole parameter tree, because the base
  weights have to reach the forward pass and `Nx.Defn.grad/2` returns the shape
  it was given. Everything that is not an adapter is then zeroed, so the
  optimizer sees the full tree but moves only `lora_a` and `lora_b`.
  """
  defn step(params, optimizer_state, tokens, opts \\ []) do
    opts = keyword!(opts, [:config, :cos, :sin, :update])

    {loss, gradients} =
      value_and_grad(params, fn params ->
        loss(params, tokens, config: opts[:config], cos: opts[:cos], sin: opts[:sin])
      end)

    {params, optimizer_state} = apply_update(params, gradients, optimizer_state, opts[:update])
    {loss, params, optimizer_state}
  end

  deftransformp apply_update(params, gradients, optimizer_state, update) do
    {updates, optimizer_state} = update.(adapters_only(gradients), optimizer_state, params)
    {Polaris.Updates.apply_updates(params, updates), optimizer_state}
  end

  defp adapters_only(%Nx.Tensor{} = tensor), do: Nx.multiply(tensor, 0)

  defp adapters_only(%{} = map) when not is_struct(map) do
    Map.new(map, fn
      {key, %Nx.Tensor{} = tensor} when key in [:lora_a, :lora_b] -> {key, tensor}
      {key, value} -> {key, adapters_only(value)}
    end)
  end

  defp adapters_only(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&adapters_only/1) |> List.to_tuple()
  end

  defp adapters_only(other), do: other

  @doc """
  Rotary tables for a sequence length, in the model's dtype.

  A sequence past the checkpoint's `max_position_embeddings` is refused rather
  than extrapolated: the table would still build, and the positions past the
  trained context would be ones the model has never seen.
  """
  def rope_table(%Config{} = config, sequence) do
    if sequence > config.max_positions do
      raise ArgumentError,
            "sequence #{sequence} exceeds the checkpoint's max_position_embeddings " <>
              "(#{config.max_positions})"
    end

    Layers.rope_table(sequence, config.head_dim, config.rope_theta, config.type)
  end
end
