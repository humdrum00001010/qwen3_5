defmodule Qwen3_5.Config do
  @moduledoc """
  Model dimensions, read from a checkpoint's `config.json`.

  None of these numbers are written down here. They belong to a checkpoint, so
  they are parsed from the metadata that ships beside its weights; a value
  transcribed into code is a value that can disagree with the tensors it
  describes. `config/config.exs` holds what this project chooses instead.

  `head_dim` is not free: FlashAttention-3 links one CUTLASS instantiation per
  head dimension, and this build carries 128 and 256. Qwen3.5 uses 128.
  """

  @enforce_keys [
    :layers,
    :hidden,
    :q_heads,
    :kv_heads,
    :head_dim,
    :intermediate,
    :vocab,
    :rope_theta,
    :norm_eps,
    :type,
    :tie_word_embeddings,
    :initializer_range,
    :max_positions
  ]
  defstruct @enforce_keys

  @doc """
  Reads a checkpoint's `config.json`.

      Qwen3_5.Config.from_json("/checkpoints/Qwen3.5-8B/config.json")

  """
  def from_json(path) when is_binary(path) do
    path |> File.read!() |> JSON.decode!() |> from_metadata()
  end

  @doc """
  The part of a HuggingFace `config.json` this model needs.

  Everything but `head_dim` and `rope_scaling` is required, so metadata from
  another architecture fails here rather than as a shape error inside a layer.
  `head_dim` is optional in the format and derived when absent, which is what
  the reference implementations do. A stated value wins: Qwen3.5 sizes below 8B
  state a `head_dim` that is not `hidden_size / num_attention_heads`.

  `tie_word_embeddings` decides whether the checkpoint carries an `lm_head`
  tensor at all, so it is fetched rather than defaulted. Guessing it wrong is
  silent: a tied checkpoint has nothing to load onto a separate output head, a
  random one survives into training, and the loss still starts near
  `ln(vocab)`.

  `hidden_act` and `rope_scaling` are read only to be refused when they name
  something this model does not implement, for the same reason.
  """
  def from_metadata(%{} = metadata) do
    hidden = Map.fetch!(metadata, "hidden_size")
    q_heads = Map.fetch!(metadata, "num_attention_heads")

    activation!(Map.fetch!(metadata, "hidden_act"))
    rope_scaling!(Map.get(metadata, "rope_scaling"))

    %__MODULE__{
      layers: Map.fetch!(metadata, "num_hidden_layers"),
      hidden: hidden,
      q_heads: q_heads,
      kv_heads: Map.fetch!(metadata, "num_key_value_heads"),
      head_dim: Map.get(metadata, "head_dim") || div(hidden, q_heads),
      intermediate: Map.fetch!(metadata, "intermediate_size"),
      vocab: Map.fetch!(metadata, "vocab_size"),
      rope_theta: Map.fetch!(metadata, "rope_theta"),
      norm_eps: Map.fetch!(metadata, "rms_norm_eps"),
      type: type!(Map.fetch!(metadata, "torch_dtype")),
      tie_word_embeddings: Map.fetch!(metadata, "tie_word_embeddings"),
      initializer_range: Map.fetch!(metadata, "initializer_range"),
      max_positions: Map.fetch!(metadata, "max_position_embeddings")
    }
  end

  @doc "Query heads per key/value head."
  def groups(%__MODULE__{q_heads: q, kv_heads: kv}), do: div(q, kv)

  # Translated faithfully rather than filtered: `FlashAttention3` is the
  # authority on what it accepts, and it refuses anything but BF16 and FP16.
  defp type!("bfloat16"), do: {:bf, 16}
  defp type!("float16"), do: {:f, 16}
  defp type!("float32"), do: {:f, 32}

  defp type!(other) do
    raise ArgumentError, "unrecognised torch_dtype: #{inspect(other)}"
  end

  # Read to be refused rather than stored. `Layers.swiglu/4` is SiLU-gated, so
  # a checkpoint trained with another activation would run here and be wrong
  # rather than fail.
  defp activation!("silu"), do: :ok

  defp activation!(other) do
    raise ArgumentError,
          "unsupported hidden_act: #{inspect(other)}, this model implements silu"
  end

  # `Layers.rope_table/4` builds unscaled positions. A scaling strategy moves
  # every position, so it is refused rather than ignored.
  defp rope_scaling!(nil), do: :ok

  defp rope_scaling!(other) do
    raise ArgumentError,
          "unsupported rope_scaling: #{inspect(other)}, this model implements none"
  end
end
