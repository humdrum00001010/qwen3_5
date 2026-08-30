defmodule Qwen3.Config do
  @moduledoc """
  Model dimensions.

  `head_dim` is not free: FlashAttention-3 links one CUTLASS instantiation per
  head dimension, and this build carries 128 and 256. Qwen3 uses 128.
  """

  @enforce_keys [:layers, :hidden, :q_heads, :kv_heads, :head_dim, :intermediate, :vocab]
  defstruct [
    :layers,
    :hidden,
    :q_heads,
    :kv_heads,
    :head_dim,
    :intermediate,
    :vocab,
    rope_theta: 1_000_000.0,
    norm_eps: 1.0e-6,
    type: {:bf, 16}
  ]

  @doc """
  Qwen3-8B.

  Verify against the checkpoint's `config.json` before a real run; these are
  the published dimensions, not values read from weights.
  """
  def qwen3_8b do
    %__MODULE__{
      layers: 36,
      hidden: 4096,
      q_heads: 32,
      kv_heads: 8,
      head_dim: 128,
      intermediate: 12288,
      vocab: 151_936
    }
  end

  @doc """
  A model small enough to run on CPU, keeping every dimension the kernel
  constrains.

  `head_dim` stays 128 and the dtype stays BF16, so the same validation runs;
  only the counts shrink.
  """
  def tiny do
    %__MODULE__{
      layers: 2,
      hidden: 512,
      q_heads: 4,
      kv_heads: 2,
      head_dim: 128,
      intermediate: 1024,
      vocab: 256
    }
  end

  @doc "Query heads per key/value head."
  def groups(%__MODULE__{q_heads: q, kv_heads: kv}), do: div(q, kv)
end
