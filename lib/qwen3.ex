defmodule Qwen3 do
  @moduledoc """
  Qwen3 decoder, with attention supplied by FlashAttention-3.

  The model is written for the kernel's contract rather than around it: Q, K,
  and V stay in BSHD `{batch, sequence, heads, dim}` from the projection to the
  call, because the custom call pins row-major layouts and a transpose in
  between would materialize a copy in every layer.

  See `Qwen3.Config` for the shapes and `Qwen3.Training` for the finetuning
  step. Only LoRA parameters are trained; see `Qwen3.LoRA` for why.
  """

  defstruct [:config, :embedding, :layers, :norm, :lm_head]

  @type t :: %__MODULE__{}
end
