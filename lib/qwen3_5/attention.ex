defmodule Qwen3_5.Attention do
  @moduledoc """
  Grouped-query attention, with the kernel doing the attention.

  Everything here is the part FlashAttention-3 does not do: projections,
  Qwen3.5's per-head QK norm, and rotary embedding. The kernel takes post-norm,
  post-RoPE Q/K/V and returns the attention output.
  """

  import Nx.Defn

  alias Qwen3_5.{Layers, LoRA}

  @doc """
  One attention block.

  Q, K, and V are reshaped straight into BSHD and handed to the kernel without
  a transpose. The custom call pins row-major layouts, so a BHSD detour would
  materialize a copy of each projection on every layer.
  """
  defn forward(x, params, opts \\ []) do
    opts = keyword!(opts, [:cos, :sin, :q_heads, :kv_heads, :head_dim, :eps])
    {batch, sequence, _hidden} = Nx.shape(x)

    q =
      x
      |> LoRA.project(params.q_proj)
      |> Nx.reshape({batch, sequence, opts[:q_heads], opts[:head_dim]})
      |> Layers.rms_norm(params.q_norm, eps: opts[:eps])
      |> Layers.rope(opts[:cos], opts[:sin])

    k =
      x
      |> LoRA.project(params.k_proj)
      |> Nx.reshape({batch, sequence, opts[:kv_heads], opts[:head_dim]})
      |> Layers.rms_norm(params.k_norm, eps: opts[:eps])
      |> Layers.rope(opts[:cos], opts[:sin])

    v =
      x
      |> LoRA.project(params.v_proj)
      |> Nx.reshape({batch, sequence, opts[:kv_heads], opts[:head_dim]})

    q
    |> FlashAttention3.attention(k, v, causal: true)
    |> Nx.reshape({batch, sequence, opts[:q_heads] * opts[:head_dim]})
    |> LoRA.project(params.o_proj)
  end
end
