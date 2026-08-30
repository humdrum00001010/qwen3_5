defmodule Qwen3_5.Layers do
  @moduledoc false

  import Nx.Defn

  @doc """
  RMS norm over the last axis, accumulated in FP32.

  BF16 has about three decimal digits, so summing squares over 4096 elements in
  it loses the small terms entirely.
  """
  defn rms_norm(x, weight, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-6)
    f32 = Nx.as_type(x, {:f, 32})

    f32
    |> Nx.pow(2)
    |> Nx.mean(axes: [-1], keep_axes: true)
    |> Nx.add(opts[:eps])
    |> Nx.rsqrt()
    |> Nx.multiply(f32)
    |> Nx.as_type(Nx.type(x))
    |> Nx.multiply(weight)
  end

  @doc """
  Rotary position embedding over a BSHD tensor.

  `cos` and `sin` are `{sequence, head_dim}` and are shared by every layer, so
  they are computed once per step rather than per layer.
  """
  defn rope(x, cos, sin) do
    {_batch, sequence, _heads, head_dim} = Nx.shape(x)

    {left, right} = Nx.split(x, div(head_dim, 2), axis: 3)
    rotated = Nx.concatenate([Nx.negate(right), left], axis: 3)

    shaped_cos = Nx.reshape(cos, {1, sequence, 1, head_dim})
    shaped_sin = Nx.reshape(sin, {1, sequence, 1, head_dim})

    Nx.add(Nx.multiply(x, shaped_cos), Nx.multiply(rotated, shaped_sin))
  end

  @doc """
  Position table for `rope/3`, in the model's dtype.
  """
  def rope_table(sequence, head_dim, theta, type) do
    half = div(head_dim, 2)

    inv_freq =
      Nx.iota({half}, type: {:f, 32})
      |> Nx.multiply(2.0 / head_dim)
      |> then(&Nx.pow(theta, &1))
      |> then(&Nx.divide(1.0, &1))

    angles =
      Nx.iota({sequence}, type: {:f, 32})
      |> Nx.new_axis(1)
      |> Nx.multiply(Nx.new_axis(inv_freq, 0))
      |> then(&Nx.concatenate([&1, &1], axis: 1))

    {Nx.as_type(Nx.cos(angles), type), Nx.as_type(Nx.sin(angles), type)}
  end

  @doc "SwiGLU feed-forward."
  defn swiglu(x, gate, up, down) do
    gated = Nx.dot(x, [2], gate, [1])
    projected = Nx.dot(x, [2], up, [1])

    gated
    |> Nx.sigmoid()
    |> Nx.multiply(gated)
    |> Nx.multiply(projected)
    |> Nx.dot([2], down, [1])
  end
end
