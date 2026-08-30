defmodule Qwen3_5.LoRAAxonTest do
  use ExUnit.Case, async: true

  alias Qwen3_5.LoRA

  defp base do
    Axon.input("x", shape: {nil, 6})
    |> Axon.dense(8, name: "up")
    |> Axon.relu()
    |> Axon.dense(4, name: "down")
  end

  defp built(model) do
    {init_fn, predict_fn} = Axon.build(model)
    {init_fn.(Nx.template({1, 6}, :f32), Axon.ModelState.empty()), predict_fn}
  end

  test "every dense layer gains an adapter beside its kernel" do
    {state, _predict} = built(LoRA.adapt(base(), rank: 2, alpha: 4))

    assert Enum.sort(Map.keys(state.data["up"])) == ["bias", "kernel", "lora_a", "lora_b"]

    assert Map.keys(LoRA.trainable(state)) |> Enum.sort() ==
             ["down.lora_a", "down.lora_b", "up.lora_a", "up.lora_b"]
  end

  test "adapting does not change what the model computes" do
    # `lora_b` starts at zero, so the delta is zero and an adapted model is the
    # base model until something trains it. This is what makes the rewrite safe
    # to apply to a checkpoint.
    {state, predict} = built(LoRA.adapt(base(), rank: 2, alpha: 4))
    x = Nx.broadcast(0.5, {1, 6})

    plain =
      x
      |> Axon.Layers.dense(state.data["up"]["kernel"], state.data["up"]["bias"])
      |> Nx.max(0.0)
      |> Axon.Layers.dense(state.data["down"]["kernel"], state.data["down"]["bias"])

    assert Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(predict.(state, x), plain)))) == 0.0
  end

  test "sites chooses which dense layers are adapted" do
    only_down = fn node -> node.name.(:dense, %{}) == "down" end
    {state, _predict} = built(LoRA.adapt(base(), rank: 2, alpha: 4, sites: only_down))

    assert Map.keys(LoRA.trainable(state)) |> Enum.sort() == ["down.lora_a", "down.lora_b"]
    assert Enum.sort(Map.keys(state.data["up"])) == ["bias", "kernel"]
  end

  test "a rank that cannot make an adapter names itself" do
    assert_raise ArgumentError, ~r/:rank must be a positive integer/, fn ->
      LoRA.adapt(base(), rank: 0, alpha: 4)
    end
  end
end
