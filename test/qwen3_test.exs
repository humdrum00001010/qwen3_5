defmodule Qwen3Test do
  use ExUnit.Case, async: true

  alias Qwen3.{Config, Model, Training}

  # FlashAttention-3 refuses anything but BF16/FP16 at head dimension 128 or
  # 256, and it refuses every client that is not CUDA. `Nx.Defn.Evaluator`
  # never consults the protocol that refuses, so the shapes and the gradient
  # can be exercised here while the kernel itself cannot.
  @compiler Nx.Defn.Evaluator

  defp fixture(sequence \\ 16) do
    config = Config.tiny()
    {params, key} = Model.init(config, Nx.Random.key(42))
    {cos, sin} = Training.rope_table(config, sequence)
    {tokens, _key} = Nx.Random.randint(key, 0, config.vocab, shape: {1, sequence})
    {config, params, tokens, cos, sin}
  end

  test "the model produces logits over the vocabulary" do
    {config, params, tokens, cos, sin} = fixture()

    logits =
      Nx.Defn.jit_apply(&Model.forward/3, [tokens, params, [config: config, cos: cos, sin: sin]],
        compiler: @compiler
      )

    assert Nx.shape(logits) == {1, 16, config.vocab}
    assert Nx.type(logits) == config.type
  end

  test "the loss is finite and beats uniform after no training" do
    {config, params, tokens, cos, sin} = fixture()

    loss =
      Nx.Defn.jit_apply(&Training.loss/3, [params, tokens, [config: config, cos: cos, sin: sin]],
        compiler: @compiler
      )
      |> Nx.to_number()

    assert is_float(loss)
    refute loss != loss, "loss is NaN"
    # Untrained, cross-entropy should sit near ln(vocab) rather than diverge.
    assert_in_delta loss, :math.log(config.vocab), 2.0
  end

  # The evaluator is interpretive, so a gradient through the whole stack is
  # slow even at this size. One layer and a short sequence keep it honest
  # without keeping it long.
  @tag timeout: 300_000
  test "at initialisation only lora_b receives gradient" do
    config = %{Config.tiny() | layers: 1}
    {params, key} = Model.init(config, Nx.Random.key(7))
    {cos, sin} = Training.rope_table(config, 8)
    {tokens, _key} = Nx.Random.randint(key, 0, config.vocab, shape: {1, 8})

    gradients =
      Nx.Defn.jit_apply(
        fn params, tokens, opts -> Nx.Defn.grad(params, &Training.loss(&1, tokens, opts)) end,
        [params, tokens, [config: config, cos: cos, sin: sin]],
        compiler: @compiler
      )

    q_proj = elem(gradients.layers, 0).attention.q_proj
    magnitude = &(&1 |> Nx.abs() |> Nx.sum() |> Nx.to_number())

    # delta = (x . a') . b' * scale. The derivative with respect to `a` carries
    # a factor of `b`, which starts at zero, so `a` cannot move on the first
    # step. `b` can, and once it is nonzero `a` follows.
    assert magnitude.(q_proj.lora_b) > 0.0
    assert magnitude.(q_proj.lora_a) == 0.0
  end

  test "rotary tables are the model dtype and shaped for the sequence" do
    config = Config.tiny()
    {cos, sin} = Training.rope_table(config, 16)

    assert Nx.shape(cos) == {16, config.head_dim}
    assert Nx.shape(sin) == {16, config.head_dim}
    assert Nx.type(cos) == config.type
  end

  test "the 8B configuration matches what the kernel admits" do
    config = Config.qwen3_8b()

    assert config.head_dim == 128, "FA3 links instantiations for 128 and 256 only"
    assert config.type == {:bf, 16}
    assert rem(config.q_heads, config.kv_heads) == 0, "GQA needs whole groups"
    assert Config.groups(config) == 4

    # Tensor parallelism shards KV heads, so a partition count has to divide them.
    assert rem(config.kv_heads, 2) == 0
  end
end
