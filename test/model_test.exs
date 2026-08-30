defmodule Qwen3_5.ModelTest do
  use ExUnit.Case, async: true

  alias Qwen3_5.Model

  # FlashAttention-3 refuses every client that is not CUDA, and only EXLA asks.
  # Under `Nx.Defn.Evaluator` the kernel's dense fallback runs instead, so the
  # model and its gradient can be exercised on CPU while the kernel cannot.
  @compiler Nx.Defn.Evaluator
  @batch 2
  @sequence 8

  # Only the counts are shrunk to run on CPU; `head_dim` and the dtype stay at
  # their defaults, 128 and BF16, the two things the kernel constrains. Two
  # query heads over one KV head, so the projections are wider than the stream
  # and a wrong layout cannot pass by coincidence.
  defp config(overrides \\ %{}) do
    Map.merge(
      %{
        "num_hidden_layers" => 2,
        "hidden_size" => 128,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "intermediate_size" => 256,
        "vocab_size" => 64
      },
      overrides
    )
  end

  defp build(config) do
    model = Model.new(config)
    {init, predict} = Axon.build(model, compiler: @compiler)
    state = init.(Nx.template({@batch, @sequence}, :u32), Axon.ModelState.empty())
    {model, state, predict}
  end

  defp tokens(config) do
    {tokens, _key} =
      Nx.Random.randint(Nx.Random.key(1), 0, config["vocab_size"],
        shape: {@batch, @sequence + 1},
        type: :u32
      )

    tokens
  end

  # A window and the same window shifted left, with the trailing axis the
  # sparse loss selects along.
  defp shifted(tokens) do
    {tokens[[.., 0..-2//1]], Nx.new_axis(tokens[[.., 1..-1//1]], -1)}
  end

  # Reduced in FP32: a normaliser summed over a 151_936-entry vocabulary in
  # BF16 loses most of the distribution.
  defp loss(targets, logits) do
    Axon.Losses.categorical_cross_entropy(targets, Nx.as_type(logits, :f32),
      from_logits: true,
      sparse: true,
      reduction: :mean
    )
  end

  test "logits over the vocabulary, in the checkpoint's dtype" do
    config = config()
    {_model, state, predict} = build(config)
    {input, _target} = shifted(tokens(config))

    logits = predict.(state, input)

    assert Nx.shape(logits) == {@batch, @sequence, 64}
    assert Nx.type(logits) == {:bf, 16}
  end

  test "parameters carry the checkpoint's names, its dtype, and Axon's layout" do
    {_model, state, _predict} = build(config())

    q_proj = state.data["layers.0.attention.q_proj"]["kernel"]
    assert Nx.type(q_proj) == {:bf, 16}
    assert Nx.shape(q_proj) == {128, 256}
    assert Nx.shape(state.data["layers.0.attention.k_proj"]["kernel"]) == {128, 128}
    assert Nx.shape(state.data["layers.0.attention.o_proj"]["kernel"]) == {256, 128}
    assert Nx.shape(state.data["layers.1.attention.k_norm"]["gamma"]) == {128}
    assert Nx.type(state.data["layers.1.attention.k_norm"]["gamma"]) == {:bf, 16}
    assert Nx.type(state.data["layers.0.input_norm"]["gamma"]) == {:bf, 16}
    assert Nx.shape(state.data["layers.1.gate_proj"]["kernel"]) == {128, 256}
    assert Nx.shape(state.data["layers.1.down_proj"]["kernel"]) == {256, 128}
    assert Nx.shape(state.data["embedding"]["kernel"]) == {64, 128}
    assert Nx.shape(state.data["lm_head"]["kernel"]) == {128, 64}
    assert Nx.shape(state.data["norm"]["gamma"]) == {128}
  end

  test "an untrained model's loss sits near ln(vocab)" do
    config = config()
    {_model, state, predict} = build(config)
    {input, target} = shifted(tokens(config))

    loss = loss(target, predict.(state, input)) |> Nx.to_number()

    assert is_float(loss)
    refute loss != loss, "loss is NaN"
    assert_in_delta loss, :math.log(64), 1.0
  end

  # The evaluator is interpretive, so a gradient through the stack is slow even
  # at this size. One layer keeps it honest without keeping it long.
  @tag timeout: 300_000
  test "Axon.Loop.trainer fits a batch" do
    config = config(%{"num_hidden_layers" => 1})
    {model, state, predict} = build(config)
    {input, target} = shifted(tokens(config))
    before = loss(target, predict.(state, input)) |> Nx.to_number()

    trained =
      model
      |> Axon.Loop.trainer(&loss/2, Polaris.Optimizers.adamw(learning_rate: 1.0e-2), log: 0)
      |> Axon.Loop.run(Stream.repeatedly(fn -> {input, target} end), state,
        iterations: 3,
        compiler: @compiler
      )

    after_ = loss(target, predict.(trained, input)) |> Nx.to_number()

    assert after_ < before

    refute Nx.to_number(Nx.all(Nx.equal(trained.data["norm"]["gamma"], 1.0))) == 1,
           "the final norm never moved"
  end

  describe "lora/2" do
    defp same?(a, b), do: Nx.to_number(Nx.all(Nx.equal(a, b))) == 1

    defp adapted(config, state) do
      model = Model.lora(Model.new(config), rank: 4, alpha: 8)
      {init, predict} = Axon.build(model, compiler: @compiler)
      {model, init.(Nx.template({@batch, @sequence}, :u32), state), predict}
    end

    test "adapters sit beside their sites, at zero, over a frozen base" do
      config = config(%{"num_hidden_layers" => 1})
      {_base, state, predict} = build(config)
      {_model, adapted, predict_adapted} = adapted(config, state)
      {input, _target} = shifted(tokens(config))

      lora_a = adapted.data["layers.0.attention.q_proj.lora_a"]["kernel"]
      lora_b = adapted.data["layers.0.attention.q_proj.lora_b"]["kernel"]
      assert Nx.shape(lora_a) == {128, 4}
      assert Nx.shape(lora_b) == {4, 256}
      assert Nx.type(lora_a) == {:f, 32}, "adapters keep FP32 master weights"
      assert Nx.type(adapted.data["layers.0.attention.q_proj"]["kernel"]) == {:bf, 16}
      refute Map.has_key?(adapted.data, "layers.0.gate_proj.lora_a")

      sites =
        for p <- ~w(q_proj k_proj v_proj o_proj),
            a <- ~w(lora_a lora_b),
            do: "layers.0.attention.#{p}.#{a}"

      assert adapted |> Axon.ModelState.trainable_parameters() |> Map.keys() |> Enum.sort() ==
               Enum.sort(sites)

      # `lora_b` starts at zero, so the adapted model is the base model.
      assert same?(predict_adapted.(adapted, input), predict.(state, input))
    end

    @tag timeout: 300_000
    test "Axon.Loop.trainer moves the adapters and nothing else" do
      config = config(%{"num_hidden_layers" => 1})
      {model, state, _predict} = adapted(config, Axon.ModelState.empty())
      {input, target} = shifted(tokens(config))

      trained =
        model
        |> Axon.Loop.trainer(&loss/2, Polaris.Optimizers.adamw(learning_rate: 1.0e-2), log: 0)
        |> Axon.Loop.run(Stream.repeatedly(fn -> {input, target} end), state,
          iterations: 2,
          compiler: @compiler
        )

      moved? = fn layer ->
        not same?(trained.data[layer]["kernel"], state.data[layer]["kernel"])
      end

      assert moved?.("layers.0.attention.q_proj.lora_b")
      # `A` gets no gradient while `B` is zero, so it moves on the second step.
      assert moved?.("layers.0.attention.q_proj.lora_a")
      refute moved?.("layers.0.attention.q_proj")
      refute moved?.("embedding")
      refute moved?.("lm_head")
      assert same?(trained.data["norm"]["gamma"], state.data["norm"]["gamma"])
    end
  end

  test "metadata this model does not implement is refused at the parse" do
    # Each would otherwise run and be quietly wrong: the wrong gate in every
    # MLP, the wrong positions in every layer, a dtype the kernel refuses.
    assert_raise ArgumentError, ~r/hidden_act/, fn ->
      Model.new(config(%{"hidden_act" => "gelu"}))
    end

    assert_raise ArgumentError, ~r/rope_scaling/, fn ->
      Model.new(config(%{"rope_scaling" => %{"type" => "linear", "factor" => 4.0}}))
    end

    assert_raise ArgumentError, ~r/torch_dtype/, fn ->
      Model.new(config(%{"torch_dtype" => "int8"}))
    end
  end

  # Axon reports a layer's error as a compile error carrying its message.
  test "a sequence past the checkpoint's context is refused" do
    {init, _predict} =
      Axon.build(Model.new(config(%{"max_position_embeddings" => @sequence - 1})),
        compiler: @compiler
      )

    assert_raise Axon.CompileError, ~r/max_position_embeddings/, fn ->
      init.(Nx.template({@batch, @sequence}, :u32), Axon.ModelState.empty())
    end
  end
end
