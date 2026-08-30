defmodule Qwen3_5Test do
  use ExUnit.Case, async: true

  alias Qwen3_5.{Config, Model, Training}

  # FlashAttention-3 refuses anything but BF16/FP16 at head dimension 128 or
  # 256, and it refuses every client that is not CUDA. `Nx.Defn.Evaluator`
  # never consults the protocol that refuses, so the shapes and the gradient
  # can be exercised here while the kernel itself cannot.
  @compiler Nx.Defn.Evaluator

  # A checkpoint's metadata, shrunk to run on CPU. `head_dim` stays 128 and the
  # dtype stays BF16 — the two things the kernel constrains — so only the
  # counts are small.
  defp metadata(overrides \\ %{}) do
    Map.merge(
      %{
        "num_hidden_layers" => 2,
        "hidden_size" => 512,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "head_dim" => 128,
        "intermediate_size" => 1024,
        "vocab_size" => 256,
        "rope_theta" => 1_000_000.0,
        "rms_norm_eps" => 1.0e-6,
        "torch_dtype" => "bfloat16",
        "hidden_act" => "silu",
        "initializer_range" => 0.02,
        "max_position_embeddings" => 4096,
        "tie_word_embeddings" => false
      },
      overrides
    )
  end

  defp config(overrides \\ %{}), do: Config.from_metadata(metadata(overrides))

  defp shapes(%Nx.Tensor{} = tensor), do: {Nx.shape(tensor), Nx.type(tensor)}
  defp shapes(%{} = map), do: Map.new(map, fn {key, value} -> {key, shapes(value)} end)

  defp shapes(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&shapes/1) |> List.to_tuple()
  end

  defp fixture(sequence \\ 16) do
    config = config()
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
    config = config(%{"num_hidden_layers" => 1})
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
    config = config()
    {cos, sin} = Training.rope_table(config, 16)

    assert Nx.shape(cos) == {16, config.head_dim}
    assert Nx.shape(sin) == {16, config.head_dim}
    assert Nx.type(cos) == config.type
  end

  test "parsed metadata matches what the kernel admits" do
    config = config()

    assert config.head_dim == 128, "FA3 links instantiations for 128 and 256 only"
    assert config.type == {:bf, 16}
    assert rem(config.q_heads, config.kv_heads) == 0, "GQA needs whole groups"
    assert Config.groups(config) == 2

    # Tensor parallelism shards KV heads, so a partition count has to divide them.
    assert rem(config.kv_heads, 2) == 0
  end

  describe "tie_word_embeddings" do
    test "a tied tree carries no lm_head, and the embedding is the output head" do
      tied = config(%{"num_hidden_layers" => 1, "tie_word_embeddings" => true})
      {params, key} = Model.init(tied, Nx.Random.key(0))

      refute Map.has_key?(params, :lm_head),
             "a tied checkpoint ships no lm_head tensor, so the tree must not hold one"

      {cos, sin} = Training.rope_table(tied, 8)
      {tokens, _key} = Nx.Random.randint(key, 0, tied.vocab, shape: {1, 8})

      logits =
        Nx.Defn.jit_apply(&Model.forward/3, [tokens, params, [config: tied, cos: cos, sin: sin]],
          compiler: @compiler
        )

      assert Nx.shape(logits) == {1, 8, tied.vocab}
    end

    test "an untied tree carries its own lm_head" do
      {params, _key} = Model.init(config(), Nx.Random.key(0))
      assert Nx.shape(params.lm_head) == {config().vocab, config().hidden}
    end
  end

  # Loading a checkpoint wants the tree, not 16 GB of values to overwrite.
  describe "template/2" do
    test "describes the same tree init/3 builds, without allocating it" do
      config = config()
      {params, _key} = Model.init(config, Nx.Random.key(0))
      template = Model.template(config)

      assert shapes(template) == shapes(params)
    end

    test "follows the config into the tied shape" do
      refute Map.has_key?(Model.template(config(%{"tie_word_embeddings" => true})), :lm_head)
    end
  end

  describe "init/3 options" do
    test "rank and alpha reach the shapes they decide" do
      {params, _key} = Model.init(config(), Nx.Random.key(0), rank: 4, alpha: 8)
      q_proj = elem(params.layers, 0).attention.q_proj

      assert {4, 512} = Nx.shape(q_proj.lora_a)
      assert Nx.to_number(q_proj.scale) == 2.0
    end

    test "a misspelled option is refused rather than silently defaulted" do
      assert_raise ArgumentError, ~r/rnak/, fn ->
        Model.init(config(), Nx.Random.key(0), rnak: 4)
      end
    end

    test "a rank that cannot make an adapter names itself" do
      # Otherwise this surfaces from inside Nx as "invalid dimension in axis 1".
      assert_raise ArgumentError, ~r/:rank must be a positive integer/, fn ->
        Model.init(config(), Nx.Random.key(0), rank: 0)
      end
    end
  end

  describe "rope_table/2" do
    test "a sequence past the checkpoint's context is refused" do
      assert_raise ArgumentError, ~r/max_position_embeddings/, fn ->
        Training.rope_table(config(), 4097)
      end
    end
  end

  # A checkpoint's `config.json` names its fields the way transformers does,
  # not the way this model does, so the mapping is the part worth pinning.
  describe "from_metadata/1" do
    test "transformers names land on this model's names" do
      config = Config.from_metadata(metadata())

      assert config.layers == 2
      assert config.hidden == 512
      assert config.q_heads == 4
      assert config.kv_heads == 2
      assert config.head_dim == 128
      assert config.intermediate == 1024
      assert config.vocab == 256
      assert config.rope_theta == 1_000_000.0
      assert config.norm_eps == 1.0e-6
      assert config.type == {:bf, 16}
      assert config.initializer_range == 0.02
      assert config.max_positions == 4096
      assert config.tie_word_embeddings == false
    end

    test "a variant this model does not implement is refused at the parse" do
      # Both would otherwise run and be quietly wrong: the wrong activation in
      # every MLP, the wrong positions in every layer.
      assert_raise ArgumentError, ~r/hidden_act/, fn ->
        Config.from_metadata(metadata(%{"hidden_act" => "gelu"}))
      end

      assert_raise ArgumentError, ~r/rope_scaling/, fn ->
        Config.from_metadata(
          metadata(%{"rope_scaling" => %{"type" => "linear", "factor" => 4.0}})
        )
      end
    end

    test "tie_word_embeddings is required rather than assumed" do
      assert_raise KeyError, fn ->
        Config.from_metadata(Map.delete(metadata(), "tie_word_embeddings"))
      end
    end

    test "a stated head_dim wins over the derived one" do
      narrow = metadata(%{"hidden_size" => 256})

      # 256 / 4 is 64, and the metadata says 128. Qwen3.5 sizes below 8B state a
      # head_dim that is not hidden_size / num_attention_heads, so deriving
      # unconditionally would build the wrong model against a real checkpoint.
      assert Config.from_metadata(narrow).head_dim == 128
      assert Config.from_metadata(Map.delete(narrow, "head_dim")).head_dim == 64
    end

    test "metadata from another architecture fails at the parse" do
      assert_raise KeyError, fn -> Config.from_metadata(%{"hidden_size" => 512}) end

      assert_raise ArgumentError, fn ->
        Config.from_metadata(metadata(%{"torch_dtype" => "int8"}))
      end
    end
  end
end

# Mutates the application environment, so it cannot run beside the async tests.
defmodule Qwen3_5.LoRASettingsTest do
  use ExUnit.Case, async: false

  alias Qwen3_5.{Config, Model}

  setup do
    configured = Application.fetch_env!(:qwen3_5_finetune, :lora)
    on_exit(fn -> Application.put_env(:qwen3_5_finetune, :lora, configured) end)
    :ok
  end

  defp config do
    Config.from_metadata(%{
      "num_hidden_layers" => 1,
      "hidden_size" => 512,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "head_dim" => 128,
      "intermediate_size" => 1024,
      "vocab_size" => 256,
      "rope_theta" => 1_000_000.0,
      "rms_norm_eps" => 1.0e-6,
      "torch_dtype" => "bfloat16",
      "hidden_act" => "silu",
      "initializer_range" => 0.02,
      "max_position_embeddings" => 4096,
      "tie_word_embeddings" => false
    })
  end

  test "supplying both rank and alpha makes the call self-contained" do
    Application.delete_env(:qwen3_5_finetune, :lora)

    {params, _key} = Model.init(config(), Nx.Random.key(0), rank: 4, alpha: 8)
    assert {4, 512} = Nx.shape(elem(params.layers, 0).attention.q_proj.lora_a)
  end

  test "what the caller leaves out still comes from the environment" do
    Application.put_env(:qwen3_5_finetune, :lora, rank: 2, alpha: 8)

    {params, _key} = Model.init(config(), Nx.Random.key(0), alpha: 4)
    q_proj = elem(params.layers, 0).attention.q_proj

    assert {2, 512} = Nx.shape(q_proj.lora_a)
    assert Nx.to_number(q_proj.scale) == 2.0
  end
end
