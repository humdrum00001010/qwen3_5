# Single training step at a size that runs on CPU.
#
#   mix run examples/train.exs
#
# On a GPU, load the kernel first and drop the compiler override:
#
#   :ok = EXLA.load_dylib(System.fetch_env!("FA3_DYLIB"))
#   Nx.Defn.default_options(compiler: EXLA, compiler_options: [client: :cuda])

alias Qwen3_5.{Config, LoRA, Model, Training}

# Metadata shaped like a checkpoint's, shrunk to run on CPU. `head_dim` stays
# 128 and the dtype stays BF16, since those are the two the kernel constrains.
# Against a real checkpoint this line is `Config.from_json(path)` instead.
config =
  Config.from_metadata(%{
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
  })

sequence = 8

{params, key} = Model.init(config, Nx.Random.key(0))
{cos, sin} = Training.rope_table(config, sequence)
{tokens, _key} = Nx.Random.randint(key, 0, config.vocab, shape: {1, sequence})

adapters = LoRA.trainable(params)
IO.puts("trainable tensors: #{map_size(adapters)}")

IO.puts("trainable elements: #{Enum.reduce(adapters, 0, fn {_, t}, acc -> acc + Nx.size(t) end)}")

{init_fn, update_fn} = Polaris.Optimizers.adamw(learning_rate: 1.0e-4)
state = init_fn.(params)

{loss, _params, _state} =
  Nx.Defn.jit_apply(
    &Training.step/4,
    [params, state, tokens, [config: config, cos: cos, sin: sin, update: update_fn]],
    compiler: Nx.Defn.Evaluator
  )

IO.puts("loss: #{Nx.to_number(loss)}  (ln vocab = #{Float.round(:math.log(config.vocab), 4)})")
