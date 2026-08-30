# Single training step at a size that runs on CPU.
#
#   mix run examples/train.exs
#
# On a GPU, load the kernel first and drop the compiler override:
#
#   :ok = EXLA.NIF.load_dylib(System.fetch_env!("FA3_DYLIB"))
#   Nx.Defn.default_options(compiler: EXLA, compiler_options: [client: :cuda])

alias Qwen3.{Config, LoRA, Model, Training}

config = Config.tiny()
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
