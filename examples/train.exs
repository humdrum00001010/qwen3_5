# A few `Axon.Loop.trainer/3` steps at a size that runs on CPU.
#
#   mix run examples/train.exs
#
# On a GPU, load the kernel before anything compiles and use EXLA:
#
#   :ok = EXLA.load_dylib(System.fetch_env!("FA3_DYLIB"))
#   Nx.Defn.default_options(compiler: EXLA, compiler_options: [client: :cuda])

alias Qwen3_5.Model

# Only the counts are shrunk to run on CPU; every other dimension is the
# default. Against a real checkpoint this is `Model.from_json(path)`.
config = %{
  "num_hidden_layers" => 2,
  "hidden_size" => 128,
  "num_attention_heads" => 2,
  "num_key_value_heads" => 1,
  "intermediate_size" => 256,
  "vocab_size" => 64
}

batch = 2
sequence = 8

# Cross-entropy of the next token, reduced in FP32: a normaliser summed over a
# 151_936-entry vocabulary in BF16 loses most of the distribution.
loss = fn targets, logits ->
  Axon.Losses.categorical_cross_entropy(targets, Nx.as_type(logits, :f32),
    from_logits: true,
    sparse: true,
    reduction: :mean
  )
end

# Batches are `{input, target}`: a window of tokens and the same window shifted
# one to the left, with the trailing axis the sparse loss selects along. Random
# ids here; a corpus would stream windows of its token ids instead.
data =
  Stream.unfold(Nx.Random.key(0), fn key ->
    {tokens, key} =
      Nx.Random.randint(key, 0, config["vocab_size"], shape: {batch, sequence + 1}, type: :u32)

    {{tokens[[.., 0..-2//1]], Nx.new_axis(tokens[[.., 1..-1//1]], -1)}, key}
  end)

IO.puts("ln(vocab) = #{Float.round(:math.log(config["vocab_size"]), 4)}")

Model.new(config)
|> Axon.Loop.trainer(loss, Polaris.Optimizers.adamw(learning_rate: 1.0e-3), log: 1)
|> Axon.Loop.run(data, Axon.ModelState.empty(), iterations: 4, compiler: Nx.Defn.Evaluator)
