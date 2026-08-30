import Config

# What this project chooses, as opposed to what a checkpoint dictates. Model
# dimensions are not here and should not be: they belong to the checkpoint and
# are read from its `config.json` by `Qwen3_5.Config.from_json/1`.
#
# Rank 16 over the four attention projections is a few tens of megabytes
# against Qwen3.5-8B's 16 GB of BF16 weights, which is the whole reason the
# finetune is LoRA and not full. `alpha / rank` is the scale the adapter's
# output is multiplied by, so the two move together.
config :qwen3_5_finetune, :lora,
  rank: 16,
  alpha: 32
