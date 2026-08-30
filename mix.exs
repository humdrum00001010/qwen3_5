defmodule Qwen3Finetune.MixProject do
  use Mix.Project

  def project do
    [
      app: :qwen3_finetune,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    worktree =
      System.get_env("NX_MLIR_ATTRIBUTES_WORKTREE") ||
        Path.expand("../nx-operation-attributes-pr", __DIR__)

    unless File.dir?(Path.join(worktree, "nx")) and File.dir?(Path.join(worktree, "exla")) do
      Mix.raise("""
      NX_MLIR_ATTRIBUTES_WORKTREE must point to the Nx worktree containing
      EXLA.CustomCall.Spec.mlir_attributes, got: #{worktree}
      """)
    end

    flash =
      System.get_env("FLASH_ATTENTION3_PATH") || Path.expand("../ex_flashattention3", __DIR__)

    unless File.dir?(flash) do
      Mix.raise("FLASH_ATTENTION3_PATH must point at the ex_flashattention3 checkout")
    end

    [
      {:nx, path: Path.join(worktree, "nx"), override: true},
      {:exla, path: Path.join(worktree, "exla")},
      {:fa3_tp_experiment, path: flash},
      {:polaris, "~> 0.1"}
    ]
  end
end
