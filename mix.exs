defmodule Qwen3_5Finetune.MixProject do
  use Mix.Project

  def project do
    [
      app: :qwen3_5_finetune,
      version: "0.1.0",
      elixir: "~> 1.20.4",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    # The custom-call attributes and `EXLA.load_dylib/1` are merged upstream but
    # unreleased, so Nx and EXLA follow elixir-nx/nx at the commit that contains
    # both; ex_flashattention3 pins the same one.
    nx_ref = "509da1b5e28380bb60bd11d67dc669e4b68231df"

    [
      {:nx, github: "elixir-nx/nx", ref: nx_ref, subdir: "nx", override: true},
      {:exla, github: "elixir-nx/nx", ref: nx_ref, subdir: "exla"},
      {:ex_flashattention3,
       github: "humdrum00001010/ex_flashattention3",
       ref: "df2e1db6fe61d2811a505a5254c0df44f25d2317"},
      # Runs the graph on a Mac, where the kernel cannot. Pinned to main rather
      # than a release because the release predates Nx 0.13. A macOS older than
      # the prebuilt MLX wants `LIBMLX_MACOS_COMPAT=true` in the environment of
      # every `mix` command.
      {:emlx,
       github: "elixir-nx/emlx", ref: "e59f24ec34bca7ff3797ff02fe1d6ec338c3debb", subdir: "emlx"},
      {:polaris, "~> 0.1"},
      {:axon, "~> 0.8"},
      # Fetches datasets and checkpoints from the Hugging Face Hub, with the
      # Hub's cache and revision pinning; Explorer reads what it fetches.
      {:hf_hub, "~> 0.3.1"},
      {:explorer, "~> 0.10"}
    ]
  end
end
