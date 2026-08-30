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
    [
      {:nx,
       github: "humdrum00001010/nx",
       ref: "62ecad529e24ae534eb4082e67153b0b3e12db73",
       subdir: "nx",
       override: true},
      {:exla,
       github: "humdrum00001010/nx",
       ref: "62ecad529e24ae534eb4082e67153b0b3e12db73",
       subdir: "exla"},
      {:ex_flashattention3,
       github: "humdrum00001010/ex_flashattention3",
       ref: "cabd23211cf304002065ad8ffc46e14e1eadac2c"},
      {:polaris, "~> 0.1"}
    ]
  end
end
