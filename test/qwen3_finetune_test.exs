defmodule Qwen3FinetuneTest do
  use ExUnit.Case
  doctest Qwen3Finetune

  test "greets the world" do
    assert Qwen3Finetune.hello() == :world
  end
end
