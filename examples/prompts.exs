# The prompt sets for finding and choosing a refusal direction, fetched from
# the Hub at pinned revisions and stored under `data/`.
#
#   mix run examples/prompts.exs
#
# Heretic's split: the first 400 of each training set find the direction, the
# first 100 of each test set score the candidates, and JailbreakBench's 100
# behaviours stay untouched as the benchmark. The paper keeps every set
# disjoint from the benchmark, so anything that also appears there is dropped
# before the split is taken.
#
# What lands on disk:
#
#   data/hub/       the Hub's own cache layout, raw files at their revisions
#   data/prompts/   direction.csv, selection.csv, benchmark.csv — one row per
#                   prompt, `label` harmful or harmless, `text` the instruction
#
# CSV rather than one prompt per line because a few hundred Alpaca prompts
# contain newlines. Explorer reads them back with `from_csv!/1`.

alias Explorer.DataFrame, as: DF
alias Explorer.Series

# hf_hub reads its cache directory from the application environment only, so
# this is what puts the Hub files under `data/hub` rather than `~/.cache`.
Application.put_env(:hf_hub, :cache_dir, Path.expand("data"))
File.mkdir_p!("data/prompts")

snapshot = fn repo, revision, patterns ->
  {:ok, dir} =
    HfHub.Download.snapshot_download(
      repo_id: repo,
      repo_type: :dataset,
      revision: revision,
      allow_patterns: patterns
    )

  dir
end

column = fn df, name -> df |> DF.pull(name) |> Series.to_list() end

# One column, `text`, one instruction per row, in a train and a test split.
splits = fn repo, revision ->
  dir = snapshot.(repo, revision, ["data/*.parquet"])

  Map.new([:train, :test], fn split ->
    {split, dir |> Path.join("data/#{split}-00000-of-00001.parquet") |> DF.from_parquet!() |> column.("text")}
  end)
end

harmful = splits.("mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7")
harmless = splits.("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f")

# The benchmark: JailbreakBench's harmful goals and their benign counterparts.
jbb = snapshot.("JailbreakBench/JBB-Behaviors", "886acc352a31533ffbcf4ef22c744658688086fc", ["data/*.csv"])
goals = fn split -> jbb |> Path.join("data/#{split}-behaviors.csv") |> DF.from_csv!() |> column.("Goal") end
benchmark = %{harmful: goals.("harmful"), harmless: goals.("benign")}

# Exact-match overlap with the benchmark, dropped before the split is taken.
held_out = MapSet.new(benchmark.harmful ++ benchmark.harmless)
disjoint = fn prompts -> Enum.reject(prompts, &MapSet.member?(held_out, &1)) end
overlap = fn prompts -> length(prompts) - length(disjoint.(prompts)) end

direction = %{
  harmful: harmful.train |> disjoint.() |> Enum.take(400),
  harmless: harmless.train |> disjoint.() |> Enum.take(400)
}

selection = %{
  harmful: harmful.test |> disjoint.() |> Enum.take(100),
  harmless: harmless.test |> disjoint.() |> Enum.take(100)
}

# ## Storing

frame = fn %{harmful: harmful, harmless: harmless} ->
  DF.new(
    label: List.duplicate("harmful", length(harmful)) ++ List.duplicate("harmless", length(harmless)),
    text: harmful ++ harmless
  )
end

store = fn name, sets ->
  path = "data/prompts/#{name}.csv"
  DF.to_csv!(frame.(sets), path)

  # Read back and compare, so a prompt with a newline in it cannot round-trip
  # wrongly without this script saying so.
  stored = DF.from_csv!(path)

  for label <- [:harmful, :harmless] do
    from_disk = stored |> DF.filter_with(&Series.equal(&1["label"], Atom.to_string(label))) |> column.("text")

    if from_disk != sets[label] do
      raise "#{path}: #{label} prompts did not round-trip"
    end
  end

  path
end

mean_length = fn prompts -> Float.round(Enum.sum(Enum.map(prompts, &String.length/1)) / length(prompts), 1) end

IO.puts("harmful_behaviors: #{length(harmful.train)} train, #{length(harmful.test)} test, " <>
  "#{overlap.(harmful.train ++ harmful.test)} also in the benchmark and dropped")

IO.puts("harmless_alpaca:   #{length(harmless.train)} train, #{length(harmless.test)} test, " <>
  "#{overlap.(harmless.train ++ harmless.test)} also in the benchmark and dropped\n")

for {name, sets} <- [direction: direction, selection: selection, benchmark: benchmark] do
  path = store.(name, sets)

  IO.puts(
    "#{String.pad_trailing(path, 28)} #{length(sets.harmful)} harmful (#{mean_length.(sets.harmful)} chars), " <>
      "#{length(sets.harmless)} harmless (#{mean_length.(sets.harmless)} chars)"
  )
end

IO.puts("\nfirst of each direction set:")
IO.puts("  harmful:  #{hd(direction.harmful)}")
IO.puts("  harmless: #{hd(direction.harmless)}")
