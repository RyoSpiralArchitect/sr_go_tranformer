#!/bin/sh
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
# Exercise actual CLI checkpoint/tokenizer/resume boundaries without downloads.
set -eu
cd "$(dirname "$0")/.."
smoke_dir=${MONOLITH_SMOKE_DIR:-runs/cli-smoke}
mkdir -p "$smoke_dir"
CGO_ENABLED=0 ./scripts/go.sh build -trimpath -o "$smoke_dir/monolith" .
"$smoke_dir/monolith" tokenizer -data examples/tiny.txt -vocab 280 \
  -out "$smoke_dir/tokenizer.json" 2> "$smoke_dir/tokenizer.log"
"$smoke_dir/monolith" train -data examples/tiny.txt -preset demo \
  -tokenizer "$smoke_dir/tokenizer.json" -seq 16 -batch 2 -accum 2 \
  -steps 12 -warmup 2 -seed 9 -eval-every 0 -save-every 0 -eval-batches 2 \
  -out "$smoke_dir/full.mglm" 2> "$smoke_dir/full.log"
"$smoke_dir/monolith" train -data examples/tiny.txt -preset demo \
  -tokenizer "$smoke_dir/tokenizer.json" -seq 16 -batch 2 -accum 2 \
  -steps 12 -warmup 2 -seed 9 -eval-every 0 -save-every 0 -eval-batches 2 \
  -stop-after 5 -out "$smoke_dir/resumed.mglm" 2> "$smoke_dir/partial.log"
"$smoke_dir/monolith" train -data examples/tiny.txt \
  -resume "$smoke_dir/resumed.mglm" -out "$smoke_dir/resumed.mglm" \
  -eval-every 0 -save-every 0 -eval-batches 2 2> "$smoke_dir/resume.log"
cmp "$smoke_dir/full.mglm" "$smoke_dir/resumed.mglm"
"$smoke_dir/monolith" eval -model "$smoke_dir/full.mglm" \
  -data examples/tiny.txt -seq 16 > "$smoke_dir/eval.json"
"$smoke_dir/monolith" generate -model "$smoke_dir/full.mglm" \
  -prompt 'the ' -max-tokens 8 -temperature 0 -json > "$smoke_dir/generated.json"
"$smoke_dir/monolith" inspect -model "$smoke_dir/full.mglm" > "$smoke_dir/inspect.json"
# Synthetic fixture only: this validates mechanics, not held-out model quality.
dataset_tmp=$(mktemp -d "$smoke_dir/dataset-XXXXXX")
trap 'rm -rf "$dataset_tmp"' EXIT HUP INT TERM
sed -n '1,96p' examples/tiny.txt > "$dataset_tmp/train.txt"
sed -n '97,128p' examples/tiny.txt > "$dataset_tmp/validation.txt"
"$smoke_dir/monolith" prepare -train "$dataset_tmp/train.txt" \
  -validation "$dataset_tmp/validation.txt" -tokenizer "$smoke_dir/tokenizer.json" \
  -out "$dataset_tmp/tokens" -shard-tokens 128
"$smoke_dir/monolith" train -dataset "$dataset_tmp/tokens/manifest.json" \
  -preset demo -seq 16 -batch 2 -accum 2 -steps 8 -warmup 2 -seed 9 \
  -data-workers 1 -prefetch 1 -eval-every 0 -save-every 0 -eval-batches 2 \
  -out "$smoke_dir/dataset-full.mglm" 2> "$smoke_dir/dataset-full.log"
"$smoke_dir/monolith" train -dataset "$dataset_tmp/tokens/manifest.json" \
  -preset demo -seq 16 -batch 2 -accum 2 -steps 8 -warmup 2 -seed 9 \
  -data-workers 2 -prefetch 3 -stop-after 3 -eval-every 0 -save-every 0 -eval-batches 2 \
  -out "$smoke_dir/dataset-resumed.mglm" 2> "$smoke_dir/dataset-partial.log"
"$smoke_dir/monolith" train -dataset "$dataset_tmp/tokens/manifest.json" \
  -resume "$smoke_dir/dataset-resumed.mglm" -out "$smoke_dir/dataset-resumed.mglm" \
  -data-workers 3 -prefetch 5 -eval-every 0 -save-every 0 -eval-batches 2 \
  -metrics "$dataset_tmp/train.jsonl" -cpu-profile "$dataset_tmp/cpu.pprof" \
  -alloc-profile "$dataset_tmp/alloc.pprof" \
  2> "$smoke_dir/dataset-resume.log"
cmp "$smoke_dir/dataset-full.mglm" "$smoke_dir/dataset-resumed.mglm"
./scripts/go.sh tool pprof -top "$dataset_tmp/cpu.pprof" > "$smoke_dir/cpu-profile.txt"
./scripts/go.sh tool pprof -top -sample_index=alloc_space "$dataset_tmp/alloc.pprof" \
  > "$smoke_dir/alloc-profile.txt"
"$smoke_dir/monolith" eval -model "$smoke_dir/dataset-full.mglm" \
  -dataset "$dataset_tmp/tokens/manifest.json" -seq 16 -batches 2 \
  > "$smoke_dir/dataset-eval.json"
"$smoke_dir/monolith" eval -model "$smoke_dir/dataset-resumed.mglm" \
  -dataset "$dataset_tmp/tokens/manifest.json" -seq 16 -batches 2 \
  > "$smoke_dir/dataset-eval-resumed.json"
cmp "$smoke_dir/dataset-eval.json" "$smoke_dir/dataset-eval-resumed.json"
printf '%s\n' 'PASS: CLI BPE/text/dataset training, exact resume across IO/telemetry settings, evaluation, pprof, generation, inspection.'
