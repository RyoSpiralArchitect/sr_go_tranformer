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
printf '%s\n' 'PASS: CLI training, BPE, byte-identical resume, evaluation, generation, inspection.'
