// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// This standalone probe is compiled with inference_perf_test.go to reuse its
// canonical little-endian FP32 digest. It is run separately from timed probes.
func TestMistralPromptBits(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()
	dir := os.Getenv("MONOLITH_BITS_MODEL")
	if dir == "" || runtime.GOMAXPROCS(0) != 4 {
		t.Fatal("requires local weights and four threads")
	}
	m, _, err := loadHFModelFormat(ctx, dir, 1024, 8192, "q4-mixed")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	s := defaultSampling()
	s.MaxTokens, s.Temperature, s.RepeatPenalty, s.Seed = 1, 0, 1, 19
	prompt := strings.Repeat("A black cat lives near the lighthouse. The light guides ships safely home. ", 16) + "The cat"
	ids, err := prepareInput(ctx, m, tok, prompt, s, false)
	if err != nil || len(ids) != 275 {
		t.Fatal("unexpected prompt", len(ids), err)
	}
	cache := m.NewCache()
	defer cache.Close()
	rows, workers := planGenerationPrompt(m, len(ids), s.MaxTokens)
	executor := newKernelExecutor(workers)
	defer executor.close()
	var w decodeWorkspace
	var logits []float32
	for lo := 0; lo < len(ids); lo += rows {
		end := min(lo+rows, len(ids))
		logits, err = m.prefillSelected(ctx, []*KVCache{cache}, [][]int{ids[lo:end]}, executor, &w, []bool{end == len(ids)})
		if err != nil {
			t.Fatal(err)
		}
	}
	if cache.Pos != len(ids) || len(logits) != m.Config.Vocab {
		t.Fatal("incomplete forward")
	}
	id, err := sampleWithBlocked(logits, ids, s, &RNG{s.Seed}, tok.blockedID())
	if err != nil {
		t.Fatal(err)
	}
	lh, kh := sha256.New(), sha256.New()
	perfHashFloats(lh, logits)
	for layer := 0; layer < m.Config.Layers; layer++ {
		for pos := 0; pos < cache.Pos; pos++ {
			k, v := cache.row(layer, pos)
			perfHashFloats(kh, k)
			perfHashFloats(kh, v)
		}
	}
	report := map[string]any{"prompt_ids": ids, "token_id": id, "chunk_rows": rows, "workers": workers,
		"positions": cache.Pos, "logit_values": len(logits), "kv_values": 2 * cache.Pos * m.Config.KVDim() * m.Config.Layers,
		"logits_sha256": hex.EncodeToString(lh.Sum(nil)), "kv_sha256": hex.EncodeToString(kh.Sum(nil))}
	f, err := os.OpenFile(os.Getenv("MONOLITH_BITS_OUTPUT"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(report); err != nil {
		t.Fatal(err)
	}
}
