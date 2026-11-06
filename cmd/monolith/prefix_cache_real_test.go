// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Manual offline functionality/equivalence check. A single model stays resident;
// this unbalanced pair is not an adoption benchmark or a language-quality test.
func TestHFPrefixCacheReal(t *testing.T) {
	dir := os.Getenv("MONOLITH_PREFIX_MODEL")
	if dir == "" {
		t.Skip("manual offline prefix-cache probe")
	}
	if runtime.GOMAXPROCS(0) != 4 {
		t.Fatal("requires four Go threads")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()
	m, plan, err := loadHFModelFormat(ctx, dir, 512, 8192, "q4-mixed")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	base := strings.Repeat("The black cat waits by the lighthouse. ", 10)
	prompts := []string{}
	promptIDs := [][]int{}
	for _, tail := range []string{"Describe the weather in one sentence.", "Describe the sea in one sentence."} {
		p, err := tok.formatChat([]ChatMessage{{Role: "user", Content: base + tail}})
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tok.encodeText(ctx, p, 504, false)
		if err != nil {
			t.Fatal(err)
		}
		prompts = append(prompts, p)
		promptIDs = append(promptIDs, ids)
	}
	common := 0
	for common < min(len(promptIDs[0]), len(promptIDs[1])) && promptIDs[0][common] == promptIDs[1][common] {
		common++
	}
	if common < 64 {
		t.Fatal("probe lacks reusable prefix", common)
	}
	s := defaultSampling()
	s.MaxTokens = 8
	s.Temperature = 0
	s.RepeatPenalty = 1
	s.Seed = 19
	type trial struct {
		Completion Completion  `json:"completion"`
		Stats      EngineStats `json:"stats"`
		Seconds    float64     `json:"seconds"`
	}
	reports := map[string][]trial{}
	for _, enabled := range []bool{false, true} {
		cfg := EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: 512 << 20, KernelWorkers: 4}
		name := "disabled"
		if enabled {
			cfg.PrefixCacheBytes = 128 << 20
			name = "enabled"
		}
		e, err := NewEngine(m, tok, cfg)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range prompts {
			start := time.Now()
			result, err := e.generateInput(ctx, p, s, true)
			if err != nil {
				e.Close()
				t.Fatal(err)
			}
			reports[name] = append(reports[name], trial{result, e.Stats(), time.Since(start).Seconds()})
		}
		e.Close()
		if e.Stats().PrefixBytes != 0 || e.Stats().KVReservedBytes != 0 {
			t.Fatal("prefix/reservation survived shutdown")
		}
	}
	for i := range prompts {
		if !reflect.DeepEqual(reports["disabled"][i].Completion, reports["enabled"][i].Completion) {
			t.Fatal("real cached completion differs", i)
		}
	}
	hit := reports["enabled"][1].Stats
	if hit.PrefixHits != 1 || hit.PrefixTokens < int64(64) || hit.KVAllocatedBytes > 512<<20 || hit.PrefixBytes > 128<<20 {
		t.Fatal("reuse/budget gate failed", hit)
	}
	result := map[string]any{"format": "monolith-real-prefix-cache-v1", "plan": plan, "threads": 4, "prompt_ids": promptIDs, "sampling": s, "common_prefix_tokens": common, "reports": reports, "exact_completions": true, "shutdown_released_prefix": true, "note": "Single unbalanced two-request pair per mode; functionality/parity only. Model source hash verification is the caller's responsibility."}
	f, err := os.OpenFile(os.Getenv("MONOLITH_PREFIX_OUTPUT"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.NewEncoder(f).Encode(result); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
}
