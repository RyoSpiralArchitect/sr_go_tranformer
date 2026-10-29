// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

type longGenerationTrial struct {
	PrefillSeconds     float64   `json:"prefill_seconds"`
	DecodeSeconds      float64   `json:"decode_seconds"`
	DecodeStepSeconds  []float64 `json:"decode_step_seconds"`
	TokenIDs           []int     `json:"token_ids"`
	LogitsSHA256       string    `json:"logits_sha256"`
	KVSHA256           string    `json:"kv_sha256"`
	KVPosition         int       `json:"kv_position"`
	KVBytes            int64     `json:"kv_bytes"`
	WorkspaceBytes     int64     `json:"workspace_bytes"`
	ScoreBytes         int       `json:"score_bytes"`
	ScoreReallocations int       `json:"score_reallocations"`
	AllocatedBytes     uint64    `json:"allocated_bytes"`
	Allocations        uint64    `json:"allocations"`
}

func TestHFLongGenerationPerformance(t *testing.T) {
	dir := os.Getenv("MONOLITH_PERF_MODEL")
	if dir == "" {
		t.Skip("manual offline long-generation probe")
	}
	if runtime.GOMAXPROCS(0) != 4 {
		t.Fatal("requires four Go threads")
	}
	order := strings.Split(os.Getenv("MONOLITH_LONG_CASE_ORDER"), ",")
	if strings.Join(order, ",") != "short,long" && strings.Join(order, ",") != "long,short" {
		t.Fatal("invalid case order")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 18*time.Minute)
	defer cancel()
	m, plan, err := loadHFModelFormat(ctx, dir, 1024, 8192, "q4-mixed")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	var corpus struct {
		Documents []struct {
			Text string `json:"text"`
		} `json:"evaluation_documents"`
	}
	b, err := os.ReadFile(os.Getenv("MONOLITH_PERF_CORPUS"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &corpus); err != nil || len(corpus.Documents) < 1 {
		t.Fatal("invalid corpus", err)
	}
	var ids []int
	text := ""
	for len(ids) < 256 {
		text += corpus.Documents[0].Text + "\n"
		ids, err = tok.encodePrompt(ctx, text, 1024)
		if err != nil {
			t.Fatal(err)
		}
	}
	e := newKernelExecutor(4)
	defer e.close()
	type caseResult struct {
		Name      string                `json:"name"`
		PromptIDs []int                 `json:"prompt_ids"`
		OutputIDs int                   `json:"output_ids"`
		Trials    []longGenerationTrial `json:"trials"`
	}
	cases := []caseResult{}
	for _, name := range order {
		prefix, outputs, trials := 8, 8, 3
		if name == "long" {
			prefix, outputs, trials = 256, 256, 1
		}
		var w decodeWorkspace
		run := func() longGenerationTrial {
			cache := m.NewCache()
			defer cache.Close()
			r := longGenerationTrial{TokenIDs: make([]int, 0, outputs), DecodeStepSeconds: make([]float64, 0, outputs-1)}
			logitsHash, kvHash := sha256.New(), sha256.New()
			var logits []float32
			chunks := make([][]int, 1)
			caches := []*KVCache{cache}
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			for offset := 0; offset < prefix; offset += 16 {
				chunks[0] = ids[offset:min(offset+16, prefix)]
				started := time.Now()
				logits, err = m.prefillBorrowed(ctx, caches, chunks, e, &w)
				r.PrefillSeconds += time.Since(started).Seconds()
				if err != nil {
					t.Fatal(err)
				}
				perfHashFloats(logitsHash, logits)
			}
			var scorePointer *float32
			if len(w.Scores) > 0 {
				scorePointer = &w.Scores[0]
			}
			for step := 0; step < outputs; step++ {
				if step > 0 {
					chunks[0] = r.TokenIDs[step-1 : step]
					started := time.Now()
					logits, err = m.prefillBorrowed(ctx, caches, chunks, e, &w)
					elapsed := time.Since(started).Seconds()
					if err != nil {
						t.Fatal(err)
					}
					r.DecodeSeconds += elapsed
					r.DecodeStepSeconds = append(r.DecodeStepSeconds, elapsed)
					perfHashFloats(logitsHash, logits)
					if &w.Scores[0] != scorePointer {
						r.ScoreReallocations++
						scorePointer = &w.Scores[0]
					}
				}
				best := 0
				for i, v := range logits {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite logits")
					}
					if v > logits[best] {
						best = i
					}
				}
				r.TokenIDs = append(r.TokenIDs, best)
				if name == "long" && (step+1)%64 == 0 {
					t.Logf("long decode completed %d output IDs, KV position %d", step+1, cache.Pos)
				}
			}
			runtime.ReadMemStats(&after)
			r.AllocatedBytes, r.Allocations = after.TotalAlloc-before.TotalAlloc, after.Mallocs-before.Mallocs
			if cache.Pos != prefix+outputs-1 {
				t.Fatal("unexpected committed position", cache.Pos)
			}
			for layer := range m.Blocks {
				for pos := 0; pos < cache.Pos; pos++ {
					k, v := cache.row(layer, pos)
					perfHashFloats(kvHash, k)
					perfHashFloats(kvHash, v)
				}
			}
			r.LogitsSHA256, r.KVSHA256 = hex.EncodeToString(logitsHash.Sum(nil)), hex.EncodeToString(kvHash.Sum(nil))
			r.KVPosition = cache.Pos
			for _, page := range cache.Pages {
				r.KVBytes += int64(len(page.Data)) * 4
			}
			r.WorkspaceBytes, r.ScoreBytes = w.floatBytes(), cap(w.Scores)*4
			return r
		}
		if name == "short" {
			_ = run()
			runtime.GC()
		}
		c := caseResult{Name: name, PromptIDs: ids[:prefix], OutputIDs: outputs}
		for trial := 0; trial < trials; trial++ {
			c.Trials = append(c.Trials, run())
		}
		cases = append(cases, c)
	}
	report := map[string]any{"format": "monolith-long-generation-v1", "plan": plan, "threads": 4, "kernel_workers": 4, "context": 1024, "prefill_chunk": 16, "case_order": order, "cases": cases, "note": "Core forward only. 255 long decode calls after a 256-token prefix; fixed-step continues at EOS. Short control has one warmup and three trials. No language-quality or HTTP throughput claim."}
	b, err = json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(os.Getenv("MONOLITH_PERF_OUTPUT"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.Write(append(b, '\n')); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
}
