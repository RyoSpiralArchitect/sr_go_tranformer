// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

// A bounded, opt-in core-forward benchmark. The Python driver builds this same
// probe against two runtime sources and checks every logit/KV bit and token ID.
import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"hash"
	"math"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func perfHashFloats(h hash.Hash, x []float32) {
	var block [4096]byte
	for len(x) != 0 {
		n := min(len(x), len(block)/4)
		for i, v := range x[:n] {
			binary.LittleEndian.PutUint32(block[i*4:], math.Float32bits(v))
		}
		h.Write(block[:n*4])
		x = x[n:]
	}
}

type inferencePerfTrial struct {
	PrefillSeconds    float64   `json:"prefill_seconds"`
	DecodeSeconds     float64   `json:"decode_seconds"`
	DecodeStepSeconds []float64 `json:"decode_step_seconds"`
	AllocatedBytes    uint64    `json:"allocated_bytes"`
	Allocations       uint64    `json:"allocations"`
	WorkspaceBytes    int64     `json:"workspace_bytes"`
	LogitsSHA256      string    `json:"logits_sha256"`
	KVSHA256          string    `json:"kv_sha256"`
	TokenIDs          [][]int   `json:"token_ids"`
}

func perfCaseOrder(value string) ([]int, error) {
	switch value {
	case "", "1,2":
		return []int{1, 2}, nil
	case "2,1":
		return []int{2, 1}, nil
	case "1":
		return []int{1}, nil
	case "2":
		return []int{2}, nil
	default:
		return nil, errors.New("case order must be 1,2; 2,1; 1; or 2")
	}
}

func TestHFInferencePerformance(t *testing.T) {
	dir := os.Getenv("MONOLITH_PERF_MODEL")
	if dir == "" {
		t.Skip("manual offline performance probe")
	}
	if runtime.GOMAXPROCS(0) != 4 {
		t.Fatal("probe requires GOMAXPROCS=4")
	}
	order, err := perfCaseOrder(os.Getenv("MONOLITH_PERF_CASE_ORDER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	m, plan, err := loadHFModelFormat(ctx, dir, 128, 8192, "q4-mixed")
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
	corpusBytes, err := os.ReadFile(os.Getenv("MONOLITH_PERF_CORPUS"))
	if err != nil || len(corpusBytes) > 64<<10 {
		t.Fatal("invalid corpus", err)
	}
	if err = json.Unmarshal(corpusBytes, &corpus); err != nil || len(corpus.Documents) < 2 {
		t.Fatal("invalid corpus", err)
	}
	prompts := make([][]int, 2)
	for i := range prompts {
		ids, err := tok.encodePrompt(ctx, corpus.Documents[i].Text, 128)
		if err != nil || len(ids) < 8 {
			t.Fatal("invalid prompt", err)
		}
		prompts[i] = append([]int(nil), ids[:8]...)
	}
	e := newKernelExecutor(4)
	defer e.close()
	type perfCase struct {
		Sessions  int                  `json:"sessions"`
		PromptIDs [][]int              `json:"prompt_ids"`
		Trials    []inferencePerfTrial `json:"trials"`
	}
	cases := []perfCase{}
	for _, sessions := range order {
		var workspace decodeWorkspace
		// Each case reuses one bounded workspace; each trial starts fresh KV.
		run := func() inferencePerfTrial {
			caches, chunks := make([]*KVCache, sessions), make([][]int, sessions)
			for i := range caches {
				caches[i] = m.NewCache()
				chunks[i] = prompts[i]
				defer caches[i].Close()
			}
			result := inferencePerfTrial{TokenIDs: make([][]int, sessions), DecodeStepSeconds: make([]float64, 0, 7)}
			logitHash, kvHash := sha256.New(), sha256.New()
			// Only forward calls are timed. Hashing, greedy selection and stats
			// stay outside those intervals. Allocation totals cover the loop.
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			for step := 0; step < 8; step++ {
				if step > 0 {
					for i := range chunks {
						chunks[i] = result.TokenIDs[i][step-1 : step]
					}
				}
				started := time.Now()
				logits, err := m.prefillBorrowed(ctx, caches, chunks, e, &workspace)
				elapsed := time.Since(started).Seconds()
				if err != nil {
					t.Fatal(err)
				}
				if step == 0 {
					result.PrefillSeconds = elapsed
				} else {
					result.DecodeSeconds += elapsed
					result.DecodeStepSeconds = append(result.DecodeStepSeconds, elapsed)
				}
				perfHashFloats(logitHash, logits)
				for i := range caches {
					row := logits[i*m.Config.Vocab : (i+1)*m.Config.Vocab]
					best := 0
					for j, v := range row {
						if !finite(float64(v)) {
							t.Fatal("nonfinite logits")
						}
						if v > row[best] {
							best = j
						}
					}
					// This fixed-step core probe feeds every selected ID, including
					// EOS if one appears. It does not model request termination.
					result.TokenIDs[i] = append(result.TokenIDs[i], best)
				}
			}
			runtime.ReadMemStats(&after)
			result.AllocatedBytes, result.Allocations = after.TotalAlloc-before.TotalAlloc, after.Mallocs-before.Mallocs
			for _, cache := range caches {
				if cache.Pos != 15 {
					t.Fatal("unexpected committed KV position", cache.Pos)
				}
				for layer := 0; layer < m.Config.Layers; layer++ {
					for pos := 0; pos < cache.Pos; pos++ {
						k, v := cache.row(layer, pos)
						perfHashFloats(kvHash, k)
						perfHashFloats(kvHash, v)
					}
				}
			}
			result.LogitsSHA256, result.KVSHA256 = hex.EncodeToString(logitHash.Sum(nil)), hex.EncodeToString(kvHash.Sum(nil))
			result.WorkspaceBytes = workspace.floatBytes()
			return result
		}
		warmup := run()
		runtime.GC()
		c := perfCase{Sessions: sessions, PromptIDs: prompts[:sessions]}
		for trial := 0; trial < 3; trial++ {
			r := run()
			if r.LogitsSHA256 != warmup.LogitsSHA256 || r.KVSHA256 != warmup.KVSHA256 || !reflect.DeepEqual(r.TokenIDs, warmup.TokenIDs) {
				t.Fatal("trial changed numerical results")
			}
			c.Trials = append(c.Trials, r)
		}
		cases = append(cases, c)
	}
	report := map[string]any{"format": "monolith-core-performance-v2", "plan": plan, "cases": cases, "case_order": order,
		"threads": 4, "kernel_workers": 4, "context": 128, "prefill_tokens_per_session": 8,
		"decode_steps": 7, "generated_ids_per_session": 8, "warmup_runs_per_case": 1,
		"note": "Core forward intervals only: excludes loading, tokenization, hashing, sampling, HTTP and SSE. Fixed-step probe does not stop at EOS."}
	b, err := json.MarshalIndent(report, "", "  ")
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

func TestPerfCaseOrder(t *testing.T) {
	for _, s := range []string{"", "1,2", "2,1", "1", "2"} {
		got, err := perfCaseOrder(s)
		want := s
		if want == "" {
			want = "1,2"
		}
		var names []string
		for _, n := range got {
			names = append(names, strconv.Itoa(n))
		}
		if err != nil || strings.Join(names, ",") != want {
			t.Fatalf("case order %q: %v, %v", s, got, err)
		}
	}
	for _, s := range []string{"0", "3", "1,1", "2,2", "2,1,2", "1, 2", "garbage"} {
		if _, err := perfCaseOrder(s); err == nil {
			t.Fatalf("accepted invalid case order %q", s)
		}
	}
}
