// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSchedulerTokenBudgetAndDecodeProgress(t *testing.T) {
	sessions := []*engineSession{
		{Request: &engineRequest{IDs: make([]int, 2)}, Cursor: 2},
		{Request: &engineRequest{IDs: make([]int, 3)}, Cursor: 2},
		{Request: &engineRequest{IDs: make([]int, 100)}},
		{Request: &engineRequest{IDs: make([]int, 100)}},
	}
	counts := make([]int, len(sessions))
	extra := make([]int, len(sessions))
	for tick := 0; tick < 12; tick++ {
		total := scheduleTokens(sessions, counts, 5, 4, 4, tick%len(sessions))
		if total != 5 {
			t.Fatalf("budget not filled: %d", total)
		}
		sum := 0
		for i, n := range counts {
			if n < 1 || n > 4 {
				t.Fatalf("session %d count %d", i, n)
			}
			sum += n
			extra[i] += n - 1
		}
		if counts[0] != 1 || counts[1] != 1 || sum != total {
			t.Fatalf("decode or budget invariant: %v", counts)
		}
	}
	if extra[2] == 0 || extra[3] == 0 {
		t.Fatalf("prefill starved: %v", extra)
	}
	total := scheduleTokens(sessions, counts, 1024, 16, 16, 0)
	if total != 34 || counts[2] != 16 || counts[3] != 16 {
		t.Fatalf("chunk limit: total=%d counts=%v", total, counts)
	}
	if total := scheduleTokens(sessions, counts, 64, 16, 1, 0); total != len(sessions) {
		t.Fatalf("mixed decode/prefill tick must stay small: %v", counts)
	}
	sessions[0].Cursor = 0
	if total := scheduleTokens(sessions, counts, 64, 16, 1, 0); total <= len(sessions) {
		t.Fatalf("prefill-only tick should use larger chunks: %v", counts)
	}

}

func TestEngineChunkConfigurationsPreserveOutputAndBudget(t *testing.T) {
	m := testModel(t)
	opts := defaultSampling()
	opts.MaxTokens = 12
	opts.Seed = 42
	prompt := strings.Repeat("cat ", 12)
	want, err := Generate(context.Background(), m, Tokenizer{}, prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range []int{1, 4, 16, 64} {
		t.Run(fmt.Sprintf("chunk_%d", chunk), func(t *testing.T) {
			e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 4, Queue: 4, CacheBytes: 1 << 20, PrefillChunk: chunk, TokenBudget: 8})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			got, err := e.Generate(context.Background(), prompt, opts)
			if err != nil || got.Text != want.Text || fmt.Sprint(got.TokenIDs) != fmt.Sprint(want.TokenIDs) {
				t.Fatalf("output mismatch: %v %v", got, err)
			}
			stats := e.Stats()
			if stats.PeakTickTokens > 8 || stats.PeakTickTokens > int64(chunk) || stats.InputTokens != int64(len(prompt)+1) || stats.GeneratedTokens != int64(len(got.TokenIDs)) {
				t.Fatalf("budget/accounting: %+v", stats)
			}
			if chunk == 1 && stats.PeakTickTokens != 1 {
				t.Fatal("baseline must remain one token per session")
			}
		})
	}
}

func TestMetricsHaveLatencyAndRuntimeEvidence(t *testing.T) {
	e, err := NewEngine(testModel(t), Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	opts := defaultSampling()
	opts.MaxTokens = 8
	got, err := e.Generate(context.Background(), "cat", opts)
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	e.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	values := map[string]float64{}
	for _, line := range strings.Split(w.Body.String(), "\n") {
		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}
		values[parts[0]], err = strconv.ParseFloat(parts[1], 64)
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"monolith_heap_alloc_bytes", "monolith_heap_sys_bytes", "monolith_goroutines", "monolith_first_token_seconds_sum"} {
		if values[name] <= 0 {
			t.Fatalf("missing positive %s: %s", name, w.Body.String())
		}
	}
	if values["monolith_first_token_seconds_count"] != 1 || values["monolith_queue_wait_seconds_count"] != 1 || values["monolith_token_gap_seconds_count"] != float64(len(got.TokenIDs)-1) {
		t.Fatalf("wrong latency counts: %s", w.Body.String())
	}
	if values["monolith_kv_reserved_bytes"] != 0 || values["monolith_active"] != 0 {
		t.Fatal("terminal result preceded KV release")
	}
	for _, prefix := range []string{"monolith_first_token_seconds", "monolith_queue_wait_seconds", "monolith_token_gap_seconds"} {
		previous := float64(0)
		for _, bound := range latencyBounds {
			v := values[fmt.Sprintf("%s_bucket{le=\"%g\"}", prefix, bound)]
			if v < previous || v > values[prefix+"_count"] {
				t.Fatalf("invalid cumulative histogram %s", prefix)
			}
			previous = v
		}
	}
}

func TestConcurrentSubmissionAndShutdownDoesNotStrandAdmission(t *testing.T) {
	for round := 0; round < 8; round++ {
		e, err := NewEngine(testModel(t), Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 16, CacheBytes: 1 << 20})
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		start := make(chan struct{})
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		for i := 0; i < 32; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				opts := defaultSampling()
				opts.MaxTokens = 8
				stream, err := e.Stream(ctx, "cat", opts)
				if err == nil {
					for range stream.Events {
					}
					_, _ = stream.Wait()
				}
			}()
		}
		close(start)
		e.Close()
		wg.Wait()
		cancel()
		if len(e.admission) != 0 || len(e.queue) != 0 || e.Stats().KVReservedBytes != 0 {
			t.Fatalf("stranded request round%d: %+v slots=%d", round, e.Stats(), len(e.admission))
		}
	}
}

func TestHTTPExpiredRequestReportsTimeout(t *testing.T) {
	e, err := NewEngine(testModel(t), Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	r := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":"cat","max_tokens":8,"stream":true}`)).WithContext(ctx)
	w := httptest.NewRecorder()
	e.Handler().ServeHTTP(w, r)
	if w.Code != http.StatusRequestTimeout || !strings.Contains(w.Body.String(), `"code":"timeout"`) {
		t.Fatalf("deadline must be distinguishable from cancellation: %d %s", w.Code, w.Body.String())
	}
}
