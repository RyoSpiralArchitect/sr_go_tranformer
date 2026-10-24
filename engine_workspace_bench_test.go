// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"strings"
	"testing"
)

// This benchmark also compiles unchanged at the parent commit for a matched
// before/after measurement of the entire engine request, including sampling.
func BenchmarkEngineWorkspaceRequest(b *testing.B) {
	m := workerTestModel(b, 2)
	m.DropOptimizer()
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{
		MaxBatch: 1, Queue: 1, CacheBytes: 8 << 20, KernelWorkers: 4,
		PrefillChunk: 16, MixedPrefillChunk: 8, TokenBudget: 32,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	s := defaultSampling()
	s.MaxTokens, s.Temperature, s.RepeatPenalty = 8, 0, 1
	prompt := strings.Repeat("the ", 8)
	run := func() {
		out, err := e.Generate(context.Background(), prompt, s)
		if err != nil || out.CompletionTokens != s.MaxTokens {
			b.Fatalf("unexpected completion: %+v, %v", out, err)
		}
	}
	run()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run()
	}
	b.StopTimer()
}
