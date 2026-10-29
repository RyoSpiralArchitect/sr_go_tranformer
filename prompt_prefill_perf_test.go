// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestHFPromptStackProbe(t *testing.T) {
	dir := os.Getenv("MONOLITH_STACK_MODEL")
	if dir == "" {
		t.Skip("manual offline probe")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()
	m, plan, err := loadHFModelFormat(ctx, dir, 1024, 8192, "q4-mixed")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	s := defaultSampling()
	s.MaxTokens = 1
	s.Temperature = 0
	s.RepeatPenalty = 1
	s.Seed = 19
	type trial struct {
		Completion Completion `json:"completion"`
		Seconds    float64    `json:"seconds"`
		Allocated  uint64     `json:"allocated_bytes"`
		Allocs     uint64     `json:"allocations"`
	}
	type result struct {
		Name   string  `json:"name"`
		Trials []trial `json:"trials"`
	}
	cases := []result{}
	order := strings.Split(os.Getenv("MONOLITH_STACK_ORDER"), ",")
	if len(order) != 2 {
		t.Fatal("order")
	}
	for _, name := range order {
		prompt := "The black cat"
		if name == "long" {
			prompt = strings.Repeat("A black cat lives near the lighthouse. The light guides ships safely home. ", 16) + "The cat"
		} else if name != "short" {
			t.Fatal(name)
		}
		r := result{Name: name}
		count := 1
		if name == "short" {
			count = 3
		}
		for i := 0; i < count; i++ {
			runtime.GC()
			var a, b runtime.MemStats
			runtime.ReadMemStats(&a)
			start := time.Now()
			c, e := Generate(ctx, m, tok, prompt, s)
			elapsed := time.Since(start).Seconds()
			runtime.ReadMemStats(&b)
			if e != nil {
				t.Fatal(e)
			}
			if c.CompletionTokens != 1 {
				t.Fatal("incomplete generation")
			}
			r.Trials = append(r.Trials, trial{c, elapsed, b.TotalAlloc - a.TotalAlloc, b.Mallocs - a.Mallocs})
			t.Log(name, i, c.PromptTokens, elapsed)
		}
		cases = append(cases, r)
	}
	out := map[string]any{"plan": plan, "cases": cases, "threads": runtime.GOMAXPROCS(0), "note": "One output token measures direct Generate first-token latency excluding loading. Fixed prompts and greedy sampling."}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(os.Getenv("MONOLITH_STACK_OUTPUT"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err = f.Write(b); err != nil {
		t.Fatal(err)
	}
}
