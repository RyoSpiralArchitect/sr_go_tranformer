// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// Frozen from 68ee4d8: includes request-owned workers/workspace, so the
// benchmark isolates prompt batching rather than remeasuring worker reuse.
func tokenwiseOwnedGenerateInput(ctx context.Context, m *Model, tok inferenceTokenizer, prompt string, s Sampling, framed bool) (Completion, error) {
	ids, err := prepareInput(ctx, m, tok, prompt, s, framed)
	if err != nil {
		return Completion{}, err
	}
	cache := m.NewCache()
	defer cache.Close()
	// One call owns its workers and scratch, including through sampling. Join
	// workers before releasing KV; no mutable execution state lives on Model.
	executor := newKernelExecutor(generationWorkerCount(m.Config, len(ids), s.MaxTokens))
	defer executor.close()
	var workspace decodeWorkspace
	caches, chunks := []*KVCache{cache}, make([][]int, 1)
	var logits []float32
	for i := range ids {
		if err = ctx.Err(); err != nil {
			return Completion{}, err
		}
		chunks[0] = ids[i : i+1]
		logits, err = m.prefillBorrowed(ctx, caches, chunks, executor, &workspace)
		if err != nil {
			return Completion{}, err
		}
	}
	history := append([]int(nil), ids...)
	out := make([]int, 0, s.MaxTokens)
	rng := &RNG{s.Seed}
	reason := "length"
	for i := 0; i < s.MaxTokens; i++ {
		if err = ctx.Err(); err != nil {
			return Completion{}, err
		}
		id, e := sampleWithBlocked(logits, history, s, rng, tok.blockedID())
		if e != nil {
			return Completion{}, e
		}
		out = append(out, id)
		history = append(history, id)
		if tok.isEOS(id) {
			reason = "eos"
			break
		}
		if i+1 < s.MaxTokens {
			chunks[0] = out[len(out)-1:]
			logits, err = m.prefillBorrowed(ctx, caches, chunks, executor, &workspace)
			if err != nil {
				return Completion{}, err
			}
		}
	}
	return completion(tok, out, len(ids), reason), nil
}

func promptPrefillTestModel(t testing.TB, format string) *Model {
	t.Helper()
	c := Config{Vocab: 258, Dim: 66, Hidden: 257, Layers: 2, Heads: 3, KVHeads: 1, Context: 256, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	m.DropOptimizer()
	for _, p := range m.Params {
		if p.Rows < 2 || format == "f32" {
			continue
		}
		q := quantizeTestParam(t, p)
		if format == "q4" {
			q = quantizeQ4TestParam(t, p)
		}
		*p = *q
	}
	return m
}

func TestPromptPrefillExactCompletionAndKV(t *testing.T) {
	for _, workers := range []int{1, 4} {
		old := runtime.GOMAXPROCS(workers)
		for _, format := range []string{"f32", "q8", "q4"} {
			t.Run(fmt.Sprintf("workers%d/%s", workers, format), func(t *testing.T) {
				m := promptPrefillTestModel(t, format)
				for _, length := range []int{1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129} {
					prompt := strings.Repeat("a", length-1)
					for _, temperature := range []float64{0, .7} {
						s := defaultSampling()
						s.MaxTokens = 8
						s.Temperature = temperature
						s.Seed = 19
						want, err := tokenwiseOwnedGenerateInput(context.Background(), m, Tokenizer{}, prompt, s, false)
						if err != nil {
							t.Fatal(err)
						}
						got, err := generateInput(context.Background(), m, Tokenizer{}, prompt, s, false)
						if err != nil || !reflect.DeepEqual(got, want) {
							t.Fatalf("prompt %d temp%g: completion changed: %v", length, temperature, err)
						}
					}
					// Independently compare every committed K/V row, not just greedy output.
					ids := prefillTokens(length, 5)
					a, b := m.NewCache(), m.NewCache()
					e := newKernelExecutor(workers)
					var w decodeWorkspace
					rows := generationPromptRows(m, len(ids), workers, (16<<20)/4)
					for i := 0; i < len(ids); i += rows {
						end := min(i+rows, len(ids))
						var want []float32
						var err error
						for _, id := range ids[i:end] {
							want, err = m.Step(b, id)
							if err != nil {
								t.Fatal(err)
							}
						}
						got, err := m.prefillBorrowed(context.Background(), []*KVCache{a}, [][]int{ids[i:end]}, e, &w)
						if err != nil {
							t.Fatal(err)
						}
						exactWorkerFloats(t, got, want)
						if a.Pos != b.Pos {
							t.Fatal("position changed")
						}
						assertWorkspaceReferencesCleared(t, &w)
					}
					for i, p := range a.Pages {
						exactWorkerFloats(t, p.Data, b.Pages[i].Data)
					}
					e.close()
					a.Close()
					b.Close()
				}
			})
		}
		runtime.GOMAXPROCS(old)
	}
}

func TestPromptPrefillSoftBudgetAndWorkerPlan(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	m := promptPrefillTestModel(t, "f32")
	if rows, workers := planGenerationPrompt(m, 129, 8); rows != 16 || workers != 4 {
		t.Fatalf("batch not parallel: %d/%d", rows, workers)
	}
	for _, tokens := range []int{1, 2, 3} {
		if got := generationPromptRows(m, tokens, 4, 1<<30); got != 1 {
			t.Fatalf("short prompt changed: %d", got)
		}
	}
	// Geometry-only fixture; no large model allocation. Validate each chosen
	// shape against the actual allocator's independent list of required arrays.
	large := &Model{Config: Config{Vocab: 262144, Dim: 8192, Hidden: 32768, Heads: 4096, KVHeads: 1, Context: 32768}}
	for _, limit := range []int64{0, 1 << 20, 2 << 20, 4 << 20, 8 << 20} {
		rows := generationPromptRows(large, 32768, 4, limit)
		if rows < 1 || rows > 16 {
			t.Fatal("unbounded rows", rows)
		}
		if rows == 1 {
			continue
		} // Soft-target fallback; hard forward admission remains authoritative.
		c := large.Config
		p := planInferenceAttention(&kernelExecutor{workers: 4}, c, rows, 32768)
		lengths := []int{rows * c.Dim, rows * c.Dim, rows * c.Dim, rows * c.KVDim(), rows * c.KVDim(), rows * c.Dim, rows * c.Dim, rows * c.Hidden, rows * c.Hidden, rows * c.Hidden, c.Vocab, p.lanes * p.stride, c.Vocab}
		used := int64(0)
		for _, n := range lengths {
			used += int64(n)
		}
		if used > limit {
			t.Fatalf("rows=%d used=%d budget=%d", rows, used, limit)
		}
	}
	if got := generationPromptRows(large, 32768, 4, 1<<20); got >= 16 {
		t.Fatal("large workspace did not shrink")
	}
}

func BenchmarkPromptPrefillGenerate(b *testing.B) {
	m := ownedGenerateTestModel(b)
	tok := fixedStepGenerateTokenizer{}
	for _, prefix := range []int{3, 33, 129, 257} {
		prompt := strings.Repeat("a", prefix-1)
		s := defaultSampling()
		s.MaxTokens = 1
		s.Temperature = 0
		s.RepeatPenalty = 1
		want, err := tokenwiseOwnedGenerateInput(context.Background(), m, tok, prompt, s, false)
		if err != nil {
			b.Fatal(err)
		}
		for _, reverse := range []bool{false, true} {
			order := []string{"before", "after"}
			if reverse {
				order[0], order[1] = order[1], order[0]
			}
			for _, variant := range order {
				b.Run(fmt.Sprintf("prefix%d/reverse%t/%s", prefix, reverse, variant), func(b *testing.B) {
					fn := generateInput
					if variant == "before" {
						fn = tokenwiseOwnedGenerateInput
					}
					got, err := fn(context.Background(), m, tok, prompt, s, false)
					if err != nil || !reflect.DeepEqual(got, want) {
						b.Fatal("completion changed", err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err := fn(context.Background(), m, tok, prompt, s, false); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

func TestPromptPrefillCancelBeforeSamplingAndRetry(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	m := ownedGenerateTestModel(t)
	prompt := strings.Repeat("a", 128)
	s := defaultSampling()
	s.MaxTokens = 8
	s.Temperature = 0
	s.RepeatPenalty = 1
	want, err := tokenwiseOwnedGenerateInput(context.Background(), m, fixedStepGenerateTokenizer{}, prompt, s, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, at := range []int{1, 4, 6, 9, 15, 24} {
		sampled := 0
		tok := sampledGenerateTokenizer{sampled: &sampled}
		ctx := &prefillCancelContext{Context: context.Background(), cancelAt: at}
		_, err := generateInput(ctx, m, tok, prompt, s, false)
		if !errors.Is(err, context.Canceled) || sampled != 0 {
			t.Fatalf("checkpoint%d: sampled=%d err=%v", at, sampled, err)
		}
		got, err := generateInput(context.Background(), m, fixedStepGenerateTokenizer{}, prompt, s, false)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatal("retry differs", err)
		}
	}
}
