// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestSelectiveHeadMixedRowsAndKV(t *testing.T) {
	for _, format := range []string{"f32", "q8", "q4"} {
		for _, workers := range []int{1, 4} {
			t.Run(fmt.Sprintf("%s/workers%d", format, workers), func(t *testing.T) {
				m := promptPrefillTestModel(t, format)
				e := newKernelExecutor(workers)
				defer e.close()
				// Every selection for five sessions exercises compaction across the
				// four/two/single-row projection paths, including holes and no head.
				for bits := 0; bits < 32; bits++ {
					a, b := make([]*KVCache, 5), make([]*KVCache, 5)
					selected := make([]bool, 5)
					for i := range a {
						a[i], b[i] = m.NewCache(), m.NewCache()
						selected[i] = bits&(1<<i) != 0
					}
					var wa, wb decodeWorkspace
					for step := 0; step < 3; step++ {
						chunks := make([][]int, 5)
						for i := range chunks {
							chunks[i] = prefillTokens([]int{16, 1, 3, 17, 31}[i], i+step)
						}
						want, err := m.prefillBorrowed(context.Background(), b, chunks, e, &wb)
						if err != nil {
							t.Fatal(err)
						}
						mask := selected
						if step == 2 {
							mask = nil
						} // Re-enable all heads in dirty scratch.
						got, err := m.prefillSelected(context.Background(), a, chunks, e, &wa, mask)
						if err != nil {
							t.Fatal(err)
						}
						for i := range a {
							row := got[i*m.Config.Vocab : (i+1)*m.Config.Vocab]
							if mask == nil || mask[i] {
								exactWorkerFloats(t, row, want[i*m.Config.Vocab:(i+1)*m.Config.Vocab])
							} else {
								for _, v := range row {
									if v != 0 {
										t.Fatal("stale unselected logits")
									}
								}
							}
							if a[i].Pos != b[i].Pos {
								t.Fatal("position changed")
							}
							for j, p := range a[i].Pages {
								exactWorkerFloats(t, p.Data, b[i].Pages[j].Data)
							}
						}
						assertWorkspaceReferencesCleared(t, &wa)
					}
					for i := range a {
						a[i].Close()
						b[i].Close()
					}
				}
			})
		}
	}
}

func TestSelectiveHeadSkipAndValidation(t *testing.T) {
	m := promptPrefillTestModel(t, "f32")
	c := m.NewCache()
	defer c.Close()
	var w decodeWorkspace
	if _, err := m.prefillSelected(context.Background(), []*KVCache{c}, [][]int{{2}}, nil, &w, []bool{}); err == nil || c.Pos != 0 || len(c.Pages) != 0 || w.floatBytes() != 0 {
		t.Fatal("invalid selection mutated cache/workspace", err)
	}
	// A skipped head must not even read the final norm or output parameter.
	final := m.Final
	m.Final = nil
	got, err := m.prefillSelected(context.Background(), []*KVCache{c}, [][]int{{2, 3, 4}}, nil, &w, []bool{false})
	m.Final = final
	if err != nil || c.Pos != 3 {
		t.Fatal("headless prefill failed", err)
	}
	for _, v := range got {
		if v != 0 {
			t.Fatal("headless result is not empty")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.prefillSelected(ctx, []*KVCache{c}, [][]int{{5}}, nil, &w, []bool{true}); err != context.Canceled || c.Pos != 3 {
		t.Fatal("canceled call committed", err)
	}
}

func TestSelectiveHeadGenerationParity(t *testing.T) {
	for _, format := range []string{"f32", "q8", "q4"} {
		m := promptPrefillTestModel(t, format)
		for _, length := range []int{1, 3, 4, 16, 17, 32, 33, 129} {
			for _, temperature := range []float64{0, .8} {
				s := defaultSampling()
				s.MaxTokens = 8
				s.Temperature = temperature
				prompt := strings.Repeat("a", length-1)
				want, err := allHeadGenerateInput(context.Background(), m, Tokenizer{}, prompt, s, false)
				if err != nil {
					t.Fatal(err)
				}
				got, err := generateInput(context.Background(), m, Tokenizer{}, prompt, s, false)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("%s length%d temp%g: %v", format, length, temperature, err)
				}
			}
		}
	}
}

func BenchmarkSelectiveHeadGenerate(b *testing.B) {
	c := Config{Vocab: 32768, Dim: 256, Hidden: 704, Layers: 1, Heads: 8, KVHeads: 2, Context: 512, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		b.Fatal(err)
	}
	m.DropOptimizer()
	tok := fixedStepGenerateTokenizer{}
	for _, shape := range [][2]int{{3, 1}, {33, 1}, {129, 1}, {257, 1}, {3, 32}} {
		prompt := strings.Repeat("a", shape[0]-1)
		s := defaultSampling()
		s.MaxTokens = shape[1]
		s.Temperature = 0
		s.RepeatPenalty = 1
		want, err := allHeadGenerateInput(context.Background(), m, tok, prompt, s, false)
		if err != nil {
			b.Fatal(err)
		}
		for _, reverse := range []bool{false, true} {
			order := []string{"before", "after"}
			if reverse {
				order[0], order[1] = order[1], order[0]
			}
			for _, variant := range order {
				b.Run(fmt.Sprintf("prompt%d/output%d/reverse%t/%s", shape[0], shape[1], reverse, variant), func(b *testing.B) {
					fn := generateInput
					if variant == "before" {
						fn = allHeadGenerateInput
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

// Frozen from 8dca84e: prompt chunks still compute every output head.
func allHeadGenerateInput(ctx context.Context, m *Model, tok inferenceTokenizer, prompt string, s Sampling, framed bool) (Completion, error) {
	ids, err := prepareInput(ctx, m, tok, prompt, s, framed)
	if err != nil {
		return Completion{}, err
	}
	plan, err := planInferenceKV(m.Config, len(ids)+s.MaxTokens, m.directKVBudget())
	if err != nil {
		return Completion{}, err
	}
	if !plan.FitsBudget {
		return Completion{}, plan.budgetError()
	}
	cache := m.NewCache()
	defer cache.Close()
	// One call owns its workers and scratch, including through sampling. Join
	// workers before releasing KV; no mutable execution state lives on Model.
	promptRows, workers := planGenerationPrompt(m, len(ids), s.MaxTokens)
	executor := newKernelExecutor(workers)
	defer executor.close()
	var workspace decodeWorkspace
	caches, chunks := []*KVCache{cache}, make([][]int, 1)
	var logits []float32
	for i := 0; i < len(ids); i += promptRows {
		if err = ctx.Err(); err != nil {
			return Completion{}, err
		}
		chunks[0] = ids[i:min(i+promptRows, len(ids))]
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
