// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestBoundedTopKCandidateSet(t *testing.T) {
	for _, size := range []int{1, 2, 7, 40, 129} {
		for _, order := range []string{"ascending", "descending", "ties"} {
			all := make([]candidate, size)
			for i := range all {
				v := float64(i)
				if order == "descending" {
					v = -v
				}
				if order == "ties" {
					v = float64(i%3 - 1)
				}
				all[i] = candidate{ID: i, Logit: v}
			}
			want := append([]candidate(nil), all...)
			sort.Slice(want, func(i, j int) bool {
				if want[i].Logit == want[j].Logit {
					return want[i].ID < want[j].ID
				}
				return want[i].Logit > want[j].Logit
			})
			for k := 1; k <= size; k++ {
				got := make([]candidate, 0, k)
				for _, c := range all {
					got = retainTopCandidate(got, c, k)
					if len(got) > k || cap(got) != k {
						t.Fatal("unbounded candidate storage")
					}
				}
				sort.Slice(got, func(i, j int) bool {
					if got[i].Logit == got[j].Logit {
						return got[i].ID < got[j].ID
					}
					return got[i].Logit > got[j].Logit
				})
				if !reflect.DeepEqual(got, want[:k]) {
					t.Fatal("candidate set differs", size, order, k)
				}
			}
		}
	}
}

func TestBoundedTopKSamplingExact(t *testing.T) {
	random := RNG{19}
	for _, size := range []int{8, 33, 258, 1024} {
		logits := make([]float32, size)
		for i := range logits {
			logits[i] = float32(random.Intn(65)-32) / 7
			if i%7 == 0 {
				logits[i] = float32(math.Copysign(0, -1))
			}
		}
		history := []int{0, 1, 1, size / 2, size - 1, -1, size + 1}
		original := append([]float32(nil), logits...)
		for _, k := range []int{0, 1, size / 4, size/4 + 1, size - 1, size, size + 1} {
			for _, blocked := range []int{-1, size / 2} {
				for _, temp := range []float64{.01, .8, 100} {
					for _, p := range []float64{.01, .95, 1} {
						for _, penalty := range []float64{1, 1.1} {
							s := defaultSampling()
							s.TopK = k
							s.Temperature = temp
							s.TopP = p
							s.RepeatPenalty = penalty
							for _, seed := range []uint64{1, 42, math.MaxUint64} {
								a, b := RNG{seed}, RNG{seed}
								got, err := sampleWithBlocked(logits, history, s, &a, blocked)
								want, werr := allCandidateSampleReference(logits, history, s, &b, blocked)
								if got != want || fmt.Sprint(err) != fmt.Sprint(werr) || a != b {
									t.Fatalf("size%d k%d temp%g p%g penalty%g seed%d: token/RNG differs", size, k, temp, p, penalty, seed)
								}
							}
						}
					}
				}
			}
		}
		exactWorkerFloats(t, logits, original)
	}
	for _, size := range []int{32768, 151936} {
		logits := make([]float32, size)
		for i := range logits {
			logits[i] = float32(random.Intn(10000)-5000) / 19
		}
		for _, seed := range []uint64{19, 42} {
			a, b := RNG{seed}, RNG{seed}
			s := defaultSampling()
			got, err := sampleWithBlocked(logits, []int{0, 40, 40}, s, &a, -1)
			want, werr := allCandidateSampleReference(logits, []int{0, 40, 40}, s, &b, -1)
			if err != nil || werr != nil || got != want || a != b {
				t.Fatal("large-vocabulary parity", size, err, werr)
			}
		}
	}
}

func TestBoundedTopKRejectsDiscardedNonfinite(t *testing.T) {
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		for _, blocked := range []int{-1, 127} {
			logits := make([]float32, 128)
			logits[0] = math.MaxFloat32
			logits[127] = bad
			s := defaultSampling()
			s.TopK = 1
			rng := RNG{19}
			if _, err := sampleWithBlocked(logits, nil, s, &rng, blocked); err == nil || rng != (RNG{19}) {
				t.Fatal("discarded/masked nonfinite value accepted")
			}
		}
	}
	for _, logits := range [][]float32{{0, 0, 0, 0, 0, 0, 0, 0}, {math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32, 0, 0, 0, 0, 0}} {
		for _, k := range []int{1, 2} {
			s := defaultSampling()
			s.TopK = k
			compareSampler(t, logits, []int{0, 0, 1}, s, -1)
		}
	}
}

func TestBoundedTopKEngineGenerationParity(t *testing.T) {
	for _, format := range []string{"f32", "q8", "q4"} {
		m := promptPrefillTestModel(t, format)
		e, err := NewEngine(m, fixedStepGenerateTokenizer{}, prefixEngineConfig(m))
		if err != nil {
			t.Fatal(err)
		}
		ctx := streamTestContext(t)
		s := defaultSampling()
		s.MaxTokens = 8
		s.Seed = 19
		for _, prompt := range []string{strings.Repeat("a", 65), strings.Repeat("a", 97), strings.Repeat("a", 129), strings.Repeat("b", 33)} {
			want, err := sortedTopKGenerateReference(ctx, m, fixedStepGenerateTokenizer{}, prompt, s, false)
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Generate(ctx, prompt, s)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatal("engine completion changed", format, err)
			}
		}
		e.Close()
	}
}

func BenchmarkBoundedTopKSampling(b *testing.B) {
	for _, shape := range []struct {
		vocab, k    int
		temperature float64
	}{{258, 40, .8}, {32768, 40, .8}, {50257, 40, .8}, {151936, 40, .8}, {32768, 0, .8}, {32768, 32768, .8}, {32768, 40, 0}} {
		logits := make([]float32, shape.vocab)
		r := RNG{19}
		for i := range logits {
			logits[i] = float32(r.Intn(10000)-5000) / 19
		}
		history := make([]int, 256)
		for i := range history {
			history[i] = r.Intn(shape.vocab)
		}
		s := defaultSampling()
		s.TopK = shape.k
		s.Temperature = shape.temperature
		for _, reverse := range []bool{false, true} {
			order := []string{"before", "after"}
			if reverse {
				order[0], order[1] = order[1], order[0]
			}
			for _, variant := range order {
				b.Run(fmt.Sprintf("vocab%d/k%d/temp%g/reverse%t/%s", shape.vocab, shape.k, shape.temperature, reverse, variant), func(b *testing.B) {
					fn := sampleWithBlocked
					if variant == "before" {
						fn = allCandidateSampleReference
					}
					a, z := RNG{19}, RNG{19}
					got, err := fn(logits, history, s, &a, -1)
					want, werr := allCandidateSampleReference(logits, history, s, &z, -1)
					if err != nil || werr != nil || got != want || a != z {
						b.Fatal("sample changed", err, werr)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err := fn(logits, history, s, &a, -1); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// Frozen from 8dca84e, including the already optimized greedy path.
func allCandidateSampleReference(logits []float32, history []int, s Sampling, rng *RNG, blocked int) (int, error) {
	if len(logits) == 0 {
		return 0, errors.New("empty sampling distribution")
	}
	var seen map[int]bool
	if s.Temperature != 0 || s.RepeatPenalty != 1 {
		seen = make(map[int]bool, len(history))
		for _, id := range history {
			seen[id] = true
		}
	}
	if s.Temperature == 0 {
		best, bestLogit := -1, float64(0)
		for id, v := range logits {
			// Validate even masked logits and logits after the eventual winner.
			if !finite(float64(v)) {
				return 0, errors.New("nonfinite sampling logit")
			}
			if id == blocked {
				continue
			}
			x := float64(v)
			if seen[id] {
				if x > 0 {
					x /= s.RepeatPenalty
				} else {
					x *= s.RepeatPenalty
				}
			}
			// Ascending IDs and strict comparison preserve the sorter's tie rule,
			// including signed zero. Greedy sampling never consumes RNG state.
			if best < 0 || x > bestLogit {
				best, bestLogit = id, x
			}
		}
		if best < 0 {
			return 0, errors.New("empty sampling distribution")
		}
		return best, nil
	}
	cs := make([]candidate, 0, len(logits)-1)
	for id, v := range logits {
		if !finite(float64(v)) {
			return 0, errors.New("nonfinite sampling logit")
		}
		if id == blocked {
			continue
		}
		x := float64(v)
		if seen[id] {
			if x > 0 {
				x /= s.RepeatPenalty
			} else {
				x *= s.RepeatPenalty
			}
		}
		cs = append(cs, candidate{ID: id, Logit: x})
	}
	if len(cs) == 0 {
		return 0, errors.New("empty sampling distribution")
	}
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].Logit == cs[j].Logit {
			return cs[i].ID < cs[j].ID
		}
		return cs[i].Logit > cs[j].Logit
	})
	if s.TopK > 0 && s.TopK < len(cs) {
		cs = cs[:s.TopK]
	}
	var total float64
	maxLogit := cs[0].Logit
	for i := range cs {
		cs[i].Prob = math.Exp((cs[i].Logit - maxLogit) / s.Temperature)
		total += cs[i].Prob
	}
	if s.TopP < 1 {
		var mass float64
		for i, c := range cs {
			mass += c.Prob
			if mass >= s.TopP*total {
				cs = cs[:i+1]
				total = mass
				break
			}
		}
	}
	draw := rng.Float64() * total
	for _, c := range cs {
		draw -= c.Prob
		if draw < 0 {
			return c.ID, nil
		}
	}
	return cs[len(cs)-1].ID, nil
}

// Frozen generator with the full-sort sampler above.
func sortedTopKGenerateReference(ctx context.Context, m *Model, tok inferenceTokenizer, prompt string, s Sampling, framed bool) (Completion, error) {
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
		id, e := allCandidateSampleReference(logits, history, s, rng, tok.blockedID())
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
