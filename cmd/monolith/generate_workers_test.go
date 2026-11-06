// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// Frozen direct-generation path from a459cb4, before request-owned execution.
func baselineGenerateInput(ctx context.Context, m *Model, tok inferenceTokenizer, prompt string, s Sampling, framed bool) (Completion, error) {
	ids, err := prepareInput(ctx, m, tok, prompt, s, framed)
	if err != nil {
		return Completion{}, err
	}
	cache := m.NewCache()
	defer cache.Close()
	var logits []float32
	for _, id := range ids {
		if err = ctx.Err(); err != nil {
			return Completion{}, err
		}
		logits, err = m.DecodeBatch(ctx, []*KVCache{cache}, []int{id})
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
			logits, err = m.DecodeBatch(ctx, []*KVCache{cache}, []int{id})
			if err != nil {
				return Completion{}, err
			}
		}
	}
	return completion(tok, out, len(ids), reason), nil
}

// The MLP projections exceed the inline threshold even for a single token.
func ownedGenerateTestModel(t testing.TB) *Model {
	t.Helper()
	c := Config{Vocab: 258, Dim: 256, Hidden: 704, Layers: 1, Heads: 4, KVHeads: 2, Context: 512, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	m.DropOptimizer()
	return m
}

func TestOwnedGenerateWorkerBoundary(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	c := Config{Vocab: 258, Dim: 128, Hidden: 256, Layers: 1, Heads: 4, KVHeads: 2, Context: 1056, RopeTheta: 10000, NormEps: 1e-5}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	// Last positions 1023 and 1024 straddle the actual attention threshold.
	if got := generationWorkerCount(c, 8, 1016); got != 1 {
		t.Fatalf("idle pool: %d", got)
	}
	if got := generationWorkerCount(c, 8, 1017); got != 4 {
		t.Fatalf("missing pool: %d", got)
	}
	if got := generationWorkerCount(ownedGenerateTestModel(t).Config, 8, 16); got != 4 {
		t.Fatalf("fixture does not use workers: %d", got)
	}
}

func TestOwnedGenerateMatchesReference(t *testing.T) {
	for _, threads := range []int{1, 4} {
		t.Run(fmt.Sprintf("threads%d", threads), func(t *testing.T) {
			old := runtime.GOMAXPROCS(threads)
			defer runtime.GOMAXPROCS(old)
			models := []struct {
				m      *Model
				tok    inferenceTokenizer
				prompt string
				framed bool
			}{
				{workerTestModel(t, 2), Tokenizer{}, "the cat", false},
				{ownedGenerateTestModel(t), Tokenizer{}, "the cat", false},
			}
			for _, dir := range []string{"testdata/hf-tokenizer", "testdata/hf-mistral-tokenizer"} {
				tok := testHFTokenizer(t, dir)
				m := hfServingModel(t, tok, -1)
				prompt, err := tok.formatChat([]ChatMessage{{Role: "user", Content: "Hello, 猫!"}})
				if err != nil {
					t.Fatal(err)
				}
				models = append(models, struct {
					m      *Model
					tok    inferenceTokenizer
					prompt string
					framed bool
				}{m, tok, prompt, true})
			}
			for _, model := range models {
				for _, temperature := range []float64{0, .7} {
					s := defaultSampling()
					s.MaxTokens = 16
					s.Temperature = temperature
					s.Seed = 19
					before, err := baselineGenerateInput(context.Background(), model.m, model.tok, model.prompt, s, model.framed)
					if err != nil {
						t.Fatal(err)
					}
					after, err := generateInput(context.Background(), model.m, model.tok, model.prompt, s, model.framed)
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(before, after) {
						t.Fatal("completion changed", before, after)
					}
				}
			}
		})
	}
}

func TestOwnedGenerateCancellationRetryAndSharedModel(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	m := ownedGenerateTestModel(t)
	tok := fixedStepGenerateTokenizer{}
	s := defaultSampling()
	s.MaxTokens = 24
	s.Temperature = 0
	s.RepeatPenalty = 1
	want, err := baselineGenerateInput(context.Background(), m, tok, "the cat", s, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, at := range []int{1, 4, 12, 24} {
		ctx := &prefillCancelContext{Context: context.Background(), cancelAt: at}
		_, err := generateInput(ctx, m, tok, "the cat", s, false)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel checkpoint %d: %v", at, err)
		}
		got, err := generateInput(context.Background(), m, tok, "the cat", s, false)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatal("retry changed", got, err)
		}
	}
	// Arm cancellation only after output sampling, so these checkpoints cover
	// generation kernels and joins rather than merely prompt processing.
	for _, at := range []int{1, 2, 3, 4, 6, 12} {
		sampled := 0
		tracking := sampledGenerateTokenizer{sampled: &sampled}
		ctx := &generatedCancelContext{Context: context.Background(), sampled: &sampled, cancelAt: at}
		_, err := generateInput(ctx, m, tracking, "the cat", s, false)
		if !errors.Is(err, context.Canceled) || sampled == 0 || ctx.calls != at {
			t.Fatalf("generation cancellation %d: sampled=%d calls=%d err=%v", at, sampled, ctx.calls, err)
		}
		got, err := generateInput(context.Background(), m, tok, "the cat", s, false)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatal("generation retry changed", got, err)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := Generate(context.Background(), m, tok, "the cat", s)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Errorf("shared generation: %v", err)
			}
		}()
	}
	wg.Wait()
}

// Keep benchmark and cancellation workloads fixed; EOS handling is covered by real tokenizers
// in TestHFGenerationSpecialIDsAndStreaming and the parity tests above.
type fixedStepGenerateTokenizer struct{ Tokenizer }

func (fixedStepGenerateTokenizer) isEOS(int) bool { return false }

type sampledGenerateTokenizer struct {
	fixedStepGenerateTokenizer
	sampled *int
}

func (t sampledGenerateTokenizer) isEOS(int) bool { *t.sampled++; return false }

type generatedCancelContext struct {
	context.Context
	sampled         *int
	calls, cancelAt int
}

func (c *generatedCancelContext) Err() error {
	if *c.sampled > 0 {
		c.calls++
		if c.calls >= c.cancelAt {
			return context.Canceled
		}
	}
	return nil
}

type decodePathGenerateTokenizer struct {
	fixedStepGenerateTokenizer
	packedBefore bool
}

func (t decodePathGenerateTokenizer) Decode(ids []int) string {
	if t.packedBefore {
		return packedTokenizerDecode(t.Tokenizer, ids)
	}
	return t.Tokenizer.Decode(ids)
}

func decodeComparisonModel(t testing.TB, tok Tokenizer) *Model {
	t.Helper()
	c := Config{Vocab: tok.Vocab(), Dim: 256, Hidden: 704, Layers: 1, Heads: 4, KVHeads: 2, Context: 512, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	m.DropOptimizer()
	return m
}

func decodeComparisonRun(t testing.TB, m *Model, tok Tokenizer, before bool, prefix, outputs int) Completion {
	t.Helper()
	s := defaultSampling()
	s.MaxTokens, s.Temperature, s.RepeatPenalty, s.Seed = outputs, 0, 1, 19
	result, err := generateInput(context.Background(), m, decodePathGenerateTokenizer{fixedStepGenerateTokenizer{tok}, before}, strings.Repeat("a", prefix-1), s, false)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestDecodeOnlyGenerationParity(t *testing.T) {
	for _, tok := range tokenizerPiecesFixtures()[:2] {
		m := decodeComparisonModel(t, tok)
		for _, shape := range [][2]int{{3, 1}, {129, 1}, {257, 1}, {33, 8}, {33, 128}} {
			before := decodeComparisonRun(t, m, tok, true, shape[0], shape[1])
			after := decodeComparisonRun(t, m, tok, false, shape[0], shape[1])
			if !reflect.DeepEqual(before, after) || after.CompletionTokens != shape[1] {
				t.Fatal("decode-only generation changed completion")
			}
		}
	}
}

func BenchmarkDecodeOnlyGenerate(b *testing.B) {
	for fixture, tok := range tokenizerPiecesFixtures()[:2] {
		m := decodeComparisonModel(b, tok)
		for _, outputs := range []int{8, 128} {
			want := decodeComparisonRun(b, m, tok, true, 33, outputs)
			for _, before := range []bool{true, false} {
				b.Run(fmt.Sprintf("fixture%d/tokens%d/before%t", fixture, outputs, before), func(b *testing.B) {
					if !reflect.DeepEqual(decodeComparisonRun(b, m, tok, before, 33, outputs), want) {
						b.Fatal("completion changed")
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						decodeComparisonRun(b, m, tok, before, 33, outputs)
					}
				})
			}
		}
	}
}

type decodeGenerationTimingResult struct {
	Name                      string
	BeforeNS, AfterNS, Ratios []float64
	MedianRatio               float64
}

func TestDecodeOnlyGenerationTiming(t *testing.T) {
	output := os.Getenv("MONOLITH_DECODE_ONLY_OUTPUT")
	if output == "" {
		t.Skip("opt-in decode-only generation comparison")
	}
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	artifact := struct {
		Protocol, GoVersion, GOOS, GOARCH, StartedUTC, FinishedUTC string
		Trials, Procs                                              int
		Hashes                                                     map[string]string
		Results                                                    []decodeGenerationTimingResult
	}{Protocol: "decode-only-generation-v1", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, Trials: 24, Procs: 4, Hashes: make(map[string]string)}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{executable, "cmd/monolith/monolith.go", "cmd/monolith/monolith_test.go", "cmd/monolith/generate_workers_test.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		artifact.Hashes[filepath.Base(path)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	artifact.StartedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	for fixture, tok := range tokenizerPiecesFixtures()[:2] {
		m := decodeComparisonModel(t, tok)
		for _, shape := range [][2]int{{3, 1}, {129, 1}, {257, 1}, {33, 8}, {33, 128}} {
			for warm := 0; warm < 3; warm++ {
				before := decodeComparisonRun(t, m, tok, true, shape[0], shape[1])
				after := decodeComparisonRun(t, m, tok, false, shape[0], shape[1])
				if !reflect.DeepEqual(before, after) {
					t.Fatal("warm completion changed")
				}
			}
			result := decodeGenerationTimingResult{Name: fmt.Sprintf("fixture%d/prefix%d/outputs%d", fixture, shape[0], shape[1])}
			for trial := 0; trial < artifact.Trials; trial++ {
				var ns [2]float64
				var completions [2]Completion
				for leg := 0; leg < 2; leg++ {
					variant := (trial + leg) % 2
					start := time.Now()
					completions[variant] = decodeComparisonRun(t, m, tok, variant == 0, shape[0], shape[1])
					ns[variant] = float64(time.Since(start).Nanoseconds())
				}
				if !reflect.DeepEqual(completions[0], completions[1]) {
					t.Fatal("timed completion changed")
				}
				result.BeforeNS = append(result.BeforeNS, ns[0])
				result.AfterNS = append(result.AfterNS, ns[1])
				result.Ratios = append(result.Ratios, ns[1]/ns[0])
			}
			ratios := append([]float64(nil), result.Ratios...)
			sort.Float64s(ratios)
			result.MedianRatio = (ratios[11] + ratios[12]) / 2
			artifact.Results = append(artifact.Results, result)
			t.Logf("%s: median %.4f", result.Name, result.MedianRatio)
		}
	}
	artifact.FinishedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	encoded, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := file.Write(append(encoded, '\n'))
	closeErr := file.Close()
	if writeErr != nil {
		t.Fatal(writeErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
}

func BenchmarkOwnedGenerate(b *testing.B) {
	c := Config{Vocab: 258, Dim: 256, Hidden: 704, Layers: 1, Heads: 4, KVHeads: 2, Context: 512, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		b.Fatal(err)
	}
	m.DropOptimizer()
	prompt := strings.Repeat("the ", 8)
	tok := fixedStepGenerateTokenizer{}
	for _, outputs := range []int{8, 128} {
		s := defaultSampling()
		s.MaxTokens = outputs
		s.Temperature = 0
		s.RepeatPenalty = 1
		want, err := baselineGenerateInput(context.Background(), m, tok, prompt, s, false)
		if err != nil {
			b.Fatal(err)
		}
		for _, reverse := range []bool{false, true} {
			order := []string{"before", "after"}
			if reverse {
				order[0], order[1] = order[1], order[0]
			}
			for _, variant := range order {
				b.Run(fmt.Sprintf("tokens%d/reverse%t/%s", outputs, reverse, variant), func(b *testing.B) {
					fn := generateInput
					if variant == "before" {
						fn = baselineGenerateInput
					}
					got, err := fn(context.Background(), m, tok, prompt, s, false)
					if err != nil || got.CompletionTokens != outputs || !reflect.DeepEqual(got, want) {
						b.Fatal("unexpected completion", err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err = fn(context.Background(), m, tok, prompt, s, false); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
