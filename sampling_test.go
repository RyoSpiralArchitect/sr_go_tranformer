// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"testing"
)

// Keep the pre-optimization sorter as an independent compatibility oracle.
// Production does not call this function. Its original contract requires at
// least one logit; empty distributions are tested separately below.
func sortedSampleReference(logits []float32, history []int, s Sampling, rng *RNG, blocked int) (int, error) {
	seen := make(map[int]bool, len(history))
	for _, id := range history {
		seen[id] = true
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
	if s.Temperature == 0 {
		return cs[0].ID, nil
	}
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

func compareSampler(t *testing.T, logits []float32, history []int, s Sampling, blocked int) {
	t.Helper()
	a, b := RNG{42}, RNG{42}
	got, err := sampleWithBlocked(logits, history, s, &a, blocked)
	want, refErr := sortedSampleReference(logits, history, s, &b, blocked)
	if got != want || fmt.Sprint(err) != fmt.Sprint(refErr) || a != b {
		t.Fatalf("sampling=%+v blocked=%d: got (%d,%v,%v), reference (%d,%v,%v)", s, blocked, got, err, a, want, refErr, b)
	}
	if s.Temperature == 0 && a != (RNG{42}) {
		t.Fatal("greedy sampling consumed RNG state")
	}
}

func TestGreedyMatchesSortedReference(t *testing.T) {
	fixtures := [][]float32{
		{0}, {3, 3, 3}, {-3, -3, -3}, {-10, -2, -4},
		{float32(math.Copysign(0, -1)), 0, -1},
		{math.SmallestNonzeroFloat32, 0, -math.SmallestNonzeroFloat32},
		{math.MaxFloat32, -math.MaxFloat32, math.MaxFloat32},
		{20, 10, 5}, // Penalty can create a tie or change the winner.
	}
	rng := RNG{17}
	for trial := 0; trial < 128; trial++ {
		x := make([]float32, 1+rng.Intn(1024))
		for i := range x {
			x[i] = float32(rng.Intn(101)-50) / 7
		}
		fixtures = append(fixtures, x)
	}
	for _, logits := range fixtures {
		for _, penalty := range []float64{1, 1.1, 2, 100} {
			for _, blocked := range []int{-1, 0, len(logits) / 2, len(logits)} {
				s := defaultSampling()
				s.Temperature, s.RepeatPenalty = 0, penalty
				history := []int{0, 0, 1, 3, -1, len(logits) + 1}
				copyLogits := append([]float32(nil), logits...)
				copyHistory := append([]int(nil), history...)
				compareSampler(t, logits, history, s, blocked)
				if !reflect.DeepEqual(logits, copyLogits) || !reflect.DeepEqual(history, copyHistory) {
					t.Fatal("sampling mutated its inputs")
				}
			}
		}
	}
}

func TestSamplingInvalidDistributions(t *testing.T) {
	for _, temperature := range []float64{0, .8} {
		s := defaultSampling()
		s.Temperature = temperature
		for _, logits := range [][]float32{nil, {}, {1}} {
			if _, err := sampleWithBlocked(logits, nil, s, &RNG{1}, 0); err == nil {
				t.Fatal("empty distribution accepted")
			}
		}
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			for _, logits := range [][]float32{{bad, 1}, {1, bad}, {100, 0, bad}} {
				for _, blocked := range []int{-1, 0, len(logits) - 1} {
					compareSampler(t, logits, nil, s, blocked)
					if _, err := sampleWithBlocked(logits, nil, s, &RNG{1}, blocked); err == nil {
						t.Fatal("nonfinite logit accepted, including masked entries")
					}
				}
			}
		}
	}
}

func TestSampledPathMatchesSortedReference(t *testing.T) {
	logits := make([]float32, 259)
	rng := RNG{37}
	for i := range logits {
		logits[i] = float32(rng.Intn(1000)-500) / 13
	}
	for _, temperature := range []float64{.01, .8, 2, 100} {
		for _, topK := range []int{0, 1, 40, 1000} {
			for _, topP := range []float64{.01, .7, 1} {
				for _, blocked := range []int{-1, 0, 258} {
					s := defaultSampling()
					s.Temperature, s.TopK, s.TopP = temperature, topK, topP
					compareSampler(t, logits, []int{0, 7, 128, 128}, s, blocked)
				}
			}
		}
	}
}

func TestGreedyWithoutPenaltyAllocatesNothing(t *testing.T) {
	logits := make([]float32, 50257)
	history := make([]int, 512)
	s := defaultSampling()
	s.Temperature, s.RepeatPenalty = 0, 1
	var err error
	if n := testing.AllocsPerRun(20, func() {
		_, err = sampleWithBlocked(logits, history, s, nil, -1)
	}); n != 0 || err != nil {
		t.Fatalf("allocs=%v err=%v", n, err)
	}
}

func BenchmarkGreedySampling(b *testing.B) {
	for _, vocab := range []int{258, 49152, 50257, 151936} {
		logits := make([]float32, vocab)
		history := make([]int, 256)
		rng := RNG{19}
		for i := range logits {
			logits[i] = float32(rng.Float64()*20 - 10)
		}
		for i := range history {
			history[i] = rng.Intn(vocab)
		}
		for _, penalty := range []float64{1, 1.1} {
			s := defaultSampling()
			s.Temperature, s.RepeatPenalty = 0, penalty
			for _, mode := range []string{"sorted", "scan"} {
				b.Run(fmt.Sprintf("vocab%d/penalty%.1f/%s", vocab, penalty, mode), func(b *testing.B) {
					fn := sortedSampleReference
					if mode == "scan" {
						fn = sampleWithBlocked
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err := fn(logits, history, s, &rng, -1); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
