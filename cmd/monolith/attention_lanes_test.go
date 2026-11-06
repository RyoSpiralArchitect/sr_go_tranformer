// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"testing"
)

// Frozen attention dispatch from 7167259; each head's attend arithmetic is shared unchanged.
func baselineInferenceAttention(executor *kernelExecutor, w *decodeWorkspace, c Config, l, n, maxPos int) {
	d, hd, group := c.Dim, c.Dim/c.Heads, c.Heads/c.KVHeads
	rowCaches, positions := w.rowCaches[:n], w.positions[:n]
	executor.parallel(n, n*c.Heads*maxPos*hd, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			cache := rowCaches[i]
			scores := w.Scores[i*maxPos : (i+1)*maxPos]
			for h := 0; h < c.Heads; h++ {
				kh := h / group
				qi := i*d + h*hd
				kv := func(pos int) ([]float32, []float32) {
					k, v := cache.row(l, pos)
					return k[kh*hd : (kh+1)*hd], v[kh*hd : (kh+1)*hd]
				}
				attend(w.Q[qi:qi+hd], w.A[qi:qi+hd], scores, positions[i]+1, kv)
			}
		}
	})
}

func attentionLaneFixture(c Config, rows, length int) (*decodeWorkspace, *KVCache) {
	m := &Model{Config: c}
	cache := m.NewCache()
	cache.ensurePages(length)
	for _, page := range cache.Pages {
		for i := range page.Data {
			page.Data[i] = float32((i*29+17)%521-260) / 257
		}
	}
	w := &decodeWorkspace{Q: make([]float32, rows*c.Dim), A: make([]float32, rows*c.Dim)}
	for i := range w.Q {
		w.Q[i] = float32((i*13+3)%127-63) / 67
	}
	for row := 0; row < rows; row++ {
		w.rowCaches[row], w.positions[row] = cache, length-1-row%min(length, 3)
	}
	return w, cache
}

func TestAttentionLanesExactAcrossHeadsPagesAndWorkers(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	for _, kvHeads := range []int{1, 2, 6} {
		c := Config{Dim: 60, Heads: 6, KVHeads: kvHeads, Layers: 2, Context: 301}
		for _, length := range []int{1, 31, 32, 33, 128, 257, 301} {
			for _, rows := range []int{1, 2, 3, 7, 31} {
				w, cache := attentionLaneFixture(c, rows, length)
				for _, layer := range []int{0, 1} {
					w.Scores = make([]float32, rows*length)
					baselineInferenceAttention(nil, w, c, layer, rows, length)
					want := append([]float32(nil), w.A...)
					for _, workers := range []int{0, 1, 2, 4, 9} {
						var e *kernelExecutor
						if workers > 0 {
							e = newKernelExecutor(workers)
						}
						plan := planInferenceAttention(e, c, rows, length)
						sentinel := make([]float32, plan.lanes*plan.stride+2)
						sentinel[0], sentinel[len(sentinel)-1] = -123, 456
						w.Scores = sentinel[1 : len(sentinel)-1]
						for i := range w.Scores {
							w.Scores[i] = float32(math.NaN())
						}
						w.attend(e, c, layer, rows, plan)
						if e != nil {
							e.close()
						}
						if sentinel[0] != -123 || sentinel[len(sentinel)-1] != 456 {
							t.Fatal("score overrun")
						}
						exactWorkerFloats(t, w.A, want)
					}
				}
				cache.Close()
			}
		}
	}
}

func TestAttentionLaneScratchGrowthAndBounds(t *testing.T) {
	c := Config{Dim: 4096, Heads: 32, KVHeads: 8, Context: 197}
	e := newKernelExecutor(4)
	defer e.close()
	var scores []float32
	allocations := 0
	for length := 65; length <= 197; length++ {
		plan := planInferenceAttention(e, c, 1, length)
		if plan.lanes != 4 || plan.stride < length || plan.stride > c.Context || plan.stride > length+pageTokens-1 {
			t.Fatal("bad scratch plan", length, plan)
		}
		var before *float32
		if len(scores) > 0 {
			before = &scores[0]
		}
		arrays := []inferenceArray{{&scores, plan.lanes * plan.stride}}
		if err := resizeInferenceArrays(arrays, 0, int64(4*c.Context)); err != nil {
			t.Fatal(err)
		}
		if before != &scores[0] {
			allocations++
		}
	}
	if allocations != 5 {
		t.Fatalf("got %d score allocations for 133 positions; want 5 page buckets", allocations)
	}
	for _, workers := range []int{1, 4, 256} {
		executor := &kernelExecutor{workers: workers}
		for _, rows := range []int{1, 2, 1024} {
			plan := planInferenceAttention(executor, c, rows, c.Context)
			if plan.lanes > min(workers, rows*c.Heads, 256) || plan.stride != 197 {
				t.Fatal("unbounded scratch", plan)
			}
		}
	}
}

func TestLongDecodeLaneParityAndReuse(t *testing.T) {
	c := Config{Vocab: 258, Dim: 64, Hidden: 100, Heads: 4, KVHeads: 2, Layers: 2, Context: 320, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	a, b := m.NewCache(), m.NewCache()
	defer a.Close()
	defer b.Close()
	serial, parallel := newKernelExecutor(1), newKernelExecutor(4)
	defer serial.close()
	defer parallel.close()
	var x, y decodeWorkspace
	for step := 0; step < 272; step++ {
		chunks := [][]int{{(step*17 + 3) % 256}}
		got, err := m.prefillBorrowed(context.Background(), []*KVCache{a}, chunks, parallel, &x)
		if err != nil {
			t.Fatal(err)
		}
		want, err := m.prefillBorrowed(context.Background(), []*KVCache{b}, chunks, serial, &y)
		if err != nil {
			t.Fatal(err)
		}
		exactWorkerFloats(t, got, want)
		assertWorkspaceReferencesCleared(t, &x)
	}
	for i, page := range a.Pages {
		exactWorkerFloats(t, page.Data, b.Pages[i].Data)
	}
}

func BenchmarkAttentionLanes(b *testing.B) {
	c := Config{Dim: 4096, Heads: 32, KVHeads: 8, Layers: 1, Context: 8192}
	for _, length := range []int{128, 2048, 8192} {
		for _, rows := range []int{1, 2, 16} {
			w, cache := attentionLaneFixture(c, rows, length)
			e := newKernelExecutor(4)
			plan := planInferenceAttention(e, c, rows, length)
			beforeScores, afterScores := make([]float32, rows*length), make([]float32, plan.lanes*plan.stride)
			w.Scores = beforeScores
			baselineInferenceAttention(e, w, c, 0, rows, length)
			want := append([]float32(nil), w.A...)
			w.Scores = afterScores
			w.attend(e, c, 0, rows, plan)
			exactWorkerFloats(b, w.A, want)
			for _, reverse := range []bool{false, true} {
				order := []string{"before", "after"}
				if reverse {
					order[0], order[1] = order[1], order[0]
				}
				for _, variant := range order {
					b.Run(fmt.Sprintf("prefix%d/rows%d/reverse%t/%s", length, rows, reverse, variant), func(b *testing.B) {
						w.Scores = afterScores
						if variant == "before" {
							w.Scores = beforeScores
						}
						b.ReportAllocs()
						b.SetBytes(int64(rows * length * c.Dim * 8))
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							if variant == "before" {
								baselineInferenceAttention(e, w, c, 0, rows, length)
							} else {
								w.attend(e, c, 0, rows, plan)
							}
						}
					})
				}
			}
			e.close()
			cache.Close()
		}
	}
}
