// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"runtime"
	"testing"
)

func assertWorkspaceReferencesCleared(t testing.TB, w *decodeWorkspace) {
	t.Helper()
	if len(w.pools) != 0 || len(w.seen) != 0 {
		t.Fatal("workspace retained KV cache/pool map keys")
	}
	for _, cache := range w.rowCaches {
		if cache != nil {
			t.Fatal("workspace retained a borrowed cache")
		}
	}
}

func TestInferenceArrayBudgetAcrossShapes(t *testing.T) {
	// A small budget exercises the same planner without a 512 MiB fixture.
	var a, b []float32
	arrays := []inferenceArray{{&a, 70}, {&b, 10}}
	if err := resizeInferenceArrays(arrays, 20, 100); err != nil {
		t.Fatal(err)
	}
	oldA := &a[0]
	arrays[0].n, arrays[1].n = 10, 70
	if err := resizeInferenceArrays(arrays, 20, 100); err != nil {
		t.Fatal(err)
	}
	if cap(a)+cap(b)+20 > 100 || &a[0] == oldA {
		t.Fatal("mixed shapes accumulated capacities beyond budget")
	}
	oldA, oldB := &a[0], &b[0]
	if err := resizeInferenceArrays(arrays, 20, 100); err != nil || &a[0] != oldA || &b[0] != oldB {
		t.Fatal("same shape did not reuse its arrays", err)
	}
	arrays[0].n = 11
	if err := resizeInferenceArrays(arrays, 20, 100); err == nil {
		t.Fatal("oversize shape accepted")
	}
	if len(a) != 10 || len(b) != 70 || &a[0] != oldA || &b[0] != oldB {
		t.Fatal("rejection resized or discarded existing buffers")
	}
}

func TestInferenceWorkspaceOwnershipAndGC(t *testing.T) {
	m := workerTestModel(t, 2)
	a, b := m.NewCache(), m.NewCache()
	defer a.Close()
	defer b.Close()
	var w decodeWorkspace
	chunks := [][]int{prefillTokens(35, 7)}
	logits, err := m.prefillBorrowed(context.Background(), []*KVCache{a}, chunks, nil, &w)
	if err != nil {
		t.Fatal(err)
	}
	if &logits[0] != &w.Logits[0] {
		t.Fatal("internal result was copied")
	}
	assertWorkspaceReferencesCleared(t, &w)
	x, out := &w.X[0], &w.Logits[0]
	retained := w.floatBytes()
	owned, err := m.PrefillBatch(context.Background(), []*KVCache{b}, chunks)
	if err != nil {
		t.Fatal(err)
	}
	saved := append([]float32(nil), owned...)
	exactWorkerFloats(t, owned, logits)
	// A GC cycle may evict a sync.Pool entry. It must not evict an actor-owned
	// workspace, nor may a different public caller alias its borrowed output.
	runtime.GC()
	runtime.GC()
	a.Pos = 0
	if _, err := m.prefillBorrowed(context.Background(), []*KVCache{a}, [][]int{prefillTokens(35, 19)}, nil, &w); err != nil {
		t.Fatal(err)
	}
	if &w.X[0] != x || &w.Logits[0] != out || w.floatBytes() != retained {
		t.Fatal("GC evicted the owned workspace")
	}
	exactWorkerFloats(t, owned, saved)
	assertWorkspaceReferencesCleared(t, &w)
}

func TestInferenceWorkspacePoolReferencesAndReuse(t *testing.T) {
	m := workerTestModel(t, 2)
	var w decodeWorkspace
	for trial := 0; trial < 12; trial++ {
		pool := &pagePool{PageElements: 2 * pageTokens * m.Config.KVDim() * m.Config.Layers, Limit: 8, Reserved: 8}
		caches := []*KVCache{m.NewCache(), m.NewCache()}
		for _, cache := range caches {
			cache.pool, cache.reserved = pool, 4
		}
		// Populate the first cache's validation maps before rejecting a second
		// invalid chunk. No cache may retain or borrow a page after this error.
		if _, err := m.prefillBorrowed(context.Background(), caches, [][]int{{1}, {-1}}, nil, &w); err == nil {
			t.Fatal("invalid second chunk accepted")
		}
		assertWorkspaceReferencesCleared(t, &w)
		if pool.Allocated != 0 {
			t.Fatal("validation allocated KV pages")
		}
		for _, chunks := range [][][]int{
			{prefillTokens(33, trial), {7}}, {{8}, prefillTokens(5, trial+1)},
		} {
			if _, err := m.prefillBorrowed(context.Background(), caches, chunks, nil, &w); err != nil {
				t.Fatal(err)
			}
			assertWorkspaceReferencesCleared(t, &w)
		}
		for _, cache := range caches {
			cache.Close()
		}
		if pool.Reserved != 0 {
			t.Fatal("workspace kept a reservation")
		}
	}
}

func BenchmarkInferenceWorkspace(b *testing.B) {
	for _, vocab := range []int{258, 50257} {
		for _, rows := range []int{1, 16} {
			for _, mode := range []string{"pooled-copy", "owned-borrow"} {
				b.Run(fmt.Sprintf("vocab%d/rows%d/%s", vocab, rows, mode), func(b *testing.B) {
					c := Config{Vocab: vocab, Dim: 64, Hidden: 100, Layers: 2, Heads: 4, KVHeads: 2, Context: 128, RopeTheta: 10000, NormEps: 1e-5}
					m, err := NewModel(c, 19)
					if err != nil {
						b.Fatal(err)
					}
					m.DropOptimizer()
					exec := newKernelExecutor(4)
					defer exec.close()
					var w decodeWorkspace
					pool := &pagePool{PageElements: 2 * pageTokens * c.KVDim() * c.Layers, Limit: 16, Reserved: 16}
					caches, chunks := make([]*KVCache, 4), make([][]int, 4)
					for i := range caches {
						caches[i] = m.NewCache()
						caches[i].pool, caches[i].reserved = pool, 4
						defer caches[i].Close()
						if _, err := m.PrefillBatch(context.Background(), caches[i:i+1], [][]int{prefillTokens(96, i)}); err != nil {
							b.Fatal(err)
						}
						chunks[i] = prefillTokens(rows, i+3)
					}
					run := func() {
						for _, cache := range caches {
							cache.Pos = 96
						}
						if mode == "owned-borrow" {
							_, err = m.prefillBorrowed(context.Background(), caches, chunks, exec, &w)
						} else {
							_, err = m.prefillBatch(context.Background(), caches, chunks, exec)
						}
						if err != nil {
							b.Fatal(err)
						}
					}
					run()
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						run()
					}
					b.StopTimer()
				})
			}
		}
	}
}
