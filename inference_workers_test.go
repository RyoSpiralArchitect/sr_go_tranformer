// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func workerTestModel(t testing.TB, kvHeads int) *Model {
	t.Helper()
	c := Config{Vocab: baseVocab, Dim: 64, Hidden: 100, Layers: 2, Heads: 4, KVHeads: kvHeads, Context: 128, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func exactWorkerFloats(t testing.TB, a, b []float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length %d != %d", len(a), len(b))
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) || !finite(float64(a[i])) {
			t.Fatalf("float bits differ at %d: %08x != %08x", i, math.Float32bits(a[i]), math.Float32bits(b[i]))
		}
	}
}

func TestKernelExecutorBoundsAndJoins(t *testing.T) {
	e := newKernelExecutor(4)
	defer e.close()
	entered := make(chan int, 4)
	resume := make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(resume) }) }
	defer open() // Release blocked workers even on a failed assertion.
	finished := make(chan struct{})
	values := make([]int, 19)
	go func() {
		e.parallel(len(values), 131072, func(lo, hi int) {
			entered <- lo
			<-resume
			for i := lo; i < hi; i++ {
				values[i]++
			}
		})
		close(finished)
	}()
	seen := map[int]bool{}
	for i := 0; i < 4; i++ {
		select {
		case lo := <-entered:
			if seen[lo] {
				t.Fatal("overlapping partition")
			}
			seen[lo] = true
		case <-streamTestContext(t).Done():
			t.Fatal("workers failed to reach the gate")
		}
	}
	select {
	case <-finished:
		t.Fatal("dispatcher returned with live borrowers")
	case <-entered:
		t.Fatal("worker bound exceeded")
	default:
	}
	open()
	waitLifecycle(t, finished)
	for _, v := range values {
		if v != 1 {
			t.Fatal("output ranges did not cover each element exactly once")
		}
	}
	// Repeated dispatches join all writes and do not allocate job mailboxes or
	// goroutines. Capture the callback once to isolate executor overhead.
	fn := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			values[i]++
		}
	}
	if allocs := testing.AllocsPerRun(100, func() { e.parallel(len(values), 131072, fn) }); allocs != 0 {
		t.Fatalf("steady dispatch allocated %.1f objects", allocs)
	}
	for _, v := range values {
		if v != 102 { // Initial batch, AllocsPerRun warmup, then 100 dispatches.
			t.Fatalf("work survived a join or was lost: %d", v)
		}
	}
}

func TestKernelExecutorInlineThreshold(t *testing.T) {
	for _, workers := range []int{1, 4} {
		e := newKernelExecutor(workers)
		for _, tc := range []struct{ n, work int }{{0, 0}, {1, 131072}, {19, 131071}} {
			calls := 0
			e.parallel(tc.n, tc.work, func(lo, hi int) {
				calls++
				if lo != 0 || hi != tc.n {
					t.Fatalf("small kernel partitioned: %d..%d", lo, hi)
				}
			})
			if calls != 1 {
				t.Fatal("small kernel did not run inline once")
			}
		}
		e.close()
	}
}

func TestWorkerLinearScalarParity(t *testing.T) {
	for _, workers := range []int{1, 2, 4} {
		e := newKernelExecutor(workers)
		for _, n := range []int{1, 3, 4, 7} {
			// Even a single row exceeds the parallel threshold. Neither the
			// input dimension nor output dimension is a tile multiple.
			x, w, _ := linearFixture(n, 513, 259)
			got, want := make([]float32, n*w.Rows), make([]float32, n*w.Rows)
			for row := 0; row < n; row++ {
				for o := 0; o < w.Rows; o++ {
					var sum float32
					for j := 0; j < w.Cols; j++ {
						sum += x[row*w.Cols+j] * w.Data[o*w.Cols+j]
					}
					want[row*w.Rows+o] = sum
				}
			}
			linearIntoWithExecutor(e, got, x, w, n)
			exactWorkerFloats(t, got, want)
		}
		e.close()
	}
}

func TestWorkerPrefillBitwiseParity(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	for _, kvHeads := range []int{1, 2, 4} {
		for _, workers := range []int{1, 2, 4} {
			t.Run(fmt.Sprintf("kv%d/workers%d", kvHeads, workers), func(t *testing.T) {
				m := workerTestModel(t, kvHeads)
				e := newKernelExecutor(workers)
				defer e.close()
				caches, refs := make([]*KVCache, 4), make([]*KVCache, 4)
				for i, prefix := range []int{0, 31, 63, 5} {
					caches[i], refs[i] = m.NewCache(), m.NewCache()
					defer caches[i].Close()
					defer refs[i].Close()
					if prefix > 0 {
						ids := prefillTokens(prefix, i)
						for _, cache := range []*KVCache{caches[i], refs[i]} {
							if _, err := m.PrefillBatch(context.Background(), []*KVCache{cache}, [][]int{ids}); err != nil {
								t.Fatal(err)
							}
						}
					}
				}
				// Mixed prefill/decode, page crossings, attention large enough for
				// worker dispatch, row/output tails, then decode on the same caches.
				for _, lengths := range [][]int{{1, 7, 33, 2}, {1, 1, 1, 1}, {5, 9, 3, 6}} {
					chunks := make([][]int, 4)
					for i, n := range lengths {
						chunks[i] = prefillTokens(n, i+7)
					}
					got, err := m.prefillBatch(context.Background(), caches, chunks, e)
					if err != nil {
						t.Fatal(err)
					}
					want, err := m.PrefillBatch(context.Background(), refs, chunks)
					if err != nil {
						t.Fatal(err)
					}
					exactWorkerFloats(t, got, want)
					for i := range caches {
						if caches[i].Pos != refs[i].Pos {
							t.Fatal("committed position changed")
						}
						for j, page := range caches[i].Pages {
							exactWorkerFloats(t, page.Data, refs[i].Pages[j].Data)
						}
					}
				}
			})
		}
	}
}

func TestWorkerCancellationAndRetry(t *testing.T) {
	m := workerTestModel(t, 2)
	e := newKernelExecutor(4)
	defer e.close()
	for _, cancelAt := range []int{1, 2, 3, 4, 5} {
		cache, ref := m.NewCache(), m.NewCache()
		for _, c := range []*KVCache{cache, ref} {
			if _, err := m.PrefillBatch(context.Background(), []*KVCache{c}, [][]int{prefillTokens(31, 3)}); err != nil {
				t.Fatal(err)
			}
		}
		ctx := &prefillCancelContext{Context: context.Background(), cancelAt: cancelAt}
		if _, err := m.prefillBatch(ctx, []*KVCache{cache}, [][]int{prefillTokens(35, 4)}, e); !errors.Is(err, context.Canceled) {
			t.Fatalf("checkpoint %d: %v", cancelAt, err)
		}
		if cache.Pos != ref.Pos {
			t.Fatal("cancellation committed future rows")
		}
		for l := range m.Blocks {
			for p := 0; p < cache.Pos; p++ {
				k, v := cache.row(l, p)
				rk, rv := ref.row(l, p)
				exactWorkerFloats(t, k, rk)
				exactWorkerFloats(t, v, rv)
			}
		}
		chunks := [][]int{prefillTokens(33, 9)}
		got, err := m.prefillBatch(context.Background(), []*KVCache{cache}, chunks, e)
		if err != nil {
			t.Fatal(err)
		}
		want, err := m.PrefillBatch(context.Background(), []*KVCache{ref}, chunks)
		if err != nil {
			t.Fatal(err)
		}
		exactWorkerFloats(t, got, want)
		cache.Close()
		ref.Close()
	}
}

func TestEngineWorkerOwnership(t *testing.T) {
	m := workerTestModel(t, 2)
	engines := make([]*Engine, 2)
	for i := range engines {
		var err error
		engines[i], err = NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 8 << 20, KernelWorkers: i + 2})
		if err != nil {
			t.Fatal(err)
		}
		defer engines[i].Close()
	}
	sampling := defaultSampling()
	sampling.MaxTokens = 4
	prompt := strings.Repeat("abc", 20)
	want, err := Generate(context.Background(), m, Tokenizer{}, prompt, sampling)
	if err != nil {
		t.Fatal(err)
	}
	results := make(chan engineReply, 2)
	for _, e := range engines {
		go func(e *Engine) {
			out, err := e.Generate(context.Background(), prompt, sampling)
			results <- engineReply{out, err}
		}(e)
	}
	for range engines {
		r := <-results
		if r.Err != nil || !reflect.DeepEqual(r.Completion, want) {
			t.Fatalf("shared model output changed: %+v", r)
		}
	}
	for i, e := range engines {
		if e.Stats().KernelWorkers != int64(i+2) {
			t.Fatalf("unexpected worker count: %+v", e.Stats())
		}
	}
	engines[0].Close()
	if engines[0].Stats().KernelWorkers != 0 || engines[1].Stats().KernelWorkers != 3 {
		t.Fatal("closing one engine affected another engine's workers")
	}
	got, err := engines[1].Generate(context.Background(), prompt, sampling)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("surviving engine failed: %v", err)
	}
	r := httptest.NewRecorder()
	engines[1].Handler().ServeHTTP(r, httptest.NewRequest("GET", "/metrics", nil))
	if !strings.Contains(r.Body.String(), "monolith_kernel_workers 3\n") {
		t.Fatal("worker ownership missing from metrics")
	}
	engines[1].Close()
	if engines[1].Stats().KernelWorkers != 0 {
		t.Fatal("Close returned before worker teardown")
	}
}

func TestEngineWorkerConfiguration(t *testing.T) {
	m := testModel(t)
	c := EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 1 << 20}
	for _, invalid := range []int{-1, 257} {
		c.KernelWorkers = invalid
		if e, err := NewEngine(m, Tokenizer{}, c); err == nil {
			e.Close()
			t.Fatalf("accepted %d workers", invalid)
		}
	}
	old := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(old)
	c.KernelWorkers = 0
	e, err := newEngine(m, Tokenizer{}, c)
	if err != nil {
		t.Fatal(err)
	}
	runtime.GOMAXPROCS(4) // Capacity was captured at construction, not each kernel.
	go e.run()
	await(t, "workers ready", func() bool { return e.Stats().KernelWorkers == 2 })
	e.Close()
	if e.Config.KernelWorkers != 2 || e.Stats().KernelWorkers != 0 {
		t.Fatal("automatic capacity changed or idle close leaked workers")
	}
	c.KernelWorkers = 1
	e, err = NewEngine(m, Tokenizer{}, c)
	if err != nil {
		t.Fatal(err)
	}
	e.Close()
	if e.Stats().KernelWorkers != 0 {
		t.Fatal("inline engine started background compute workers")
	}
}

func BenchmarkInferenceLinearWorkers(b *testing.B) {
	for _, n := range []int{1, 4, 64} {
		for _, mode := range []string{"spawn", "reuse"} {
			b.Run(fmt.Sprintf("rows%d/%s", n, mode), func(b *testing.B) {
				var e *kernelExecutor
				if mode == "reuse" {
					e = newKernelExecutor(runtime.GOMAXPROCS(0))
					defer e.close()
				}
				w := &Param{Rows: 704, Cols: 256, Data: make([]float32, 704*256)}
				x, y := make([]float32, n*256), make([]float32, n*704)
				for i := range w.Data {
					w.Data[i] = float32(i%23-11) / 32
				}
				for i := range x {
					x[i] = float32(i%19-9) / 16
				}
				linearIntoWithExecutor(e, y, x, w, n)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					linearIntoWithExecutor(e, y, x, w, n)
				}
				b.StopTimer()
			})
		}
	}
}

func BenchmarkInferenceBatchWorkers(b *testing.B) {
	for _, rows := range []int{1, 16} {
		for _, mode := range []string{"spawn", "reuse"} {
			b.Run(fmt.Sprintf("rows%d/%s", rows, mode), func(b *testing.B) {
				c, err := preset("small")
				if err != nil {
					b.Fatal(err)
				}
				m, err := NewModel(c, 19)
				if err != nil {
					b.Fatal(err)
				}
				m.DropOptimizer()
				var e *kernelExecutor
				if mode == "reuse" {
					e = newKernelExecutor(runtime.GOMAXPROCS(0))
					defer e.close()
				}
				caches, chunks := make([]*KVCache, 4), make([][]int, 4)
				for i := range caches {
					caches[i] = m.NewCache()
					defer caches[i].Close()
					if _, err := m.PrefillBatch(context.Background(), caches[i:i+1], [][]int{prefillTokens(96, i)}); err != nil {
						b.Fatal(err)
					}
					chunks[i] = prefillTokens(rows, i+3)
				}
				run := func() {
					for _, cache := range caches {
						cache.Pos = 96 // Fixed prefixes and writes; keep already allocated pages.
					}
					if _, err := m.prefillBatch(context.Background(), caches, chunks, e); err != nil {
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
