package main

import (
	"fmt"
	"runtime"
	"testing"
	"time"
)

func TestAttentionInvocationAllocation(t *testing.T) {
	c := Config{Dim: 256, Heads: 4, KVHeads: 2, Layers: 1, Context: 512}
	for _, workers := range []int{1, 4} {
		for _, shape := range [][2]int{{1, 31}, {7, 257}} {
			t.Run(fmt.Sprintf("workers%d/rows%d/prefix%d", workers, shape[0], shape[1]), func(t *testing.T) {
				e := newKernelExecutor(workers)
				defer e.close()
				w, cache := attentionLaneFixture(c, shape[0], shape[1])
				defer cache.Close()
				plan := planInferenceAttention(e, c, shape[0], shape[1])
				w.Scores = make([]float32, plan.lanes*plan.stride)
				w.attend(e, c, 0, shape[0], plan)
				if allocs := testing.AllocsPerRun(20, func() { w.attend(e, c, 0, shape[0], plan) }); allocs != 0 {
					t.Fatalf("warm attention allocated %.1f objects", allocs)
				}
				if w.attention != (attentionKernel{}) {
					t.Fatal("attention retained invocation references after join")
				}
			})
		}
	}
}

func BenchmarkAttentionInvocationReuse(b *testing.B) {
	c := Config{Dim: 4096, Heads: 32, KVHeads: 8, Layers: 1, Context: 128}
	e := newKernelExecutor(4)
	defer e.close()
	w, cache := attentionLaneFixture(c, 1, 128)
	defer cache.Close()
	plan := planInferenceAttention(e, c, 1, 128)
	w.Scores = make([]float32, plan.lanes*plan.stride)
	before := func() { originalAttentionInvocation(e, w, c, 0, 1, plan) }
	after := func() { w.attend(e, c, 0, 1, plan) }
	before()
	want := append([]float32(nil), w.A...)
	after()
	exactWorkerFloats(b, w.A, want)
	for index, name := range []string{"referenceA", "candidateA", "candidateB", "referenceB"} {
		fn := after
		if index == 0 || index == 3 {
			fn = before
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fn()
			}
		})
	}
}

func TestAttentionInvocationWaitsBeforeClearing(t *testing.T) {
	c := Config{Dim: 256, Heads: 4, KVHeads: 2, Layers: 1, Context: 512}
	e := newKernelExecutor(4)
	w, cache := attentionLaneFixture(c, 7, 257)
	plan := planInferenceAttention(e, c, 7, 257)
	w.Scores = make([]float32, plan.lanes*plan.stride)
	originalAttentionInvocation(e, w, c, 0, 7, plan)
	want := append([]float32(nil), w.A...)
	started, release, done := make(chan struct{}, 4), make(chan struct{}), make(chan struct{})
	w.attentionFn = func(lo, hi int) {
		started <- struct{}{}
		<-release
		attentionRange(&w.attention, lo, hi)
	}
	go func() { w.attend(e, c, 0, 7, plan); close(done) }()
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
		<-done
		e.close()
		cache.Close()
	})
	for lane := 0; lane < plan.lanes; lane++ {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("attention lane did not start")
		}
	}
	select {
	case <-done:
		t.Fatal("attention returned with borrowed ranges outstanding")
	default:
	}
	if w.attention.w != w {
		t.Fatal("attention invocation cleared while workers borrowed it")
	}
	close(release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("attention join did not finish")
	}
	if w.attention != (attentionKernel{}) {
		t.Fatal("attention invocation retained after join")
	}
	exactWorkerFloats(t, w.A, want)
	w.clearReferences()
	assertWorkspaceReferencesCleared(t, w)
}

func callerParticipatingDispatch(e *kernelExecutor, n, work int, fn func(int, int)) {
	workers := min(e.workers, n)
	if workers < 2 || work < 131072 {
		fn(0, n)
		return
	}
	for w := 0; w < workers-1; w++ {
		e.jobs <- kernelJob{lo: n * w / workers, hi: n * (w + 1) / workers, fn: fn}
	}
	fn(n*(workers-1)/workers, n)
	for w := 0; w < workers-1; w++ {
		<-e.completed
	}
}

func TestWorkerLinearSteadyAllocation(t *testing.T) {
	for _, workers := range []int{1, 2, 4} {
		for _, rows := range []int{1, 4, 17} {
			t.Run(fmt.Sprintf("workers%d/rows%d", workers, rows), func(t *testing.T) {
				e := newKernelExecutor(workers)
				defer e.close()
				x, w, _ := linearFixture(rows, 513, 259)
				y := make([]float32, rows*w.Rows)
				linearIntoWithExecutor(e, y, x, w, rows)
				allocs := testing.AllocsPerRun(20, func() {
					linearIntoWithExecutor(e, y, x, w, rows)
				})
				if allocs != 0 {
					t.Fatalf("warm FP32 kernel allocated %.1f objects", allocs)
				}
				if e.linear.x != nil || e.linear.y != nil || e.linear.w != nil || e.linear.n != 0 {
					t.Fatal("executor retained FP32 inputs after join")
				}
			})
		}
	}
}

func BenchmarkKernelDispatchHypotheses(b *testing.B) {
	for _, columns := range []int{256, 4096} {
		for _, mode := range []string{"workers", "caller"} {
			b.Run(fmt.Sprintf("columns%d/%s", columns, mode), func(b *testing.B) {
				e := newKernelExecutor(4)
				defer e.close()
				x, w, _ := linearFixture(1, columns, 704)
				y := make([]float32, w.Rows)
				fn := func(lo, hi int) {
					for o := lo; o < hi; o++ {
						var sum float32
						for j, v := range x {
							sum += v * w.Data[o*columns+j]
						}
						y[o] = sum
					}
				}
				run := func() { e.parallel(w.Rows, columns*w.Rows, fn) }
				if mode == "caller" {
					run = func() { callerParticipatingDispatch(e, w.Rows, columns*w.Rows, fn) }
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run()
				}
				b.StopTimer()
				runtime.KeepAlive(y)
			})
		}
	}
}

func BenchmarkLinearWorkerWidths(b *testing.B) {
	for _, rows := range []int{1, 4, 64} {
		for _, workers := range []int{1, 2, 4} {
			b.Run(fmt.Sprintf("rows%d/workers%d", rows, workers), func(b *testing.B) {
				e := newKernelExecutor(workers)
				defer e.close()
				x, w, _ := linearFixture(rows, 256, 704)
				y := make([]float32, rows*w.Rows)
				linearIntoWithExecutor(e, y, x, w, rows)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					linearIntoWithExecutor(e, y, x, w, rows)
				}
				b.StopTimer()
			})
		}
	}
}
