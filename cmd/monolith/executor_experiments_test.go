package main

import (
	"fmt"
	"runtime"
	"testing"
)

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
