package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"testing"
	"time"
)

func originalFP32Dispatch(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	outTiles := (out + 31) / 32
	tasks := ((n + 3) / 4) * outTiles
	e.parallel(tasks, n*in*out, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			r := (task / outTiles) * 4
			o0 := (task % outTiles) * 32
			o1 := min(o0+32, out)
			if r+4 <= n {
				x0, x1, x2, x3 := x[r*in:(r+1)*in], x[(r+1)*in:(r+2)*in], x[(r+2)*in:(r+3)*in], x[(r+3)*in:(r+4)*in]
				for o := o0; o < o1; o++ {
					wr := w.Data[o*in : (o+1)*in]
					var a, b, c, d float32
					for j, v := range wr {
						a += x0[j] * v
						b += x1[j] * v
						c += x2[j] * v
						d += x3[j] * v
					}
					y[r*out+o], y[(r+1)*out+o] = a, b
					y[(r+2)*out+o], y[(r+3)*out+o] = c, d
				}
			} else {
				for row := r; row < n; row++ {
					xr := x[row*in : (row+1)*in]
					for o := o0; o < o1; o++ {
						wr := w.Data[o*in : (o+1)*in]
						var sum float32
						for j, v := range xr {
							sum += v * wr[j]
						}
						y[row*out+o] = sum
					}
				}
			}
		}
	})
}

type pairedFP32Result struct {
	Name         string    `json:"name"`
	BeforeNS     []float64 `json:"before_ns"`
	AfterNS      []float64 `json:"after_ns"`
	PairedRatios []float64 `json:"paired_ratios"`
	MedianRatio  float64   `json:"median_ratio"`
	BeforeAllocs float64   `json:"before_allocs"`
	AfterAllocs  float64   `json:"after_allocs"`
}

func TestFP32DispatchPairedTiming(t *testing.T) {
	output := os.Getenv("MONOLITH_DISPATCH_COMPARE_OUTPUT")
	if output == "" {
		t.Skip("opt-in paired kernel timing")
	}
	if runtime.GOMAXPROCS(0) != 4 {
		t.Fatal("paired comparison requires four threads")
	}
	var results []pairedFP32Result
	for _, workers := range []int{0, 1, 4} {
		for _, shape := range [][3]int{{1, 256, 704}, {4, 256, 704}, {64, 256, 704}, {1, 1536, 8960}, {4, 1536, 8960}} {
			rows, cols, outputs := shape[0], shape[1], shape[2]
			var e *kernelExecutor
			if workers > 0 {
				e = newKernelExecutor(workers)
			}
			closed := e == nil
			t.Cleanup(func() {
				if !closed {
					e.close()
				}
			})
			x, w, _ := linearFixture(rows, cols, outputs)
			before, after := make([]float32, rows*outputs), make([]float32, rows*outputs)
			functions := []func(){
				func() { originalFP32Dispatch(e, before, x, w, rows) },
				func() { linearIntoWithExecutor(e, after, x, w, rows) },
			}
			for warmup := 0; warmup < 4; warmup++ {
				functions[0]()
				functions[1]()
			}
			exactWorkerFloats(t, before, after)
			runtime.GC()
			iterations := max(1, (1<<20)/(rows*cols*outputs))
			result := pairedFP32Result{Name: fmt.Sprintf("workers%d/rows%d/%dx%d", workers, rows, outputs, cols)}
			for trial := 0; trial < 24; trial++ {
				var elapsed [2]float64
				for position := 0; position < 2; position++ {
					variant := (trial + position) % 2
					start := time.Now()
					for i := 0; i < iterations; i++ {
						functions[variant]()
					}
					elapsed[variant] = float64(time.Since(start).Nanoseconds()) / float64(iterations)
				}
				result.BeforeNS = append(result.BeforeNS, elapsed[0])
				result.AfterNS = append(result.AfterNS, elapsed[1])
				result.PairedRatios = append(result.PairedRatios, elapsed[1]/elapsed[0])
			}
			ratios := append([]float64(nil), result.PairedRatios...)
			sort.Float64s(ratios)
			result.MedianRatio = (ratios[11] + ratios[12]) / 2
			result.BeforeAllocs = testing.AllocsPerRun(20, functions[0])
			result.AfterAllocs = testing.AllocsPerRun(20, functions[1])
			exactWorkerFloats(t, before, after)
			if e != nil {
				e.close()
			}
			closed = true
			results = append(results, result)
			t.Logf("%s: paired ratio %.3f, allocations %.0f -> %.0f", result.Name, result.MedianRatio, result.BeforeAllocs, result.AfterAllocs)
		}
	}
	encoded, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(output, append(encoded, '\n'), 0644); err != nil {
		t.Fatal(err)
	}
}
