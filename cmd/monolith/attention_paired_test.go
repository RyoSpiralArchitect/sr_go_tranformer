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

func originalAttentionInvocation(e *kernelExecutor, w *decodeWorkspace, c Config, layer, rows int, plan inferenceAttentionPlan) {
	d, hd, group := c.Dim, c.Dim/c.Heads, c.Heads/c.KVHeads
	tasks := rows * c.Heads
	e.parallel(plan.lanes, plan.work, func(lo, hi int) {
		for lane := lo; lane < hi; lane++ {
			scores := w.Scores[lane*plan.stride : (lane+1)*plan.stride]
			for task := tasks * lane / plan.lanes; task < tasks*(lane+1)/plan.lanes; task++ {
				i, h := task/c.Heads, task%c.Heads
				cache := w.rowCaches[i]
				kh, qi := h/group, i*d+h*hd
				kv := func(pos int) ([]float32, []float32) {
					k, v := cache.row(layer, pos)
					return k[kh*hd : (kh+1)*hd], v[kh*hd : (kh+1)*hd]
				}
				attend(w.Q[qi:qi+hd], w.A[qi:qi+hd], scores, w.positions[i]+1, kv)
			}
		}
	})
}

func TestAttentionInvocationPairedTiming(t *testing.T) {
	output := os.Getenv("MONOLITH_ATTENTION_COMPARE_OUTPUT")
	if output == "" {
		t.Skip("opt-in paired timing")
	}
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	c := Config{Dim: 4096, Heads: 32, KVHeads: 8, Layers: 1, Context: 8192}
	var results []pairedFP32Result
	for _, workers := range []int{1, 4} {
		for _, shape := range [][2]int{{1, 128}, {16, 128}, {1, 2048}, {2, 2048}, {1, 8192}} {
			rows, length := shape[0], shape[1]
			e := newKernelExecutor(workers)
			w, cache := attentionLaneFixture(c, rows, length)
			closed := false
			t.Cleanup(func() {
				if !closed {
					e.close()
					cache.Close()
				}
			})
			plan := planInferenceAttention(e, c, rows, length)
			w.Scores = make([]float32, plan.lanes*plan.stride)
			before := func() { originalAttentionInvocation(e, w, c, 0, rows, plan) }
			after := func() { w.attend(e, c, 0, rows, plan) }
			before()
			want := append([]float32(nil), w.A...)
			after()
			exactWorkerFloats(t, w.A, want)
			result := pairedFP32Result{Name: fmt.Sprintf("workers%d/rows%d/prefix%d", workers, rows, length)}
			functions := []func(){before, after}
			for trial := 0; trial < 24; trial++ {
				var elapsed [2]float64
				for position := 0; position < 2; position++ {
					variant := (trial + position) % 2
					start := time.Now()
					functions[variant]()
					elapsed[variant] = float64(time.Since(start).Nanoseconds())
				}
				result.BeforeNS = append(result.BeforeNS, elapsed[0])
				result.AfterNS = append(result.AfterNS, elapsed[1])
				result.PairedRatios = append(result.PairedRatios, elapsed[1]/elapsed[0])
			}
			ratios := append([]float64(nil), result.PairedRatios...)
			sort.Float64s(ratios)
			result.MedianRatio = (ratios[11] + ratios[12]) / 2
			result.BeforeAllocs = testing.AllocsPerRun(20, before)
			result.AfterAllocs = testing.AllocsPerRun(20, after)
			exactWorkerFloats(t, w.A, want)
			e.close()
			cache.Close()
			closed = true
			results = append(results, result)
			t.Logf("%s: ratio %.3f, allocations %.0f -> %.0f", result.Name, result.MedianRatio, result.BeforeAllocs, result.AfterAllocs)
		}
	}
	encoded, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if _, err := file.Write(append(encoded, '\n')); err != nil {
		t.Fatal(err)
	}
}
