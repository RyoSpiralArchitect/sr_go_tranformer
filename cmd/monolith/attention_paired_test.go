package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
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

type attentionControlSample struct {
	Kind                string
	Trial, FirstVariant int
	NSPerCall           [2]float64
	Ratio               float64
}

func TestAttentionInvocationControlledTiming(t *testing.T) {
	output := os.Getenv("MONOLITH_ATTENTION_CONTROL_OUTPUT")
	if output == "" {
		t.Skip("opt-in controlled timing")
	}
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	const calls, trials = 4, 16
	c := Config{Dim: 4096, Heads: 32, KVHeads: 8, Layers: 1, Context: 8192}
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
	artifact := struct {
		Protocol, StartedUTC, FinishedUTC, GoVersion, GOOS, GOARCH string
		CallsPerBlock, TrialsPerKind, Workers, Procs               int
		Hashes                                                     map[string]string
		Samples                                                    []attentionControlSample
	}{Protocol: "attention-small-controlled-v1", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, CallsPerBlock: calls, TrialsPerKind: trials, Workers: 4, Procs: 4, Hashes: make(map[string]string)}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{executable, "cmd/monolith/monolith.go", "cmd/monolith/attention_paired_test.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		artifact.Hashes[filepath.Base(path)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	for i := 0; i < 8; i++ {
		before()
		after()
	}
	exactWorkerFloats(t, w.A, want)
	types := []struct {
		name      string
		functions [2]func()
	}{
		{"reference/reference", [2]func(){before, before}},
		{"candidate/candidate", [2]func(){after, after}},
		{"reference/candidate", [2]func(){before, after}},
	}
	rng := rand.New(rand.NewSource(42))
	artifact.StartedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	for trial := 0; trial < trials; trial++ {
		for _, kind := range rng.Perm(len(types)) {
			pair := types[kind]
			sample := attentionControlSample{Kind: pair.name, Trial: trial, FirstVariant: trial % 2}
			for leg := 0; leg < 2; leg++ {
				variant := (trial + leg) % 2
				fn := pair.functions[variant]
				start := time.Now()
				for call := 0; call < calls; call++ {
					fn()
				}
				sample.NSPerCall[variant] = float64(time.Since(start).Nanoseconds()) / calls
				exactWorkerFloats(t, w.A, want)
			}
			sample.Ratio = sample.NSPerCall[1] / sample.NSPerCall[0]
			artifact.Samples = append(artifact.Samples, sample)
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
	for _, kind := range types {
		var ratios []float64
		for _, sample := range artifact.Samples {
			if sample.Kind == kind.name {
				ratios = append(ratios, sample.Ratio)
			}
		}
		sort.Float64s(ratios)
		t.Logf("%s: median %.4f, range %.4f..%.4f", kind.name, (ratios[7]+ratios[8])/2, ratios[0], ratios[15])
	}
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
