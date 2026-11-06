package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"
)

type capturedAttentionExperimentJob struct {
	w                                 *decodeWorkspace
	c                                 Config
	dim, headDim, group, layer, tasks int
	plan                              inferenceAttentionPlan
}

type capturedAttentionExperiment struct {
	job capturedAttentionExperimentJob
	fn  func(int, int)
}

func (s *capturedAttentionExperiment) invoke(e *kernelExecutor, w *decodeWorkspace, c Config, layer, rows int, plan inferenceAttentionPlan) {
	if s.fn == nil {
		s.fn = func(lo, hi int) {
			j := s.job
			w, c, d, hd, group, layer, tasks, plan := j.w, j.c, j.dim, j.headDim, j.group, j.layer, j.tasks, j.plan
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
		}
	}
	s.job = capturedAttentionExperimentJob{w: w, c: c, dim: c.Dim, headDim: c.Dim / c.Heads, group: c.Heads / c.KVHeads, layer: layer, tasks: rows * c.Heads, plan: plan}
	e.parallel(plan.lanes, plan.work, s.fn)
	s.job = capturedAttentionExperimentJob{}
}

func transientAttentionExperiment(e *kernelExecutor, w *decodeWorkspace, c Config, layer, rows int, plan inferenceAttentionPlan) {
	value := attentionKernel{w: w, dim: c.Dim, heads: c.Heads, headDim: c.Dim / c.Heads, group: c.Heads / c.KVHeads, layer: layer, rows: rows, plan: plan}
	e.parallel(plan.lanes, plan.work, func(lo, hi int) {
		local := value
		attentionRange(&local, lo, hi)
	})
}

func snapshotHeadsAttentionExperiment(job *attentionKernel, lo, hi int) {
	w, d, heads, hd, group, layer, plan := job.w, job.dim, job.heads, job.headDim, job.group, job.layer, job.plan
	tasks := job.rows * heads
	for lane := lo; lane < hi; lane++ {
		scores := w.Scores[lane*plan.stride : (lane+1)*plan.stride]
		for task := tasks * lane / plan.lanes; task < tasks*(lane+1)/plan.lanes; task++ {
			i, h := task/heads, task%heads
			cache := w.rowCaches[i]
			kh, qi := h/group, i*d+h*hd
			kv := func(pos int) ([]float32, []float32) {
				k, v := cache.row(layer, pos)
				return k[kh*hd : (kh+1)*hd], v[kh*hd : (kh+1)*hd]
			}
			attend(w.Q[qi:qi+hd], w.A[qi:qi+hd], scores, w.positions[i]+1, kv)
		}
	}
}

func attentionExperimentPaths(e *kernelExecutor, w *decodeWorkspace, c Config, layer, rows int, plan inferenceAttentionPlan) ([]string, []func(), *capturedAttentionExperiment) {
	captured := &capturedAttentionExperiment{}
	w.attend(e, c, layer, rows, plan)
	standard := w.attentionFn
	localHeads := func(lo, hi int) { snapshotHeadsAttentionExperiment(&w.attention, lo, hi) }
	matched := func(fn func(int, int)) {
		previous := w.attentionFn
		w.attentionFn = fn
		w.attend(e, c, layer, rows, plan)
		w.attentionFn = previous
	}
	return []string{"reference", "transient-helper", "cached-captured", "cached-helper", "matched-helper", "matched-local-heads"}, []func(){
		func() { originalAttentionInvocation(e, w, c, layer, rows, plan) },
		func() { transientAttentionExperiment(e, w, c, layer, rows, plan) },
		func() { captured.invoke(e, w, c, layer, rows, plan) },
		func() { w.attend(e, c, layer, rows, plan) },
		func() { matched(standard) },
		func() { matched(localHeads) },
	}, captured
}

func TestAttentionCallpathExperiments(t *testing.T) {
	for _, kvHeads := range []int{1, 2, 6} {
		for _, length := range []int{31, 33, 257} {
			for _, workers := range []int{1, 4} {
				for _, rows := range []int{1, 17} {
					t.Run(fmt.Sprintf("kv%d/length%d/workers%d/rows%d", kvHeads, length, workers, rows), func(t *testing.T) {
						c := Config{Dim: 60, Heads: 6, KVHeads: kvHeads, Layers: 2, Context: 301}
						e := newKernelExecutor(workers)
						defer e.close()
						w, cache := attentionLaneFixture(c, rows, length)
						defer cache.Close()
						plan := planInferenceAttention(e, c, rows, length)
						w.Scores = make([]float32, plan.lanes*plan.stride)
						for layer := 0; layer < 2; layer++ {
							names, paths, captured := attentionExperimentPaths(e, w, c, layer, rows, plan)
							paths[0]()
							want := append([]float32(nil), w.A...)
							for i, fn := range paths {
								fn()
								exactWorkerFloats(t, w.A, want)
								wantAllocs := float64(1)
								if i >= 2 {
									wantAllocs = 0
								}
								if allocs := testing.AllocsPerRun(3, fn); allocs != wantAllocs {
									t.Fatalf("%s: %.0f allocations, want %.0f", names[i], allocs, wantAllocs)
								}
								if captured.job != (capturedAttentionExperimentJob{}) || w.attention != (attentionKernel{}) {
									t.Fatal("callpath retained state")
								}
							}
							job := attentionKernel{w: w, dim: c.Dim, heads: c.Heads, headDim: c.Dim / c.Heads, group: c.Heads / c.KVHeads, layer: layer, rows: rows, plan: plan}
							snapshotHeadsAttentionExperiment(&job, 0, plan.lanes)
							exactWorkerFloats(t, w.A, want)
						}
					})
				}
			}
		}
	}
}

func TestAttentionCallpathTiming(t *testing.T) {
	output := os.Getenv("MONOLITH_ATTENTION_CALLPATH_OUTPUT")
	if output == "" {
		t.Skip("opt-in callpath comparison")
	}
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	const calls, trials = 64, 32
	c := Config{Dim: 4096, Heads: 32, KVHeads: 8, Layers: 1, Context: 8192}
	e := newKernelExecutor(4)
	defer e.close()
	w, cache := attentionLaneFixture(c, 1, 128)
	defer cache.Close()
	plan := planInferenceAttention(e, c, 1, 128)
	w.Scores = make([]float32, plan.lanes*plan.stride)
	names, paths, captured := attentionExperimentPaths(e, w, c, 0, 1, plan)
	paths[0]()
	want := append([]float32(nil), w.A...)
	artifact := struct {
		Protocol, StartedUTC, FinishedUTC, GoVersion, GOOS, GOARCH, LoadBefore, LoadAfter string
		CallsPerBlock, TrialsPerKind, Workers, Procs                                      int
		Hashes                                                                            map[string]string
		Allocations                                                                       map[string]float64
		Samples                                                                           []attentionControlSample
	}{Protocol: "attention-callpath-v2", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, CallsPerBlock: calls, TrialsPerKind: trials, Workers: 4, Procs: 4, Hashes: make(map[string]string), Allocations: make(map[string]float64)}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{executable, "cmd/monolith/monolith.go", "cmd/monolith/attention_paired_test.go", "cmd/monolith/executor_experiments_test.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		artifact.Hashes[filepath.Base(path)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	for warm := 0; warm < 16; warm++ {
		for _, fn := range paths {
			fn()
		}
	}
	pairs := [][2]int{{0, 0}, {3, 3}, {0, 1}, {0, 2}, {0, 3}, {2, 3}, {3, 4}, {4, 4}, {5, 5}, {4, 5}}
	rng := rand.New(rand.NewSource(20261106))
	load := func() string {
		data, err := exec.Command("uptime").Output()
		if err != nil {
			return err.Error()
		}
		return string(data)
	}
	artifact.LoadBefore = load()
	artifact.StartedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	for trial := 0; trial < trials; trial++ {
		for _, kind := range rng.Perm(len(pairs)) {
			pair := pairs[kind]
			sample := attentionControlSample{Kind: names[pair[0]] + "/" + names[pair[1]], Trial: trial, FirstVariant: trial % 2}
			for leg := 0; leg < 2; leg++ {
				variant := (trial + leg) % 2
				fn := paths[pair[variant]]
				start := time.Now()
				for call := 0; call < calls; call++ {
					fn()
				}
				sample.NSPerCall[variant] = float64(time.Since(start).Nanoseconds()) / calls
				exactWorkerFloats(t, w.A, want)
				if captured.job != (capturedAttentionExperimentJob{}) || w.attention != (attentionKernel{}) {
					t.Fatal("callpath retained state")
				}
			}
			sample.Ratio = sample.NSPerCall[1] / sample.NSPerCall[0]
			artifact.Samples = append(artifact.Samples, sample)
		}
	}
	artifact.FinishedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	artifact.LoadAfter = load()
	for i, fn := range paths {
		artifact.Allocations[names[i]] = testing.AllocsPerRun(20, fn)
	}
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
	for _, pair := range pairs {
		name := names[pair[0]] + "/" + names[pair[1]]
		var ratios []float64
		for _, sample := range artifact.Samples {
			if sample.Kind == name {
				ratios = append(ratios, sample.Ratio)
			}
		}
		sort.Float64s(ratios)
		t.Logf("%s: median %.4f, range %.4f..%.4f", name, (ratios[15]+ratios[16])/2, ratios[0], ratios[31])
	}
}

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
