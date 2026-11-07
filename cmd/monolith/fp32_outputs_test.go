package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"
)

func frozenSingleOutputFP32Range(job *fp32Kernel, lo, hi int) {
	y, x, w, n := job.y, job.x, job.w, job.n
	in, out := w.Cols, w.Rows
	outTiles := (out + 31) / 32
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
				y[r*out+o], y[(r+1)*out+o], y[(r+2)*out+o], y[(r+3)*out+o] = a, b, c, d
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
}

func pairedOutputFP32Range(job *fp32Kernel, lo, hi int) {
	y, x, w, n := job.y, job.x, job.w, job.n
	in, out := w.Cols, w.Rows
	outTiles := (out + 31) / 32
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
				y[r*out+o], y[(r+1)*out+o], y[(r+2)*out+o], y[(r+3)*out+o] = a, b, c, d
			}
		} else {
			for row := r; row < n; row++ {
				xr := x[row*in : (row+1)*in]
				o := o0
				for ; o+1 < o1; o += 2 {
					w0, w1 := w.Data[o*in:(o+1)*in], w.Data[(o+1)*in:(o+2)*in]
					var a, b float32
					for j, v := range xr {
						a += v * w0[j]
						b += v * w1[j]
					}
					y[row*out+o], y[row*out+o+1] = a, b
				}
				for ; o < o1; o++ {
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
}

type fp32OutputRunner struct {
	job      fp32Kernel
	rangeFn  func(*fp32Kernel, int, int)
	callback func(int, int)
}

func newFP32OutputRunner(fn func(*fp32Kernel, int, int)) *fp32OutputRunner {
	r := &fp32OutputRunner{rangeFn: fn}
	r.callback = func(lo, hi int) { r.rangeFn(&r.job, lo, hi) }
	return r
}

func (r *fp32OutputRunner) invoke(e *kernelExecutor, y, x []float32, w *Param, n int) {
	tasks := ((n + 3) / 4) * ((w.Rows + 31) / 32)
	work := n * w.Cols * w.Rows
	r.job = fp32Kernel{x: x, y: y, w: w, n: n}
	if e.workers < 2 || tasks < 2 || work < 131072 {
		r.rangeFn(&r.job, 0, tasks)
	} else {
		e.parallel(tasks, work, r.callback)
	}
	r.job = fp32Kernel{}
}

func TestPairedOutputFP32Exact(t *testing.T) {
	for _, rows := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 17} {
		for _, cols := range []int{0, 1, 3, 31, 32, 33, 255} {
			for _, outputs := range []int{0, 1, 2, 3, 31, 32, 33, 63, 65} {
				x, w, _ := linearFixture(rows, cols, outputs)
				want, got := make([]float32, rows*outputs), make([]float32, rows*outputs)
				tasks := ((rows + 3) / 4) * ((outputs + 31) / 32)
				frozenSingleOutputFP32Range(&fp32Kernel{x: x, y: want, w: w, n: rows}, 0, tasks)
				job := fp32Kernel{x: x, y: got, w: w, n: rows}
				for task := 0; task < tasks; task++ {
					pairedOutputFP32Range(&job, task, task+1)
				}
				exactWorkerFloats(t, got, want)
				linearFP32Range(&job, 0, tasks)
				exactWorkerFloats(t, got, want)
			}
		}
	}
	for _, shape := range [][3]int{{1, 256, 704}, {3, 257, 705}, {4, 256, 704}, {5, 704, 257}, {7, 65, 513}} {
		for _, workers := range []int{1, 4} {
			e := newKernelExecutor(workers)
			x, w, _ := linearFixture(shape[0], shape[1], shape[2])
			want, got := make([]float32, shape[0]*shape[2]), make([]float32, shape[0]*shape[2])
			r := newFP32OutputRunner(pairedOutputFP32Range)
			tasks := ((shape[0] + 3) / 4) * ((shape[2] + 31) / 32)
			frozenSingleOutputFP32Range(&fp32Kernel{x: x, y: want, w: w, n: shape[0]}, 0, tasks)
			linearIntoWithExecutor(e, got, x, w, shape[0])
			exactWorkerFloats(t, got, want)
			r.invoke(e, got, x, w, shape[0])
			exactWorkerFloats(t, got, want)
			allocs := testing.AllocsPerRun(5, func() { r.invoke(e, got, x, w, shape[0]) })
			if allocs != 0 || r.job.x != nil || r.job.y != nil || r.job.w != nil || r.job.n != 0 {
				e.close()
				t.Fatal("paired kernel allocated or retained invocation state", allocs)
			}
			e.close()
		}
	}
}

func TestPairedOutputFP32SpecialBits(t *testing.T) {
	bits := []uint32{0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345}
	for _, rows := range []int{1, 3, 4, 5} {
		x, w, _ := linearFixture(rows, 33, 35)
		for i := range x {
			x[i] = math.Float32frombits(bits[i%len(bits)])
		}
		for i := range w.Data {
			w.Data[i] = math.Float32frombits(bits[(i*7+3)%len(bits)])
		}
		want, got := make([]float32, rows*35), make([]float32, rows*35)
		tasks := ((rows + 3) / 4) * 2
		frozenSingleOutputFP32Range(&fp32Kernel{x: x, y: want, w: w, n: rows}, 0, tasks)
		for _, fn := range []func(*fp32Kernel, int, int){pairedOutputFP32Range, linearFP32Range} {
			fn(&fp32Kernel{x: x, y: got, w: w, n: rows}, 0, tasks)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("special bits rows%d/element%d", rows, i)
				}
			}
		}
	}
}

func BenchmarkPairedOutputFP32(b *testing.B) {
	for _, workers := range []int{1, 4} {
		for _, shape := range [][3]int{{1, 256, 704}, {1, 704, 256}, {1, 256, 258}, {3, 257, 705}, {4, 256, 704}, {5, 256, 704}, {1, 1536, 8960}} {
			for _, variant := range []struct {
				name string
				fn   func(*fp32Kernel, int, int)
			}{{"single", frozenSingleOutputFP32Range}, {"paired", pairedOutputFP32Range}} {
				b.Run(fmt.Sprintf("workers%d/rows%d/%dx%d/%s", workers, shape[0], shape[2], shape[1], variant.name), func(b *testing.B) {
					e := newKernelExecutor(workers)
					defer e.close()
					x, w, _ := linearFixture(shape[0], shape[1], shape[2])
					y := make([]float32, shape[0]*shape[2])
					r := newFP32OutputRunner(variant.fn)
					r.invoke(e, y, x, w, shape[0])
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						r.invoke(e, y, x, w, shape[0])
					}
					b.StopTimer()
					runtime.KeepAlive(y)
				})
			}
		}
	}
}

func TestPairedOutputFP32Timing(t *testing.T) {
	output := os.Getenv("MONOLITH_FP32_OUTPUTS_OUTPUT")
	if output == "" {
		t.Skip("opt-in paired output comparison")
	}
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	artifact := struct {
		Protocol, GoVersion, GOOS, GOARCH string
		Hashes                            map[string]string
		Results                           []pairedFP32Result
	}{Protocol: "fp32-paired-outputs-v1", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, Hashes: make(map[string]string)}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{executable, "cmd/monolith/monolith.go", "cmd/monolith/fp32_outputs_test.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		artifact.Hashes[filepath.Base(path)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	for _, workers := range []int{1, 4} {
		for _, shape := range [][3]int{{1, 256, 704}, {1, 704, 256}, {1, 256, 258}, {3, 257, 705}, {4, 256, 704}, {5, 256, 704}, {1, 1536, 8960}} {
			e := newKernelExecutor(workers)
			x, w, _ := linearFixture(shape[0], shape[1], shape[2])
			before, after := make([]float32, shape[0]*shape[2]), make([]float32, shape[0]*shape[2])
			runners := []*fp32OutputRunner{newFP32OutputRunner(frozenSingleOutputFP32Range), newFP32OutputRunner(pairedOutputFP32Range)}
			functions := []func(){func() { runners[0].invoke(e, before, x, w, shape[0]) }, func() { runners[1].invoke(e, after, x, w, shape[0]) }}
			for warm := 0; warm < 4; warm++ {
				functions[0]()
				functions[1]()
			}
			exactWorkerFloats(t, before, after)
			iterations := max(1, (1<<22)/max(1, shape[0]*shape[1]*shape[2]))
			result := pairedFP32Result{Name: fmt.Sprintf("workers%d/rows%d/%dx%d", workers, shape[0], shape[2], shape[1])}
			for trial := 0; trial < 32; trial++ {
				var elapsed [2]float64
				for leg := 0; leg < 2; leg++ {
					variant := (trial + leg) % 2
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
			result.MedianRatio = (ratios[(len(ratios)-1)/2] + ratios[len(ratios)/2]) / 2
			result.BeforeAllocs = testing.AllocsPerRun(5, functions[0])
			result.AfterAllocs = testing.AllocsPerRun(5, functions[1])
			exactWorkerFloats(t, before, after)
			e.close()
			artifact.Results = append(artifact.Results, result)
			t.Logf("%s: ratio %.4f, allocations %.0f -> %.0f", result.Name, result.MedianRatio, result.BeforeAllocs, result.AfterAllocs)
		}
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
}
