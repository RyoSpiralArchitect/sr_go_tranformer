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

// octOutputFP32Range shares each remainder-row input load across four output
// accumulators; each output keeps its ascending-index accumulation order.
func octOutputFP32Range(job *fp32Kernel, lo, hi int) {
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
				for ; o+7 < o1; o += 8 {
					w0, w1 := w.Data[o*in:(o+1)*in], w.Data[(o+1)*in:(o+2)*in]
					w2, w3 := w.Data[(o+2)*in:(o+3)*in], w.Data[(o+3)*in:(o+4)*in]
					w4, w5 := w.Data[(o+4)*in:(o+5)*in], w.Data[(o+5)*in:(o+6)*in]
					w6, w7 := w.Data[(o+6)*in:(o+7)*in], w.Data[(o+7)*in:(o+8)*in]
					var a, b, c, d, e2, f, g, h float32
					for j, v := range xr {
						a += v * w0[j]
						b += v * w1[j]
						c += v * w2[j]
						d += v * w3[j]
						e2 += v * w4[j]
						f += v * w5[j]
						g += v * w6[j]
						h += v * w7[j]
					}
					y[row*out+o], y[row*out+o+1], y[row*out+o+2], y[row*out+o+3] = a, b, c, d
					y[row*out+o+4], y[row*out+o+5], y[row*out+o+6], y[row*out+o+7] = e2, f, g, h
				}
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

func TestOctOutputFP32Exact(t *testing.T) {
	for _, rows := range []int{1, 2, 3, 5, 7, 9} {
		for _, cols := range []int{1, 3, 31, 32, 33, 255} {
			for _, outputs := range []int{1, 2, 3, 4, 5, 31, 32, 33, 63, 65} {
				x, w, _ := linearFixture(rows, cols, outputs)
				want, got := make([]float32, rows*outputs), make([]float32, rows*outputs)
				tasks := ((rows + 3) / 4) * ((outputs + 31) / 32)
				frozenSingleOutputFP32Range(&fp32Kernel{x: x, y: want, w: w, n: rows}, 0, tasks)
				job := fp32Kernel{x: x, y: got, w: w, n: rows}
				for task := 0; task < tasks; task++ {
					octOutputFP32Range(&job, task, task+1)
				}
				for i := range want {
					if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
						t.Fatalf("rows%d cols%d outputs%d index%d", rows, cols, outputs, i)
					}
				}
			}
		}
	}
}

func TestOctOutputFP32Timing(t *testing.T) {
	output := os.Getenv("MONOLITH_FP32_OCT_OUTPUT")
	if output == "" {
		t.Skip("opt-in oct output comparison")
	}
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	artifact := struct {
		Protocol, GoVersion, GOOS, GOARCH string
		Hashes                            map[string]string
		Results                           []pairedFP32Result
	}{Protocol: "fp32-oct-outputs-v1", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, Hashes: make(map[string]string)}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{executable, "cmd/monolith/monolith.go", "cmd/monolith/fp32_outputs4_test.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		artifact.Hashes[filepath.Base(path)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	for _, workers := range []int{1, 4} {
		for _, shape := range [][3]int{{1, 256, 704}, {1, 704, 256}, {1, 256, 258}, {3, 257, 705}, {5, 256, 704}, {1, 1536, 8960}} {
			e := newKernelExecutor(workers)
			x, w, _ := linearFixture(shape[0], shape[1], shape[2])
			before, after := make([]float32, shape[0]*shape[2]), make([]float32, shape[0]*shape[2])
			runners := []*fp32OutputRunner{newFP32OutputRunner(quadOutputFP32Range), newFP32OutputRunner(octOutputFP32Range)}
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
			t.Logf("%s: ratio %.4f", result.Name, result.MedianRatio)
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
