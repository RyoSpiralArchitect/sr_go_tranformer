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

// pairedOutputQ8Into pairs two output columns for the last single remainder
// row so each activation load serves two dot products. Every output keeps its
// original per-group accumulator order, scalar tail and scale application.
func pairedOutputQ8Into(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	if q8NeedsWideSum(x[:n*in]) {
		linearQ8WideInto(e, y, x, w, n)
		return
	}
	groups, tiles := (in+q8Block-1)/q8Block, (out+31)/32
	e.parallel(((n+3)/4)*tiles, n*in*out, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			r, o0 := (task/tiles)*4, (task%tiles)*32
			o1 := min(o0+32, out)
			if r+4 <= n {
				for o := o0; o < o1; o++ {
					wr := w.Q8.Values[o*in : (o+1)*in]
					scales := w.Q8.Scales[o*groups : (o+1)*groups]
					x0, x1 := x[r*in:(r+1)*in], x[(r+1)*in:(r+2)*in]
					x2, x3 := x[(r+2)*in:(r+3)*in], x[(r+3)*in:(r+4)*in]
					var s0, s1, s2, s3 float32
					for g, scale := range scales {
						var a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 float32
						j, end := g*q8Block, min((g+1)*q8Block, in)
						for ; j+4 <= end; j += 4 {
							v0, v1, v2, v3 := float32(wr[j]), float32(wr[j+1]), float32(wr[j+2]), float32(wr[j+3])
							a0 += x0[j] * v0
							b0 += x0[j+1] * v1
							c0 += x0[j+2] * v2
							d0 += x0[j+3] * v3
							a1 += x1[j] * v0
							b1 += x1[j+1] * v1
							c1 += x1[j+2] * v2
							d1 += x1[j+3] * v3
							a2 += x2[j] * v0
							b2 += x2[j+1] * v1
							c2 += x2[j+2] * v2
							d2 += x2[j+3] * v3
							a3 += x3[j] * v0
							b3 += x3[j+1] * v1
							c3 += x3[j+2] * v2
							d3 += x3[j+3] * v3
						}
						p0, p1, p2, p3 := (a0+b0)+(c0+d0), (a1+b1)+(c1+d1), (a2+b2)+(c2+d2), (a3+b3)+(c3+d3)
						for ; j < end; j++ {
							v := float32(wr[j])
							p0 += x0[j] * v
							p1 += x1[j] * v
							p2 += x2[j] * v
							p3 += x3[j] * v
						}
						s0 += p0 * scale
						s1 += p1 * scale
						s2 += p2 * scale
						s3 += p3 * scale
					}
					y[r*out+o], y[(r+1)*out+o] = s0, s1
					y[(r+2)*out+o], y[(r+3)*out+o] = s2, s3
				}
			} else {
				row := r
				if row+2 <= n {
					for o := o0; o < o1; o++ {
						wr := w.Q8.Values[o*in : (o+1)*in]
						scales := w.Q8.Scales[o*groups : (o+1)*groups]
						x0, x1 := x[row*in:(row+1)*in], x[(row+1)*in:(row+2)*in]
						var s0, s1 float32
						for g, scale := range scales {
							j, end := g*q8Block, min((g+1)*q8Block, in)
							if end-j == 64 {
								p0, p1 := dotQ8PairBlock64((*[64]float32)(x0[j:end]), (*[64]float32)(x1[j:end]), (*[64]int8)(wr[j:end]))
								s0 += p0 * scale
								s1 += p1 * scale
								continue
							}
							var a0, b0, c0, d0, a1, b1, c1, d1 float32
							for ; j+4 <= end; j += 4 {
								v0, v1, v2, v3 := float32(wr[j]), float32(wr[j+1]), float32(wr[j+2]), float32(wr[j+3])
								a0 += x0[j] * v0
								b0 += x0[j+1] * v1
								c0 += x0[j+2] * v2
								d0 += x0[j+3] * v3
								a1 += x1[j] * v0
								b1 += x1[j+1] * v1
								c1 += x1[j+2] * v2
								d1 += x1[j+3] * v3
							}
							p0, p1 := (a0+b0)+(c0+d0), (a1+b1)+(c1+d1)
							for ; j < end; j++ {
								v := float32(wr[j])
								p0 += x0[j] * v
								p1 += x1[j] * v
							}
							s0 += p0 * scale
							s1 += p1 * scale
						}
						y[row*out+o], y[(row+1)*out+o] = s0, s1
					}
					row += 2
				}
				for ; row < min(r+4, n); row++ {
					xr := x[row*in : (row+1)*in]
					o := o0
					for ; o+1 < o1; o += 2 {
						w0, w1 := w.Q8.Values[o*in:(o+1)*in], w.Q8.Values[(o+1)*in:(o+2)*in]
						s0, s1 := w.Q8.Scales[o*groups:(o+1)*groups], w.Q8.Scales[(o+1)*groups:(o+2)*groups]
						var sum0, sum1 float32
						for g := range s0 {
							j, end := g*q8Block, min((g+1)*q8Block, in)
							var a0, b0, c0, d0, a1, b1, c1, d1 float32
							for ; j+4 <= end; j += 4 {
								v0, v1, v2, v3 := xr[j], xr[j+1], xr[j+2], xr[j+3]
								a0 += v0 * float32(w0[j])
								a1 += v0 * float32(w1[j])
								b0 += v1 * float32(w0[j+1])
								b1 += v1 * float32(w1[j+1])
								c0 += v2 * float32(w0[j+2])
								c1 += v2 * float32(w1[j+2])
								d0 += v3 * float32(w0[j+3])
								d1 += v3 * float32(w1[j+3])
							}
							p0, p1 := (a0+b0)+(c0+d0), (a1+b1)+(c1+d1)
							for ; j < end; j++ {
								v := xr[j]
								p0 += v * float32(w0[j])
								p1 += v * float32(w1[j])
							}
							sum0 += p0 * s0[g]
							sum1 += p1 * s1[g]
						}
						y[row*out+o], y[row*out+o+1] = sum0, sum1
					}
					for ; o < o1; o++ {
						wr := w.Q8.Values[o*in : (o+1)*in]
						scales := w.Q8.Scales[o*groups : (o+1)*groups]
						var sum float32
						for g, scale := range scales {
							j, end := g*q8Block, min((g+1)*q8Block, in)
							var a, b, c, d float32
							for ; j+4 <= end; j += 4 {
								a += xr[j] * float32(wr[j])
								b += xr[j+1] * float32(wr[j+1])
								c += xr[j+2] * float32(wr[j+2])
								d += xr[j+3] * float32(wr[j+3])
							}
							part := (a + b) + (c + d)
							for ; j < end; j++ {
								part += xr[j] * float32(wr[j])
							}
							sum += part * scale
						}
						y[row*out+o] = sum
					}
				}
			}
		}
	})
}

func TestPairedOutputQ8Exact(t *testing.T) {
	for _, rows := range []int{1, 2, 3, 4, 5, 7, 9} {
		for _, cols := range []int{1, 3, 63, 64, 65, 129, 255, 256} {
			for _, outputs := range []int{1, 2, 3, 31, 32, 33, 63, 65} {
				x, dense, _ := linearFixture(rows, cols, outputs)
				w := quantizeTestParam(t, dense)
				want := make([]float32, rows*outputs)
				baselineLinearQ8Into(nil, want, x, w, rows)
				got := make([]float32, rows*outputs+2)
				got[0], got[len(got)-1] = -777, 777
				pairedOutputQ8Into(nil, got[1:len(got)-1], x, w, rows)
				if got[0] != -777 || got[len(got)-1] != 777 {
					t.Fatal("sentinel changed")
				}
				for i := range want {
					if math.Float32bits(want[i]) != math.Float32bits(got[i+1]) {
						t.Fatalf("rows%d cols%d outputs%d index%d", rows, cols, outputs, i)
					}
				}
			}
		}
	}
	for _, shape := range [][3]int{{1, 256, 704}, {3, 257, 705}, {5, 704, 257}, {7, 65, 513}, {1, 1536, 8960}} {
		x, dense, _ := linearFixture(shape[0], shape[1], shape[2])
		w := quantizeTestParam(t, dense)
		want := make([]float32, shape[0]*shape[2])
		baselineLinearQ8Into(nil, want, x, w, shape[0])
		for _, workers := range []int{1, 4} {
			e := newKernelExecutor(workers)
			got := make([]float32, shape[0]*shape[2])
			pairedOutputQ8Into(e, got, x, w, shape[0])
			e.close()
			for i := range want {
				if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
					t.Fatalf("shape %v workers %d index %d", shape, workers, i)
				}
			}
		}
	}
}

func TestPairedOutputQ8SpecialBits(t *testing.T) {
	bits := []uint32{0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000}
	for _, rows := range []int{1, 3, 5} {
		x, dense, _ := linearFixture(rows, 129, 35)
		w := quantizeTestParam(t, dense)
		for i := range x {
			x[i] = math.Float32frombits(bits[i%len(bits)]) * 0.5
		}
		for i := range w.Q8.Scales {
			if i%5 == 0 {
				w.Q8.Scales[i] = 0
			}
			if i%11 == 0 {
				w.Q8.Scales[i] = math.SmallestNonzeroFloat32
			}
		}
		want := make([]float32, rows*35)
		baselineLinearQ8Into(nil, want, x, w, rows)
		got := make([]float32, rows*35)
		pairedOutputQ8Into(nil, got, x, w, rows)
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("special bits rows%d index%d", rows, i)
			}
		}
	}
}

type pairedQ8Result struct {
	Name         string    `json:"name"`
	BeforeNS     []float64 `json:"before_ns"`
	AfterNS      []float64 `json:"after_ns"`
	PairedRatios []float64 `json:"paired_ratios"`
	MedianRatio  float64   `json:"median_ratio"`
	BeforeAllocs float64   `json:"before_allocs"`
	AfterAllocs  float64   `json:"after_allocs"`
}

func TestPairedOutputQ8Timing(t *testing.T) {
	output := os.Getenv("MONOLITH_Q8_OUTPUTS_OUTPUT")
	if output == "" {
		t.Skip("opt-in paired output comparison")
	}
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	artifact := struct {
		Protocol, GoVersion, GOOS, GOARCH string
		Hashes                            map[string]string
		Results                           []pairedQ8Result
	}{Protocol: "q8-paired-outputs-v1", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, Hashes: make(map[string]string)}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{executable, "cmd/monolith/monolith.go", "cmd/monolith/q8_outputs_test.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		artifact.Hashes[filepath.Base(path)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	for _, workers := range []int{1, 4} {
		for _, shape := range [][3]int{{1, 4096, 14336}, {1, 2048, 5632}, {3, 257, 705}, {5, 704, 257}, {4, 256, 704}, {1, 1536, 8960}} {
			e := newKernelExecutor(workers)
			x, dense, _ := linearFixture(shape[0], shape[1], shape[2])
			w := quantizeTestParam(t, dense)
			before, after := make([]float32, shape[0]*shape[2]), make([]float32, shape[0]*shape[2])
			functions := []func(){
				func() { linearQ8Into(e, before, x, w, shape[0]) },
				func() { pairedOutputQ8Into(e, after, x, w, shape[0]) },
			}
			for warm := 0; warm < 4; warm++ {
				functions[0]()
				functions[1]()
			}
			exactWorkerFloats(t, before, after)
			iterations := max(1, (1<<22)/max(1, shape[0]*shape[1]*shape[2]))
			result := pairedQ8Result{Name: fmt.Sprintf("workers%d/rows%d/%dx%d", workers, shape[0], shape[2], shape[1])}
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
			artifact.Results = append(artifact.Results, result)
			t.Logf("%s: ratio %.4f, allocations %.0f -> %.0f", result.Name, result.MedianRatio, result.BeforeAllocs, result.AfterAllocs)
			e.close()
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
