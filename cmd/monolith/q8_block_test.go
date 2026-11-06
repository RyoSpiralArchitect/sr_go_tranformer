// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"testing"
)

// Ordinary Q8 kernel frozen from cfdca31 for exact-bit and paired timing comparisons.
func baselineLinearQ8Into(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	if q8NeedsWideSum(x[:n*in]) {
		linearQ8WideInto(e, y, x, w, n)
		return
	}
	groups, tiles := (in+q8Block-1)/q8Block, (out+31)/32
	e.parallel(((n+3)/4)*tiles, n*in*out, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			r, o0 := (task/tiles)*4, (task%tiles)*32
			for o := o0; o < min(o0+32, out); o++ {
				wr := w.Q8.Values[o*in : (o+1)*in]
				scales := w.Q8.Scales[o*groups : (o+1)*groups]
				if r+4 <= n {
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
				} else {
					for row := r; row < min(r+4, n); row++ {
						xr := x[row*in : (row+1)*in]
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

func TestQ8BlocksEveryCodeAndColumn(t *testing.T) {
	w := newInferenceParam("block-basis", 1, 64, true)
	w.Q8.Scales[0] = .25
	var x [128]float32
	var want [2]float32
	var got [4]float32
	for code := -128; code < 128; code++ {
		for j := range w.Q8.Values {
			w.Q8.Values[j] = int8(code + 37*j)
		}
		for col := 0; col < 64; col++ {
			x[col], x[64+63-col] = 1.25, -.75
			baselineLinearQ8Into(nil, want[:], x[:], w, 2)
			got = [4]float32{-123, -123, -123, -123}
			linearQ8Into(nil, got[1:3], x[:], w, 2)
			if got[0] != -123 || got[3] != -123 {
				t.Fatal("output overrun")
			}
			for row := range want {
				if math.Float32bits(got[row+1]) != math.Float32bits(want[row]) {
					t.Fatalf("code=%d col=%d row=%d", code, col, row)
				}
			}
			var single [1]float32
			linearQ8Into(nil, single[:], x[:64], w, 1)
			if math.Float32bits(single[0]) != math.Float32bits(want[0]) {
				t.Fatal("single block differs", code, col)
			}
			x[col], x[64+63-col] = 0, 0
		}
	}
}

func TestQ8BlocksExactAcrossTailsRowsAndWorkers(t *testing.T) {
	for _, cols := range []int{1, 3, 63, 64, 65, 127, 128, 129, 511, 4096} {
		x, dense, _ := linearFixture(7, cols, 33)
		w := quantizeTestParam(t, dense)
		for i := range x {
			if i%13 == 0 {
				x[i] = math.SmallestNonzeroFloat32
			}
			if i%17 == 0 {
				x[i] = math.Float32frombits(1 << 31)
			}
		}
		for i := range w.Q8.Scales {
			if i%5 == 0 {
				w.Q8.Scales[i] = 0
			}
			if i%11 == 0 {
				w.Q8.Scales[i] = math.SmallestNonzeroFloat32
			}
		}
		for _, rows := range []int{1, 2, 3, 4, 5, 6, 7} {
			want := make([]float32, rows*w.Rows)
			baselineLinearQ8Into(nil, want, x, w, rows)
			for _, workers := range []int{1, 2, 4} {
				e := newKernelExecutor(workers)
				got := make([]float32, len(want)+2)
				got[0], got[len(got)-1] = -777, 777
				linearQ8Into(e, got[1:len(got)-1], x, w, rows)
				e.close()
				if got[0] != -777 || got[len(got)-1] != 777 {
					t.Fatal("sentinel changed")
				}
				for j := range want {
					if math.Float32bits(want[j]) != math.Float32bits(got[j+1]) {
						t.Fatalf("cols=%d rows=%d workers=%d index=%d: %g != %g", cols, rows, workers, j, got[j+1], want[j])
					}
				}
			}
		}
	}
}

func BenchmarkQ8Blocks(b *testing.B) {
	for _, rows := range []int{1, 2, 3, 4} {
		x, dense, _ := linearFixture(rows, 4096, 14336)
		w := quantizeTestParam(b, dense)
		for _, reverse := range []bool{false, true} {
			order := []string{"before", "after"}
			if reverse {
				order[0], order[1] = order[1], order[0]
			}
			for _, mode := range order {
				b.Run(fmt.Sprintf("14336x4096/rows%d/reverse%t/%s", rows, reverse, mode), func(b *testing.B) {
					e := newKernelExecutor(4)
					defer e.close()
					y := make([]float32, rows*w.Rows)
					fn := linearQ8Into
					if mode == "before" {
						fn = baselineLinearQ8Into
					}
					fn(e, y, x, w, rows)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						fn(e, y, x, w, rows)
					}
				})
			}
		}
	}
}
