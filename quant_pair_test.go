// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"testing"
)

func TestQ4PairMatchesLegacy(t *testing.T) {
	for _, in := range []int{1, 2, 3, 4, 63, 64, 65, 127, 128, 129, 513, 1024} {
		for _, rows := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15} {
			for _, out := range []int{1, 31, 32, 33, 65} {
				x, dense, _ := linearFixture(rows, in, out)
				q, legacy := quantizeQ4TestParam(t, dense), legacyLinearQ4Into
				for _, workers := range []int{0, 1, 4} {
					var e *kernelExecutor
					if workers != 0 {
						e = newKernelExecutor(workers)
					}
					got, want := make([]float32, rows*out+2), make([]float32, rows*out+2)
					for i := range got {
						got[i], want[i] = -123, -123
					}
					legacy(e, want[1:len(want)-1], x, q, rows)
					linearIntoWithExecutor(e, got[1:len(got)-1], x, q, rows)
					for i := range got {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("q4 rows=%d in=%d out=%d workers=%d index=%d: %g != %g", rows, in, out, workers, i, got[i], want[i])
						}
					}
					if e != nil {
						e.close()
					}
				}
			}
		}
	}
}

func TestQ4PairExtremeInputsMatchLegacy(t *testing.T) {
	for _, in := range []int{3, 64, 65, 129} {
		for _, rows := range []int{2, 3, 6, 7} {
			x, dense, _ := linearFixture(rows, in, 65)
			for i := range dense.Data {
				dense.Data[i] *= .0001
			}
			q, legacy := quantizeQ4TestParam(t, dense), legacyLinearQ4Into
			for _, extreme := range []float32{0, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, math.MaxFloat32 / 4, -1e37} {
				x[(rows-1)*in] = extreme
				x[in-1] = float32(math.Copysign(0, -1))
				for _, workers := range []int{1, 4} {
					e := newKernelExecutor(workers)
					got, want := make([]float32, rows*q.Rows), make([]float32, rows*q.Rows)
					legacy(e, want, x, q, rows)
					linearIntoWithExecutor(e, got, x, q, rows)
					for i := range got {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("q4 rows=%d in=%d extreme=%g workers=%d index=%d: %g != %g", rows, in, extreme, workers, i, got[i], want[i])
						}
					}
					e.close()
				}
			}
		}
	}
}

func BenchmarkQ4Pair(b *testing.B) {
	for _, shape := range [][2]int{{1536, 8960}, {3584, 18944}} {
		for _, rows := range []int{1, 2, 3, 4, 6} {
			x, dense, _ := linearFixture(rows, shape[0], shape[1])
			q, legacy := quantizeQ4TestParam(b, dense), legacyLinearQ4Into
			for _, reverse := range []bool{false, true} {
				order := []string{"legacy", "pair"}
				if reverse {
					order[0], order[1] = order[1], order[0]
				}
				for _, mode := range order {
					b.Run(fmt.Sprintf("%dx%d/rows%d/q4/reverse%t/%s", shape[1], shape[0], rows, reverse, mode), func(b *testing.B) {
						e := newKernelExecutor(4)
						defer e.close()
						y := make([]float32, rows*q.Rows)
						fn := linearIntoWithExecutor
						if mode == "legacy" {
							fn = legacy
						}
						fn(e, y, x, q, rows)
						b.ReportAllocs()
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							fn(e, y, x, q, rows)
						}
					})
				}
			}
		}
	}
}

// Frozen kernels from main at 167adfb4dd609d0dc6f9c2ce9f29b9d49023ca91.
// Retained only in tests for exact arithmetic and same-binary timing comparisons.
// Integer partial sums precede the block scale. A rare, very large activation
// can overflow them even when the scaled result is finite. The factor of two
// leaves rounding headroom; ordinary activations only pay one input scan.
func legacyQ8NeedsWideSum(x []float32) bool {
	const limit = float32(math.MaxFloat32 / (2 * q8Block * 127))
	for _, v := range x {
		if v > limit || v < -limit {
			return true
		}
	}
	return false
}

// Packed pairs are decoded once per four input columns and reused by up to
// four rows. Both paths use the same reduction order across batching/workers.
func legacyLinearQ4Into(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	if legacyQ8NeedsWideSum(x[:n*in]) {
		legacyLinearQ4WideInto(e, y, x, w, n)
		return
	}

	groups, tiles := (in+q8Block-1)/q8Block, (out+31)/32
	e.parallel(((n+3)/4)*tiles, n*in*out, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			r, o0 := (task/tiles)*4, (task%tiles)*32
			for o := o0; o < min(o0+32, out); o++ {
				wr := w.Q4.Values[o*((in+1)/2) : (o+1)*((in+1)/2)]
				scales := w.Q4.Scales[o*groups : (o+1)*groups]
				if r+4 <= n {
					x0, x1 := x[r*in:(r+1)*in], x[(r+1)*in:(r+2)*in]
					x2, x3 := x[(r+2)*in:(r+3)*in], x[(r+3)*in:(r+4)*in]
					var s0, s1, s2, s3 float32
					for g, scale := range scales {
						var a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 float32
						j, end := g*q8Block, min((g+1)*q8Block, in)
						for ; j+4 <= end; j += 4 {
							v0, v1, v2, v3 := q4Pairs[wr[j/2]][0], q4Pairs[wr[j/2]][1], q4Pairs[wr[j/2+1]][0], q4Pairs[wr[j/2+1]][1]
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
							v := q4Pairs[wr[(j)/2]][(j)%2]
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
								v0, v1 := q4Pairs[wr[j/2]][0], q4Pairs[wr[j/2]][1]
								v2, v3 := q4Pairs[wr[j/2+1]][0], q4Pairs[wr[j/2+1]][1]
								a += xr[j] * v0
								b += xr[j+1] * v1
								c += xr[j+2] * v2
								d += xr[j+3] * v3
							}
							part := (a + b) + (c + d)
							for ; j < end; j++ {
								part += xr[j] * q4Pairs[wr[(j)/2]][(j)%2]
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

func legacyLinearQ4WideInto(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	groups := (in + q8Block - 1) / q8Block
	for row := 0; row < n; row++ {
		xr, yr := x[row*in:(row+1)*in], y[row*out:(row+1)*out]
		if !legacyQ8NeedsWideSum(xr) {
			// Select per row, so a neighbouring extreme input never changes
			// the reduction order or result of an ordinary row in the batch.
			legacyLinearQ4Into(e, yr, xr, w, 1)
			continue
		}
		e.parallel(out, in*out, func(lo, hi int) {
			for o := lo; o < hi; o++ {
				wr := w.Q4.Values[o*((in+1)/2) : (o+1)*((in+1)/2)]
				var sum float64
				for g := 0; g < groups; g++ {
					var part float64
					for j := g * q8Block; j < min((g+1)*q8Block, in); j++ {
						part += float64(xr[j]) * float64(q4Pairs[wr[j/2]][j%2])
					}
					sum += part * float64(w.Q4.Scales[o*groups+g])
				}
				yr[o] = float32(sum)
			}
		})
	}
}
