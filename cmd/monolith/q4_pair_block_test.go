// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"testing"
)

func TestQ4PairBlockEveryCodeAndColumn(t *testing.T) {
	// Independent basis inputs catch swapped/shared rows and every packed
	// byte/column offset. The frozen scalar-row kernel is the bit reference.
	w := newInferenceParamFormat("pair-block", 1, 64, true, "q4")
	w.Q4.Scales[0] = 0.25
	var x [128]float32
	var got [4]float32
	var want [2]float32
	for code := 0; code < 256; code++ {
		for j := range w.Q4.Values {
			w.Q4.Values[j] = byte(code + 37*j)
		}
		for col := 0; col < 64; col++ {
			x[col], x[64+63-col] = 1.25, -0.75
			got = [4]float32{-123, -123, -123, -123}
			legacyLinearQ4Into(nil, want[:], x[:], w, 2)
			linearQ4Into(nil, got[1:3], x[:], w, 2)
			if got[0] != -123 || got[3] != -123 {
				t.Fatal("output overrun", got)
			}
			for row := range want {
				if math.Float32bits(got[row+1]) != math.Float32bits(want[row]) {
					t.Fatalf("code=%d column=%d row=%d: got %g want %g", code, col, row, got[row+1], want[row])
				}
			}
			x[col], x[64+63-col] = 0, 0
		}
	}
}

func BenchmarkQ4PairBlock(b *testing.B) {
	for _, shape := range [][2]int{{1536, 8960}, {3584, 18944}} {
		for _, rows := range []int{1, 2, 3, 4, 6} {
			x, dense, _ := linearFixture(rows, shape[0], shape[1])
			q := quantizeQ4TestParam(b, dense)
			for _, reverse := range []bool{false, true} {
				order := []string{"before", "after"}
				if reverse {
					order[0], order[1] = order[1], order[0]
				}
				for _, mode := range order {
					b.Run(fmt.Sprintf("%dx%d/rows%d/reverse%t/%s", shape[1], shape[0], rows, reverse, mode), func(b *testing.B) {
						e := newKernelExecutor(4)
						defer e.close()
						y := make([]float32, rows*q.Rows)
						fn := linearQ4Into
						if mode == "before" {
							fn = pairBlockBaselineLinearQ4Into
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

// Full ordinary Q4 kernel frozen from PR33 main 9a8db57 (same runtime as PR32).
// Retaining the full callback controls for compiler layout across row paths.
// Benchmarks use ordinary inputs, so the shared wide fallback is never taken;
// the unchanged single-row block helper is shared by both implementations.
func pairBlockBaselineLinearQ4Into(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	if q8NeedsWideSum(x[:n*in]) {
		linearQ4WideInto(e, y, x, w, n)
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
					row := r
					if row+2 <= n {
						x0, x1 := x[row*in:(row+1)*in], x[(row+1)*in:(row+2)*in]
						var s0, s1 float32
						for g, scale := range scales {
							var a0, b0, c0, d0, a1, b1, c1, d1 float32
							j, end := g*q8Block, min((g+1)*q8Block, in)
							for ; j+4 <= end; j += 4 {
								v0, v1 := q4Pairs[wr[j/2]][0], q4Pairs[wr[j/2]][1]
								v2, v3 := q4Pairs[wr[j/2+1]][0], q4Pairs[wr[j/2+1]][1]
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
								v := q4Pairs[wr[j/2]][j%2]
								p0 += x0[j] * v
								p1 += x1[j] * v
							}
							s0 += p0 * scale
							s1 += p1 * scale
						}
						y[row*out+o], y[(row+1)*out+o] = s0, s1
						row += 2
					}
					for ; row < min(r+4, n); row++ {
						xr := x[row*in : (row+1)*in]
						var sum float32
						for g, scale := range scales {
							j, end := g*q8Block, min((g+1)*q8Block, in)
							if end-j == 64 {
								// Safe array views check the complete block once, without copying.
								sum += dotQ4Block64((*[64]float32)(xr[j:end]), (*[32]byte)(wr[j/2:end/2])) * scale
								continue
							}
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
