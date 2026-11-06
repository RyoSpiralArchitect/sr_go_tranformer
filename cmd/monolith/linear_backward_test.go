// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"fmt"
	"math"
	"runtime"
	"testing"
)

// Scalar oracle preserves the original reduction and accumulation order.
func referenceLinearBackward(x []float32, w *Param, dy []float32, n int) []float32 {
	in, out := w.Cols, w.Rows
	dx := make([]float32, n*in)
	for r := 0; r < n; r++ {
		for o := 0; o < out; o++ {
			g := dy[r*out+o]
			for i, v := range w.Data[o*in : (o+1)*in] {
				dx[r*in+i] += g * v
			}
		}
	}
	for o := 0; o < out; o++ {
		for r := 0; r < n; r++ {
			g := dy[r*out+o]
			for i, v := range x[r*in : (r+1)*in] {
				w.Grad[o*in+i] += g * v
			}
		}
	}
	return dx
}

func linearFixture(n, in, out int) ([]float32, *Param, []float32) {
	rng := RNG{51}
	values := func(count int) []float32 {
		v := make([]float32, count)
		for i := range v {
			v[i] = float32(rng.Float64()*2 - 1)
		}
		return v
	}
	return values(n * in), &Param{Rows: out, Cols: in, Data: values(in * out), Grad: values(in * out)}, values(n * out)
}

func TestLinearBackwardExactAccumulation(t *testing.T) {
	old := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(old)
	for _, threads := range []int{1, 4} {
		runtime.GOMAXPROCS(threads)
		for _, shape := range [][3]int{{0, 3, 5}, {1, 3, 5}, {3, 8, 9}, {4, 12, 8}, {5, 8, 7}, {9, 65, 33}, {128, 64, 192}, {128, 256, 704}} {
			n, in, out := shape[0], shape[1], shape[2]
			t.Run(fmt.Sprintf("threads%d/%dx%dx%d", threads, n, in, out), func(t *testing.T) {
				x, w, dy := linearFixture(n, in, out)
				ref := *w
				ref.Grad = append([]float32(nil), w.Grad...)
				for micro := 0; micro < 2; micro++ {
					want := referenceLinearBackward(x, &ref, dy, n)
					got := linearBackward(x, w, dy, n)
					exactLinearBits(t, got, want)
					exactLinearBits(t, w.Grad, ref.Grad)
				}
			})
		}
	}
}

func exactLinearBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("shape changed")
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: %.9g != %.9g", i, got[i], want[i])
		}
	}
}

func BenchmarkLinearBackward(b *testing.B) {
	for _, shape := range [][3]int{{128, 64, 192}, {128, 256, 704}, {3, 12, 258}} {
		n, in, out := shape[0], shape[1], shape[2]
		b.Run(fmt.Sprintf("%dx%dx%d", n, in, out), func(b *testing.B) {
			x, w, dy := linearFixture(n, in, out)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				linearBackward(x, w, dy, n)
			}
		})
	}
}
