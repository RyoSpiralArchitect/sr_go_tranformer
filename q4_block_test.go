// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"math"
	"testing"
)

func TestQ4FullBlockEveryCodeAndColumn(t *testing.T) {
	// Each packed offset sees all 256 byte values. Basis inputs isolate every
	// activation/weight index, including the negative nibble encodings.
	w := newInferenceParamFormat("block", 1, 64, true, "q4")
	w.Q4.Scales[0] = 0.25
	var x [64]float32
	var got [3]float32
	for code := 0; code < 256; code++ {
		for j := range w.Q4.Values {
			w.Q4.Values[j] = byte(code + 37*j)
		}
		for col := range x {
			x[col] = 1.25
			got[0], got[1], got[2] = -123, -123, -123
			linearQ4Into(nil, got[1:2], x[:], w, 1)
			packed := w.Q4.Values[col/2]
			value := int8(packed<<4) >> 4
			if col%2 != 0 {
				value = int8(packed) >> 4
			}
			want := x[col] * float32(value) * w.Q4.Scales[0]
			if math.Float32bits(got[1]) != math.Float32bits(want) || got[0] != -123 || got[2] != -123 {
				t.Fatalf("code=%d column=%d: got %v, want %g", code, col, got, want)
			}
			x[col] = 0
		}
	}
}

func TestQ4FullBlockAndEveryTailMatchLegacy(t *testing.T) {
	// Include no full block, multiple full blocks and every partial-block
	// length. Mixed magnitudes exercise rounding/cancellation in all four
	// accumulators. Other Q4 tests cover the per-row wide fallback and workers.
	for width := 1; width <= 193; width++ {
		for _, rows := range []int{1, 3, 4, 5, 7} {
			x, dense, _ := linearFixture(rows, width, 5)
			for i := range x {
				x[i] = float32(math.Ldexp(float64(x[i]), (i*37)%181-90))
			}
			x[0] = float32(math.Copysign(0, -1))
			q := quantizeQ4TestParam(t, dense)
			got, want := make([]float32, rows*q.Rows), make([]float32, rows*q.Rows)
			legacyLinearQ4Into(nil, want, x, q, rows)
			linearQ4Into(nil, got, x, q, rows)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("width=%d rows=%d index=%d: %g != %g", width, rows, i, got[i], want[i])
				}
			}
		}
	}
}
