// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"math"
	"testing"
)

// A one-hot input isolates every packed column. Distinct signed rows make a
// swapped activation row or nibble visible even when the sum would alias it.
func TestQ4FourRowBlockColumns(t *testing.T) {
	var x [4][64]float32
	var packed [32]byte
	for code := 0; code < 256; code++ {
		for i := range packed {
			packed[i] = byte(code + 37*i)
		}
		for j := 0; j < 64; j++ {
			for r, value := range []float32{0.5, -2, 4, -8} {
				x[r][j] = value
			}
			a, b, c, d := dotQ4FourBlock64(&x[0], &x[1], &x[2], &x[3], &packed)
			nibble := int8(packed[j/2]<<4) >> 4
			if j%2 != 0 {
				nibble = int8(packed[j/2]) >> 4
			}
			for r, got := range [4]float32{a, b, c, d} {
				want := float32(0) + x[r][j]*float32(nibble)
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("code=%d col=%d row=%d: got %g, want %g", code, j, r, got, want)
				}
				x[r][j] = 0
			}
		}
	}
}
