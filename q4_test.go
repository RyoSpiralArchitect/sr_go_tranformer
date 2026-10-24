// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestQ4StreamedOddRowsAndPlan(t *testing.T) {
	// A 65-column row crosses both the reader and quantization boundaries.
	_, src, _ := linearFixture(1, 65, 3)
	var data []byte
	for _, v := range src.Data {
		data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
	}
	path := filepath.Join(t.TempDir(), "weights")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	a := &hfArchive{Plan: HFPlan{ConversionBufferBytes: 64}, tensors: map[string]hfTensor{
		"odd": {DType: "F32", count: len(src.Data), file: f},
	}}
	p := newInferenceParamFormat(src.Name, 3, 65, true, "q4")
	if err := a.readParam(context.Background(), "odd", p); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p.Q4, quantizeQ4TestParam(t, src).Q4) {
		t.Fatal("odd streamed rows differ")
	}

	h := hfTestConfig()
	h.Hidden = 65
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), h)
	_, c, err := readHFConfig(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	m := allocateModelStorageFormat(c, nil, "q4")
	var payload, scales, count int64
	for _, p := range m.Params {
		if p.Q4 == nil {
			payload += int64(len(p.Data)) * 4
			continue
		}
		count += int64(p.Rows * p.Cols)
		scales += int64(len(p.Q4.Scales)) * 4
		payload += int64(len(p.Q4.Values) + 4*len(p.Q4.Scales))
	}
	got, q, s, err := hfWeightStorage(c, h, "q4")
	if err != nil || got != payload || q != count || s != scales {
		t.Fatal(got, q, s, payload, count, scales, err)
	}
}

func quantizeQ4TestParam(t testing.TB, src *Param) *Param {
	t.Helper()
	p := newInferenceParamFormat(src.Name, src.Rows, src.Cols, true, "q4")
	groups, stride := (src.Cols+63)/64, (src.Cols+1)/2
	for row := 0; row < src.Rows; row++ {
		for g := 0; g < groups; g++ {
			lo, hi := g*64, min((g+1)*64, src.Cols)
			var maximum float32
			for _, v := range src.Data[row*src.Cols+lo : row*src.Cols+hi] {
				maximum = max(maximum, float32(math.Abs(float64(v))))
			}
			scale := lowBitScale(maximum, 7)
			p.Q4.Scales[row*groups+g] = scale
			for j := lo; j < hi; j++ {
				var v int
				if scale != 0 {
					v = int(math.Round(float64(src.Data[row*src.Cols+j]) / float64(scale)))
					v = max(-7, min(7, v))
				}
				p.Q4.Values[row*stride+j/2] |= byte(v&15) << uint(4*(j%2))
			}
		}
	}
	return p
}

func TestQ4PackingAndFiniteScales(t *testing.T) {
	for b := 0; b < 256; b++ {
		p := newInferenceParamFormat("odd", 2, 3, true, "q4")
		for row := 0; row < 2; row++ {
			p.Q4.Scales[row] = 1
			for j := 0; j < 3; j++ {
				v := (b >> uint(4*(j%2))) & 15
				if v >= 8 {
					v -= 16
				}
				p.Q4.put(3, row, j, int8(v))
			}
			got := make([]float32, 3)
			p.copyRow(got, row)
			for j, v := range got {
				want := (b >> uint(4*(j%2))) & 15
				if want >= 8 {
					want -= 16
				}
				if v != float32(want) {
					t.Fatal(b, j, v, want)
				}
			}
			if p.Q4.Values[row*2] != byte(b) || p.Q4.Values[row*2+1]>>4 != 0 {
				t.Fatal("padding or row overlap", b, p.Q4.Values)
			}
			// Assignment clears stale bits, including replacing a negative value by zero.
			p.Q4.put(3, row, 0, 0)
			if p.Q4.Values[row*2]&15 != 0 {
				t.Fatal("write did not replace nibble")
			}
		}
	}
	for _, maximum := range []float32{0, math.SmallestNonzeroFloat32, 1e-43, 1e-37, .03, 1, math.MaxFloat32} {
		src := &Param{Rows: 1, Cols: 5, Data: []float32{-maximum, -maximum / 3, 0, maximum / 3, maximum}}
		q := quantizeQ4TestParam(t, src)
		got := make([]float32, 5)
		q.copyRow(got, 0)
		scale := q.Q4.Scales[0]
		if !finite(float64(scale)) || scale < 0 || maximum > 0 && scale == 0 {
			t.Fatal(maximum, scale)
		}
		for j, v := range got {
			if !finite(float64(v)) || math.Abs(float64(v)-float64(src.Data[j])) > float64(scale)*.501+float64(maximum)*2e-7 {
				t.Fatal(maximum, j, v, scale)
			}
		}
	}
}

func TestQ4LinearReferenceAndBatchDeterminism(t *testing.T) {
	for _, in := range []int{1, 2, 3, 4, 63, 64, 65, 129, 513} {
		for _, extreme := range []bool{false, true} {
			const rows, out = 7, 259
			x, dense, _ := linearFixture(rows, in, out)
			if extreme {
				for i := range dense.Data {
					dense.Data[i] *= .001
				}
				x[2*in] = math.MaxFloat32 / 4
				x[5*in+in-1] = -1e37
			}
			q := quantizeQ4TestParam(t, dense)
			var baseline []float32
			for _, workers := range []int{1, 2, 4} {
				e := newKernelExecutor(workers)
				got := make([]float32, rows*out)
				linearIntoWithExecutor(e, got, x, q, rows)
				for row := 0; row < rows; row++ {
					one := make([]float32, out)
					linearIntoWithExecutor(e, one, x[row*in:(row+1)*in], q, 1)
					exactWorkerFloats(t, one, got[row*out:(row+1)*out])
					for o, v := range one {
						var want float64
						for j := 0; j < in; j++ {
							bits := (q.Q4.Values[o*((in+1)/2)+j/2] >> uint((j%2)*4)) & 15
							weight := int(bits)
							if weight >= 8 {
								weight -= 16
							}
							want += float64(x[row*in+j]) * float64(weight) * float64(q.Q4.Scales[o*((in+63)/64)+j/64])
						}
						if !finite(float64(v)) || math.Abs(float64(v)-want) > 2e-5*max(1, math.Abs(want)) {
							t.Fatalf("in=%d row=%d out=%d extreme=%v got=%g want=%g", in, row, o, extreme, v, want)
						}
					}
				}
				if baseline != nil {
					exactWorkerFloats(t, got, baseline)
				} else {
					baseline = got
				}
				e.close()
			}
		}
	}
}

func BenchmarkQ4Linear(b *testing.B) {
	for _, shape := range [][2]int{{768, 3072}, {1536, 8960}} {
		for _, rows := range []int{1, 4, 16} {
			x, w, _ := linearFixture(rows, shape[0], shape[1])
			q8, q4 := quantizeTestParam(b, w), quantizeQ4TestParam(b, w)
			for _, format := range []string{"f32", "q8", "q4"} {
				b.Run(fmt.Sprintf("%dx%d/rows%d/%s", shape[1], shape[0], rows, format), func(b *testing.B) {
					p := w
					if format == "q8" {
						p = q8
					}
					if format == "q4" {
						p = q4
					}
					e := newKernelExecutor(4)
					defer e.close()
					y := make([]float32, rows*p.Rows)
					linearIntoWithExecutor(e, y, x, p, rows)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						linearIntoWithExecutor(e, y, x, p, rows)
					}
				})
			}
		}
	}
}
