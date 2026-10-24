// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func quantizeTestParam(t testing.TB, src *Param) *Param {
	t.Helper()
	p := newInferenceParam(src.Name, src.Rows, src.Cols, true)
	groups := (p.Cols + q8Block - 1) / q8Block
	for row := 0; row < p.Rows; row++ {
		for g := 0; g < groups; g++ {
			lo, hi := row*p.Cols+g*q8Block, row*p.Cols+min((g+1)*q8Block, p.Cols)
			var amax float32
			for _, v := range src.Data[lo:hi] {
				amax = max(amax, float32(math.Abs(float64(v))))
			}
			scale := q8Scale(amax)
			p.Q8.Scales[row*groups+g] = scale
			for j := lo; j < hi; j++ {
				p.Q8.Values[j] = q8Value(src.Data[j], scale)
			}
		}
	}
	return p
}

func TestQ8ScaleFiniteExtremes(t *testing.T) {
	for _, maximum := range []float32{0, math.SmallestNonzeroFloat32, 1e-43, 1e-37, .03, 1, math.MaxFloat32} {
		scale := q8Scale(maximum)
		if !finite(float64(scale)) || scale < 0 || maximum > 0 && scale == 0 {
			t.Fatal("invalid scale", maximum, scale)
		}
		for _, x := range []float32{-maximum, -maximum / 3, 0, maximum / 3, maximum} {
			q := q8Value(x, scale)
			got := float32(q) * scale
			if !finite(float64(got)) || q < -127 || math.Abs(float64(got)-float64(x)) > float64(scale)*.501+float64(maximum)*2e-7 {
				t.Fatal("quantization error outside one half-step plus FP32 rounding", x, got, scale)
			}
		}
	}
}

func TestQ8LinearReferenceAndBatchDeterminism(t *testing.T) {
	for _, in := range []int{1, 2, 63, 64, 65, 129, 513} {
		x, dense, _ := linearFixture(7, in, 259)
		q := quantizeTestParam(t, dense)
		reconstructed := make([]float32, 259*in)
		for row := 0; row < 259; row++ {
			q.copyRow(reconstructed[row*in:(row+1)*in], row)
		}
		want := make([]float32, 7*259)
		for row := 0; row < 7; row++ {
			for out := 0; out < 259; out++ {
				var sum float64
				for j := 0; j < in; j++ {
					sum += float64(x[row*in+j]) * float64(reconstructed[out*in+j])
				}
				want[row*259+out] = float32(sum)
			}
		}
		var baseline []float32
		for _, workers := range []int{1, 2, 4} {
			e := newKernelExecutor(workers)
			got := make([]float32, len(want))
			linearIntoWithExecutor(e, got, x, q, 7)
			for i, v := range got {
				if !finite(float64(v)) || math.Abs(float64(v-want[i])) > 2e-5*max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("shape in=%d workers=%d index=%d: %g != %g", in, workers, i, v, want[i])
				}
			}
			if baseline != nil {
				exactWorkerFloats(t, got, baseline)
			} else {
				baseline = got
			}
			for row := 0; row < 7; row++ {
				one := make([]float32, 259)
				linearIntoWithExecutor(e, one, x[row*in:(row+1)*in], q, 1)
				exactWorkerFloats(t, one, got[row*259:(row+1)*259])
			}
			e.close()
		}
	}
}

func TestQ8ExtremeActivationsRemainFiniteAndBatchInvariant(t *testing.T) {
	x, dense, _ := linearFixture(7, 65, 259)
	for i := range dense.Data {
		dense.Data[i] = float32(i%13-6) * .0001
	}
	x[2*65], x[5*65+64] = math.MaxFloat32/4, -1e37
	q := quantizeTestParam(t, dense)
	var baseline []float32
	for _, workers := range []int{1, 2, 4} {
		e := newKernelExecutor(workers)
		got := make([]float32, 7*259)
		linearIntoWithExecutor(e, got, x, q, 7)
		for row := 0; row < 7; row++ {
			one := make([]float32, 259)
			linearIntoWithExecutor(e, one, x[row*65:(row+1)*65], q, 1)
			exactWorkerFloats(t, one, got[row*259:(row+1)*259])
			for o, v := range one {
				var ref float64
				for j := 0; j < 65; j++ {
					ref += float64(x[row*65+j]) * float64(q.Q8.Values[o*65+j]) * float64(q.Q8.Scales[o*2+j/64])
				}
				if !finite(float64(v)) || math.Abs(float64(v)-ref) > 2e-5*max(1, math.Abs(ref)) {
					t.Fatal("avoidable quantized overflow/error", row, o, v, ref)
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

func TestQ8ExtremeFiniteHFModel(t *testing.T) {
	h := hfTestConfig()
	h.Vocab, h.Dim, h.Hidden, h.Layers, h.Heads, h.KVHeads, h.Tied = 258, 2, 2, 1, 1, 1, false
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), h)
	_, c, err := readHFConfig(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	header := map[string]any{}
	var data []byte
	for _, s := range hfProfileSpecs(c, h) {
		count := 1
		for _, d := range s.Shape {
			count *= d
		}
		start := len(data)
		for i := 0; i < count; i++ {
			var v float32
			switch s.Name {
			case "model.embed_tokens.weight":
				if i%2 == 0 {
					v = 1
				}
			case "model.norm.weight":
				v = 1e37
			case "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight":
				v = 1
			case "lm_head.weight":
				if i%2 == 0 {
					v = .01
				}
			}
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		header[s.Name] = map[string]any{"dtype": "F32", "shape": s.Shape, "data_offsets": []int{start, len(data)}}
	}
	hfWriteShard(t, filepath.Join(dir, "model.safetensors"), header, data)
	var reference []float32
	for _, format := range []string{"f32", "q8"} {
		m, _, err := loadHFModelFormat(context.Background(), dir, 32, 1, format)
		if err != nil {
			t.Fatal(err)
		}
		r, err := inferHFIDs(context.Background(), m, []int{1, 3, 4, 5}, 2, 4, true)
		if err != nil {
			t.Fatal("finite model rejected", format, err)
		}
		row := r.Logits[0]
		if reference == nil {
			reference = row
		} else {
			for i, v := range row {
				if !finite(float64(v)) || math.Abs(float64(v-reference[i])) > math.Abs(float64(reference[i]))*1e-5 {
					t.Fatal("extreme HF result", v, reference[i])
				}
			}
		}
	}
}

func TestQ8LoaderStorageAndReconstructedModel(t *testing.T) {
	for _, profile := range []string{"hf-llama/f32", "hf-llama/f16", "hf-llama/bf16", "hf-llama/untied-f32",
		"hf-qwen2/f32", "hf-qwen2/f16", "hf-qwen2/untied-bf16", "hf-gpt2/legacy-f32", "hf-gpt2/f16", "hf-gpt2/bf16"} {
		t.Run(profile, func(t *testing.T) {
			dir := filepath.Join("testdata", profile)
			m, plan, err := loadHFModelFormat(context.Background(), dir, 64, 1, "q8")
			if err != nil {
				t.Fatal(err)
			}
			dense, _, err := loadHFModel(context.Background(), dir, 64, 1)
			if err != nil {
				t.Fatal(err)
			}
			var payload, scales, count int64
			for i, p := range m.Params {
				ref := dense.Params[i]
				if p.Rows != ref.Rows || p.Cols != ref.Cols || p.Name != ref.Name {
					t.Fatal("parameter layout changed")
				}
				if len(p.Grad)+len(p.M)+len(p.V) != 0 {
					t.Fatal("optimizer allocated")
				}
				if p.Q8 == nil {
					payload += int64(len(p.Data)) * 4
					exactWorkerFloats(t, p.Data, ref.Data)
					continue
				}
				if len(p.Data) != 0 {
					t.Fatal("full FP32 copy retained")
				}
				expected := quantizeTestParam(t, ref)
				if len(p.Q8.Values) != len(expected.Q8.Values) || len(p.Q8.Scales) != len(expected.Q8.Scales) {
					t.Fatal("quantized shape")
				}
				for j, v := range p.Q8.Values {
					if v != expected.Q8.Values[j] {
						t.Fatal("streamed quantization/transpose differs", p.Name, j)
					}
				}
				exactWorkerFloats(t, p.Q8.Scales, expected.Q8.Scales)
				scales += int64(len(p.Q8.Scales)) * 4
				count += int64(len(p.Q8.Values))
				payload += int64(len(p.Q8.Values)) + int64(len(p.Q8.Scales))*4
				for row := 0; row < p.Rows; row++ {
					p.copyRow(ref.Data[row*p.Cols:(row+1)*p.Cols], row)
				}
			}
			if payload != plan.WeightBytes || scales != plan.ScaleBytes || count != plan.QuantizedParameters ||
				plan.FP32WeightBytes != plan.Parameters*4 || payload >= plan.FP32WeightBytes {
				t.Fatal("storage accounting", payload, plan)
			}
			if a, err := openHFArchiveFormat(dir, 64, payload-1, "q8"); err == nil {
				a.Close()
				t.Fatal("budget ignored")
			}
			a, err := openHFArchiveFormat(dir, 64, payload, "q8")
			if err != nil {
				t.Fatal("exact budget rejected", err)
			}
			a.Close()
			var w decodeWorkspace
			exec := newKernelExecutor(4)
			defer exec.close()
			ca, cb := m.NewCache(), dense.NewCache()
			defer ca.Close()
			defer cb.Close()
			for _, n := range []int{33, 1, 7, 1} {
				chunks := [][]int{prefillTokens(n, 7)}
				got, err := m.prefillBorrowed(context.Background(), []*KVCache{ca}, chunks, exec, &w)
				if err != nil {
					t.Fatal(err)
				}
				want, err := dense.PrefillBatch(context.Background(), []*KVCache{cb}, chunks)
				if err != nil {
					t.Fatal(err)
				}
				hfNear(t, got, want, 3e-5)
				assertWorkspaceReferencesCleared(t, &w)
			}
			if _, err := m.Forward([]int{1}, 1, 1); err == nil {
				t.Fatal("quantized training accepted")
			}
			if _, err := m.AdamW(defaultTrainSpec(), 1); err == nil {
				t.Fatal("quantized optimizer accepted")
			}
			if err := SaveCheckpoint(filepath.Join(t.TempDir(), "q.mglm"), m, Tokenizer{}, nil); err == nil {
				t.Fatal("quantized native save accepted")
			}
		})
	}
}

func TestQ8LoadCancellationAndInvalidWeights(t *testing.T) {
	for _, sharded := range []bool{false, true} {
		dir, _, _ := hfFixture(t, sharded)
		if _, _, err := loadHFModelFormat(context.Background(), dir, 32, 1, "q8"); err != nil {
			t.Fatal(err)
		}
	}
	dir, header, payload := hfFixture(t, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := loadHFModelFormat(ctx, dir, 32, 1, "q8"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(payload, math.Float32bits(float32(math.NaN())))
	hfWriteShard(t, filepath.Join(dir, "model.safetensors"), header, payload)
	if _, _, err := loadHFModelFormat(context.Background(), dir, 32, 1, "q8"); err == nil {
		t.Fatal("nonfinite source accepted")
	}
	// Public/private inference guards also keep canceled q8 batches uncommitted.
	m, _, err := loadHFModelFormat(context.Background(), "testdata/hf-qwen2/f32", 64, 1, "q8")
	if err != nil {
		t.Fatal(err)
	}
	var w decodeWorkspace
	for _, at := range []int{1, 2, 3, 4, 5} {
		cache := m.NewCache()
		ctx := &prefillCancelContext{Context: context.Background(), cancelAt: at}
		if _, err := m.prefillBorrowed(ctx, []*KVCache{cache}, [][]int{{1, 2, 3}}, nil, &w); !errors.Is(err, context.Canceled) {
			t.Fatal(at, err)
		}
		if cache.Pos != 0 {
			t.Fatal("cancellation committed positions")
		}
		assertWorkspaceReferencesCleared(t, &w)
		cache.Close()
	}
}

func TestQ8LargeModelPlanBeforeAllocation(t *testing.T) {
	h := hfTestConfig()
	h.ModelType, h.Architectures = "qwen2", []string{"Qwen2ForCausalLM"}
	h.Vocab, h.Dim, h.Hidden, h.Layers, h.Heads, h.KVHeads = 151936, 1536, 8960, 28, 12, 2
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), h)
	_, c, err := readHFConfig(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	if hfParameterCount(c, h) <= 1_000_000_000 || c.Validate() == nil {
		t.Fatal("wrong inference/training limits")
	}
	bytes, _, _, err := hfWeightStorage(c, h, "q8")
	if err != nil || bytes >= 2<<30 || bytes < 1<<30 {
		t.Fatal(bytes, err)
	}
	if _, err := openHFArchiveFormat(dir, 32, bytes-1, "q8"); err == nil || errors.Is(err, os.ErrNotExist) {
		t.Fatal("budget was not checked before opening weights", err)
	}
	if _, err := openHFArchiveFormat(dir, 32, bytes, "q8"); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("unexpected metadata-only acceptance", err)
	}
}

func BenchmarkQ8Linear(b *testing.B) {
	for _, shape := range [][2]int{{768, 3072}, {1536, 8960}} {
		for _, rows := range []int{1, 4, 16} {
			x, w, _ := linearFixture(rows, shape[0], shape[1])
			q := quantizeTestParam(b, w)
			for _, format := range []string{"f32", "q8"} {
				b.Run(fmt.Sprintf("%dx%d/rows%d/%s", shape[1], shape[0], rows, format), func(b *testing.B) {
					p := w
					if format == "q8" {
						p = q
					}
					exec := newKernelExecutor(4)
					defer exec.close()
					y := make([]float32, rows*p.Rows)
					linearIntoWithExecutor(exec, y, x, p, rows)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						linearIntoWithExecutor(exec, y, x, p, rows)
					}
				})
			}
		}
	}
}
