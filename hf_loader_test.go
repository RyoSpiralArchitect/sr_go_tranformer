package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func hfTestConfig() hfConfig {
	return hfConfig{Architectures: []string{"LlamaForCausalLM"}, ModelType: "llama", Vocab: 258, Dim: 8, Hidden: 12, Layers: 2, Heads: 2, KVHeads: 1, Context: 64, Theta: 10000, Eps: 1e-5, Act: "silu", Tied: true, BOS: 1, EOS: 2}
}
func hfWriteJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}
func hfWriteShard(t *testing.T, path string, header map[string]any, payload []byte) {
	t.Helper()
	b, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(b)))
	out = append(out, b...)
	out = append(out, payload...)
	if err := os.WriteFile(path, out, 0600); err != nil {
		t.Fatal(err)
	}
}
func hfFixture(t *testing.T, sharded bool) (string, map[string]any, []byte) {
	t.Helper()
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), hfTestConfig())
	_, c, err := readHFConfig(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	header := map[string]any{}
	var payload []byte
	index := map[string]string{}
	for i, s := range hfTensorSpecs(c) {
		n := 1
		for _, d := range s.Shape {
			n *= d
		}
		dtype := []string{"F32", "F16", "BF16"}[i%3]
		start := len(payload)
		for j := 0; j < n; j++ {
			switch dtype {
			case "F32":
				payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(1.5))
			case "F16":
				payload = binary.LittleEndian.AppendUint16(payload, 0x3e00)
			case "BF16":
				payload = binary.LittleEndian.AppendUint16(payload, 0x3fc0)
			}
		}
		header[s.Name] = map[string]any{"dtype": dtype, "shape": s.Shape, "data_offsets": []int{start, len(payload)}}
		index[s.Name] = "part.safetensors"
	}
	name := "model.safetensors"
	if sharded {
		name = "part.safetensors"
		hfWriteJSON(t, filepath.Join(dir, "model.safetensors.index.json"), map[string]any{"weight_map": index})
	}
	hfWriteShard(t, filepath.Join(dir, name), header, payload)
	return dir, header, payload
}
func TestHFArchiveAndConversion(t *testing.T) {
	for _, sharded := range []bool{false, true} {
		dir, _, _ := hfFixture(t, sharded)
		a, err := openHFArchive(dir, 32, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		if a.Plan.Config.Context != 32 || a.Plan.WeightBytes != a.Plan.Parameters*4 || len(a.Plan.DTypes) != 3 {
			t.Fatal(a.Plan)
		}
		for name, tensor := range a.tensors {
			dst := make([]float32, tensor.count)
			if err := a.readTensor(context.Background(), name, dst); err != nil {
				t.Fatal(err)
			}
			for _, v := range dst {
				if v != 1.5 {
					t.Fatalf("%s: %g", name, v)
				}
			}
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		for name, tensor := range a.tensors {
			if err := a.readTensor(ctx, name, make([]float32, tensor.count)); err != context.Canceled {
				t.Fatal(err)
			}
			break
		}
		a.Close()
	}
}
func TestHFRejectMetadata(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(map[string]any)
	}{
		{"architecture", func(c map[string]any) { c["model_type"] = "qwen2" }},
		{"untied", func(c map[string]any) { c["tie_word_embeddings"] = false }},
		{"bias", func(c map[string]any) { c["attention_bias"] = true }},
		{"scaled rope", func(c map[string]any) { c["rope_scaling"] = map[string]any{"type": "linear", "factor": 2} }},
		{"unknown compute", func(c map[string]any) { c["head_dim"] = 4 }},
		{"bad dim", func(c map[string]any) { c["hidden_size"] = math.MaxInt64 }},
		{"bad special", func(c map[string]any) { c["eos_token_id"] = nil }},
		{"missing special", func(c map[string]any) { delete(c, "bos_token_id") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir, _, _ := hfFixture(t, false)
			b, _ := json.Marshal(hfTestConfig())
			var c map[string]any
			_ = json.Unmarshal(b, &c)
			tc.change(c)
			hfWriteJSON(t, filepath.Join(dir, "config.json"), c)
			if a, e := openHFArchive(dir, 0, 1<<20); e == nil {
				a.Close()
				t.Fatal("accepted unsupported config")
			}
		})
	}
	dir, _, _ := hfFixture(t, false)
	if a, e := openHFArchive(dir, 0, 1); e == nil {
		a.Close()
		t.Fatal("ignored budget")
	}
	if a, e := openHFArchive(dir, 65, 1<<20); e == nil {
		a.Close()
		t.Fatal("ignored context bound")
	}
}
func TestHFRejectTensorCorruption(t *testing.T) {
	for _, kind := range []string{"missing", "extra", "shape", "dtype", "overflow", "overlap", "truncated", "trailing", "duplicate", "header bound", "nan", "bad shard path", "wrong shard"} {
		t.Run(kind, func(t *testing.T) {
			dir, header, payload := hfFixture(t, false)
			name := "model.embed_tokens.weight"
			meta := header[name].(map[string]any)
			switch kind {
			case "missing":
				delete(header, name)
			case "extra":
				header["lm_head.weight"] = meta
			case "shape":
				meta["shape"] = []int{math.MaxInt, 8}
			case "dtype":
				meta["dtype"] = "I32"
			case "overflow":
				meta["data_offsets"] = []int64{0, math.MaxInt64}
			case "overlap":
				meta["data_offsets"] = []int{1, 258*8*4 + 1}
			case "truncated":
				payload = payload[:len(payload)-1]
			case "trailing":
				payload = append(payload, 0)
			case "nan":
				binary.LittleEndian.PutUint32(payload, math.Float32bits(float32(math.NaN())))
			case "bad shard path", "wrong shard":
				index := map[string]string{}
				for k := range header {
					index[k] = "model.safetensors"
				}
				if kind == "bad shard path" {
					index[name] = "../model.safetensors"
				} else {
					index[name] = "other.safetensors"
					hfWriteShard(t, filepath.Join(dir, "other.safetensors"), map[string]any{}, nil)
				}
				hfWriteJSON(t, filepath.Join(dir, "model.safetensors.index.json"), map[string]any{"weight_map": index})
			}
			path := filepath.Join(dir, "model.safetensors")
			hfWriteShard(t, path, header, payload)
			if kind == "header bound" {
				b := make([]byte, 8)
				binary.LittleEndian.PutUint64(b, math.MaxUint64)
				_ = os.WriteFile(path, b, 0600)
			}
			if kind == "duplicate" {
				b := []byte(`{"x":{},"x":{}}`)
				out := binary.LittleEndian.AppendUint64(nil, uint64(len(b)))
				out = append(out, b...)
				_ = os.WriteFile(path, out, 0600)
			}
			a, err := openHFArchive(dir, 0, 1<<20)
			if kind == "nan" {
				if err != nil {
					t.Fatal(err)
				}
				defer a.Close()
				err = a.readTensor(context.Background(), name, make([]float32, 258*8))
			}
			if err == nil {
				a.Close()
				t.Fatal("accepted corrupt tensor")
			}
		})
	}
}
func TestHalfFloat(t *testing.T) {
	for _, tc := range []struct {
		bits uint16
		want float32
	}{
		{0, 0}, {0x8000, float32(math.Copysign(0, -1))}, {1, 1.0 / (1 << 24)}, {0x3c00, 1}, {0xc000, -2}, {0x7bff, 65504}, {0x0400, 1.0 / (1 << 14)},
	} {
		if math.Float32bits(halfFloat(tc.bits)) != math.Float32bits(tc.want) {
			t.Fatalf("%x: %g", tc.bits, halfFloat(tc.bits))
		}
	}
	if !math.IsInf(float64(halfFloat(0x7c00)), 1) || !math.IsNaN(float64(halfFloat(0x7e00))) {
		t.Fatal("half special values")
	}
}
func TestUniqueJSON(t *testing.T) {
	for _, s := range []string{`{"a":1,"a":2}`, `{"a":{"b":1,"b":2}}`, `{} {}`, `{"x":`, strings.Repeat("[", 34) + strings.Repeat("]", 34), "{\"\xff\":1}"} {
		var v any
		if uniqueJSON([]byte(s), &v) == nil {
			t.Fatalf("accepted %q", s)
		}
	}
}
