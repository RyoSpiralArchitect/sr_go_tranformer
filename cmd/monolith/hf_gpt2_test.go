package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func gpt2FixtureCopy(t *testing.T, label string) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range []string{"config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join("testdata/hf-gpt2", label, name))
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestHFGPT2StorageAndLimits(t *testing.T) {
	m, plan, err := loadHFModel(context.Background(), "testdata/hf-gpt2/legacy-f32", 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	var actual int64
	for _, p := range m.Params {
		actual += int64(len(p.Data))
	}
	if actual != plan.Parameters || actual*4 != plan.WeightBytes || plan.RopeBytes != 0 || len(m.Cos)+len(m.Sin) != 0 ||
		m.Position.Rows != 64 || m.Config.Context != 32 || m.Blocks[0].Up != nil {
		t.Fatal("GPT-2 storage accounting", actual, plan)
	}
	if _, err = openHFArchive("testdata/hf-gpt2/legacy-f32", 32, actual*4-1); err == nil {
		t.Fatal("accepted insufficient budget")
	}
	a, err := openHFArchive("testdata/hf-gpt2/legacy-f32", 32, actual*4)
	if err != nil {
		t.Fatal(err)
	}
	a.Close()
	cache := m.NewCache()
	defer cache.Close()
	if _, err = m.PrefillBatch(context.Background(), []*KVCache{cache}, [][]int{make([]int, 32)}); err != nil {
		t.Fatal(err)
	}
	if _, err = m.Step(cache, 0); err == nil || cache.Pos != 32 {
		t.Fatal("absolute position overflow accepted")
	}
}

func TestHFGPT2ConfigBoundaries(t *testing.T) {
	b, err := os.ReadFile("testdata/hf-gpt2/legacy-f32/config.json")
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]any{"tie_word_embeddings": false, "scale_attn_weights": false,
		"scale_attn_by_inverse_layer_idx": true, "reorder_and_upcast_attn": true,
		"add_cross_attention": true, "activation_function": "gelu", "n_ctx": 128,
		"n_positions": -1, "n_inner": -1, "position_embedding_type": "rotary",
		"layer_norm_epsilon": nil, "bos_token_id": []int{1, 2}}
	for key, value := range cases {
		t.Run(key, func(t *testing.T) {
			var cfg map[string]any
			if err := json.Unmarshal(b, &cfg); err != nil {
				t.Fatal(err)
			}
			cfg[key] = value
			dir := t.TempDir()
			hfWriteJSON(t, filepath.Join(dir, "config.json"), cfg)
			if _, _, err := readHFConfig(dir, 0); err == nil {
				t.Fatal("accepted unsupported GPT-2 config")
			}
		})
	}
}

func TestHFGPT2LegacyMaskAndShards(t *testing.T) {
	for _, label := range []string{"legacy-f32", "f16"} {
		t.Run(label, func(t *testing.T) {
			dir := gpt2FixtureCopy(t, label)
			a, err := openHFArchive(dir, 64, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			index := map[string]string{}
			for name := range a.tensors {
				if label == "f16" {
					name = "transformer." + name
				}
				index[name] = "part.safetensors"
			}
			a.Close()
			if err = os.Rename(filepath.Join(dir, "model.safetensors"), filepath.Join(dir, "part.safetensors")); err != nil {
				t.Fatal(err)
			}
			hfWriteJSON(t, filepath.Join(dir, "model.safetensors.index.json"), map[string]any{"weight_map": index})
			m, _, err := loadHFModel(context.Background(), dir, 64, 1)
			if err != nil {
				t.Fatal(err)
			}
			want, _, err := loadHFModel(context.Background(), filepath.Join("testdata/hf-gpt2", label), 64, 1)
			if err != nil {
				t.Fatal(err)
			}
			ca, cb := m.NewCache(), want.NewCache()
			defer ca.Close()
			defer cb.Close()
			x, err := m.Step(ca, 1)
			if err != nil {
				t.Fatal(err)
			}
			y, err := want.Step(cb, 1)
			if err != nil {
				t.Fatal(err)
			}
			hfNear(t, x, y, 0)
		})
	}
	dir := gpt2FixtureCopy(t, "legacy-f32")
	a, err := openHFArchive(dir, 64, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	offset := a.tensors["h.0.attn.bias"].start
	a.Close()
	f, err := os.OpenFile(filepath.Join(dir, "model.safetensors"), os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	var value [4]byte
	binary.LittleEndian.PutUint32(value[:], math.Float32bits(0.5))
	_, err = f.WriteAt(value[:], offset)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = loadHFModel(context.Background(), dir, 64, 1); err == nil {
		t.Fatal("accepted non-causal legacy mask")
	}
}

func TestHFGPT2TokenizerReference(t *testing.T) {
	testHFTokenizerReference(t, "testdata/hf-gpt2-tokenizer", "testdata/hf-gpt2-tokenizer/reference.json")
	tok := testHFTokenizer(t, "testdata/hf-gpt2-tokenizer")
	if !tok.gpt2 || tok.chat {
		t.Fatal("wrong tokenizer profile")
	}
	if _, err := tok.formatChat([]ChatMessage{{Role: "user", Content: "hi"}}); err == nil {
		t.Fatal("fabricated GPT-2 chat template")
	}
	ids, err := tok.encodePrompt(context.Background(), "<|endoftext|>", 64)
	if err != nil || len(ids) != 1 || ids[0] != tok.eos {
		t.Fatal("GPT-2 special token", ids, err)
	}
}

func TestHFGPT2RejectAliasedTensors(t *testing.T) {
	dir := gpt2FixtureCopy(t, "legacy-f32")
	b, err := os.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	n := int(binary.LittleEndian.Uint64(b[:8]))
	var header map[string]any
	if err = json.Unmarshal(b[8:8+n], &header); err != nil {
		t.Fatal(err)
	}
	header["transformer.wte.weight"] = header["wte.weight"]
	hfWriteShard(t, filepath.Join(dir, "model.safetensors"), header, b[8+n:])
	if a, err := openHFArchive(dir, 64, 1<<20); err == nil {
		a.Close()
		t.Fatal("accepted aliased duplicate tensor")
	}
}

func TestHFGPT2RejectTokenizerPipelineChanges(t *testing.T) {
	for _, change := range []string{"prefix", "regex", "null", "normalizer"} {
		t.Run(change, func(t *testing.T) {
			dir := t.TempDir()
			for _, name := range []string{"config.json", "tokenizer_config.json", "tokenizer.json"} {
				b, err := os.ReadFile(filepath.Join("testdata/hf-gpt2-tokenizer", name))
				if err != nil {
					t.Fatal(err)
				}
				if name == "tokenizer.json" {
					var raw map[string]any
					if err = json.Unmarshal(b, &raw); err != nil {
						t.Fatal(err)
					}
					pre := raw["pre_tokenizer"].(map[string]any)
					switch change {
					case "prefix":
						pre["add_prefix_space"] = true
					case "regex":
						pre["use_regex"] = false
					case "null":
						pre["use_regex"] = nil
					case "normalizer":
						raw["normalizer"] = map[string]any{"type": "NFC"}
					}
					b, err = json.Marshal(raw)
					if err != nil {
						t.Fatal(err)
					}
				}
				if err = os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			h, _, err := readHFConfig(dir, 64)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = loadHFTokenizer(dir, h); err == nil {
				t.Fatal("accepted changed GPT-2 pipeline")
			}
		})
	}
}
