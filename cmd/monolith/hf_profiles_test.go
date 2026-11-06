package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHFProfileLimitsAndBudget(t *testing.T) {
	h := hfTestConfig()
	h.Vocab, h.Dim, h.Hidden, h.Layers, h.Heads, h.KVHeads = 151936, 896, 4864, 24, 14, 2
	h.ModelType, h.Architectures = "qwen2", []string{"Qwen2ForCausalLM"}
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), h)
	_, c, err := readHFConfig(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	if c.Validate() == nil {
		t.Fatal("expanded native training limits")
	}
	if n := hfParameterCount(c, h); n != 494032768 {
		t.Fatalf("Qwen parameter count %d", n)
	}
	if _, err := openHFArchive(dir, 32, 1024<<20); err == nil || !strings.Contains(err.Error(), "-weight-mib") {
		t.Fatal("must reject budget before touching missing weights", err)
	}
	for _, field := range []string{"sliding", "layer type", "oversized"} {
		bad := h
		switch field {
		case "sliding":
			bad.Sliding = true
		case "layer type":
			bad.LayerTypes = []string{"sliding_attention"}
		case "oversized":
			bad.Layers = 100
			bad.Dim, bad.Hidden, bad.Heads, bad.KVHeads = 4096, 11008, 32, 8
		}
		hfWriteJSON(t, filepath.Join(dir, "config.json"), bad)
		if _, _, err := readHFConfig(dir, 32); err == nil {
			t.Fatal("accepted", field)
		}
	}
}

func TestHFExtraWeightsCountAndTrainingRejection(t *testing.T) {
	for _, name := range []string{"hf-qwen2/f32", "hf-qwen2/untied-bf16", "hf-llama/untied-f32", "hf-mistral/f32"} {
		dir := filepath.Join("testdata", name)
		m, plan, err := loadHFModel(context.Background(), dir, 64, 1)
		if err != nil {
			t.Fatal(err)
		}
		var actual int64
		for _, p := range m.Params {
			actual += int64(len(p.Data))
		}
		if actual != plan.Parameters || actual*4 != plan.WeightBytes || actual != m.parameterCount() {
			t.Fatal("unaccounted weights", actual, plan)
		}
		if _, err := openHFArchive(dir, 64, plan.WeightBytes-1); err == nil {
			t.Fatal("extra weights escaped budget")
		}
		if _, err := m.Forward([]int{1}, 1, 1); err == nil {
			t.Fatal("accepted HF training forward")
		}
		if _, err := m.AdamW(defaultTrainSpec(), 1); err == nil {
			t.Fatal("accepted HF optimizer")
		}
		if _, err := newEvaluationWorkspace(m, 1, 1); err == nil {
			t.Fatal("accepted native evaluation framing")
		}
	}
	// A tied archive cannot satisfy an untied config, even with matching shapes.
	dir := t.TempDir()
	for _, name := range []string{"config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join("testdata", "hf-qwen2", "f32", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	var cfg hfConfig
	if err := readBoundedJSON(filepath.Join(dir, "config.json"), 1<<20, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.Tied = false
	hfWriteJSON(t, filepath.Join(dir, "config.json"), cfg)
	if _, err := openHFArchive(dir, 64, 1<<20); err == nil {
		t.Fatal("accepted missing untied output head")
	}
}

func TestHFQwenNormalizationAndAddedTokens(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-qwen-tokenizer")
	if _, err := tok.encodePrompt(context.Background(), "a"+strings.Repeat("\u0301", 35), 512); err == nil {
		t.Fatal("silently inserted stream-safe CGJ")
	}
	ids, err := tok.encodePrompt(context.Background(), "<tool_call>e\u0301</tool_call>", 512)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Decode(ids); got != "<tool_call>é</tool_call>" {
		t.Fatal("normalization or non-special added token lost", got)
	}
	if got := tok.Decode([]int{tok.Vocab() - 1}); got != "" {
		t.Fatal("fabricated a padding token", got)
	}
}
