// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLlama3ConfigBoundaries(t *testing.T) {
	source, err := os.ReadFile("testdata/hf-llama3/f32/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		change func(map[string]any, map[string]any)
	}{
		{"unknown algorithm", func(c, r map[string]any) { r["rope_type"] = "yarn" }},
		{"unknown field", func(c, r map[string]any) { r["attention_factor"] = 2 }},
		{"case alias override", func(c, r map[string]any) { r["FACTOR"] = 1 }},
		{"case alias replacement", func(c, r map[string]any) { r["FACTOR"] = r["factor"]; delete(r, "factor") }},
		{"null threshold", func(c, r map[string]any) { r["high_freq_factor"] = nil }},
		{"missing factor", func(c, r map[string]any) { delete(r, "factor") }},
		{"zero factor", func(c, r map[string]any) { r["factor"] = 0 }},
		{"huge factor", func(c, r map[string]any) { r["factor"] = 1e39 }},
		{"zero low", func(c, r map[string]any) { r["low_freq_factor"] = 0 }},
		{"inverted thresholds", func(c, r map[string]any) { r["high_freq_factor"] = 0.5 }},
		{"rounded equal thresholds", func(c, r map[string]any) { r["high_freq_factor"] = 1.000000001 }},
		{"negative original context", func(c, r map[string]any) { r["original_max_position_embeddings"] = -1 }},
		{"original exceeds model", func(c, r map[string]any) { r["original_max_position_embeddings"] = 129 }},
		{"fractional original", func(c, r map[string]any) { r["original_max_position_embeddings"] = 31.5 }},
		{"null head", func(c, r map[string]any) { c["head_dim"] = nil }},
		{"wrong head", func(c, r map[string]any) { c["head_dim"] = 8 }},
		{"qwen scaling", func(c, r map[string]any) {
			c["model_type"] = "qwen2"
			c["architectures"] = []string{"Qwen2ForCausalLM"}
			delete(c, "head_dim")
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var c map[string]any
			if err := json.Unmarshal(source, &c); err != nil {
				t.Fatal(err)
			}
			tc.change(c, c["rope_scaling"].(map[string]any))
			dir := t.TempDir()
			hfWriteJSON(t, filepath.Join(dir, "config.json"), c)
			if _, _, err := readHFConfig(dir, 64); err == nil {
				t.Fatal("accepted unsupported scaling/head semantics")
			}
		})
	}
	h, c, err := readHFConfig("testdata/hf-llama3/f32", 64)
	if err != nil || c.Context != 64 || h.RoPE == nil || h.RoPE.Original != 32 {
		t.Fatal(h, c, err)
	}
	// A runtime context cap does not redefine the pretrained scaling parameters.
	h, c, err = readHFConfig("testdata/hf-llama3/f32", 16)
	if err != nil || c.Context != 16 || h.RoPE.Original != 32 {
		t.Fatal(h, c, err)
	}
}

func TestQualityExternalTokenBoundaries(t *testing.T) {
	good := qualityTokenInputs{Format: "monolith-quality-tokens-v1", Source: "model@revision", Corpus: "corpus", TokenizerSHA: strings.Repeat("0", 64), Version: "0.22.2", Documents: [][]int{{1, 2, 3, 4, 5}}, Prompts: [][]int{{1, 2}}}
	if err := good.validate("model@revision", "corpus", 1, 1, 1, 258); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*qualityTokenInputs){
		func(q *qualityTokenInputs) { q.Source = "other" },
		func(q *qualityTokenInputs) { q.Corpus = "other" },
		func(q *qualityTokenInputs) { q.Documents = nil },
		func(q *qualityTokenInputs) { q.Prompts = nil },
		func(q *qualityTokenInputs) { q.Documents = [][]int{{1, 2, 3, 4}} },
		func(q *qualityTokenInputs) { q.Documents = [][]int{{1, 2, 3, 4, 258}} },
		func(q *qualityTokenInputs) { q.Prompts = [][]int{{-1}} },
		func(q *qualityTokenInputs) { q.Prompts = [][]int{make([]int, 113)} },
	} {
		got := good
		mutate(&got)
		if got.validate("model@revision", "corpus", 1, 1, 1, 258) == nil {
			t.Fatal("invalid external token inputs accepted")
		}
	}
}

// Exercises both writing and reading the reference, including raw-ID
// continuations after the reader reaches EOF. No pretrained model is used.
func TestQualityExternalTokensEndToEnd(t *testing.T) {
	corpus, err := os.ReadFile("validation/quality/corpus.json")
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(corpus)
	ids := qualityTokenInputs{Format: "monolith-quality-tokens-v1", Source: "synthetic@fixture", Corpus: hex.EncodeToString(hash[:]), TokenizerSHA: strings.Repeat("0", 64), Version: "synthetic-test", Prompts: [][]int{{1, 41}, {1, 52}}}
	for i := 0; i < 6; i++ {
		ids.Documents = append(ids.Documents, []int{1, 11, 31, 7, 9})
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "ids.json")
	hfWriteJSON(t, input, ids)
	t.Setenv("MONOLITH_QUALITY_MODEL", "testdata/hf-llama3/f32")
	t.Setenv("MONOLITH_QUALITY_SOURCE", ids.Source)
	t.Setenv("MONOLITH_QUALITY_TOKEN_INPUTS", input)
	t.Setenv("MONOLITH_QUALITY_STEPS", "1")
	t.Setenv("MONOLITH_QUALITY_DOCUMENTS", "6")
	t.Setenv("MONOLITH_QUALITY_WEIGHT_MIB", "1")
	t.Setenv("MONOLITH_QUALITY_REFERENCE_FORMAT", "f32")
	t.Setenv("MONOLITH_QUALITY_REFERENCE", filepath.Join(dir, "reference.f32"))
	for i, format := range []string{"f32", "q8", "q4-mixed"} {
		t.Run(format, func(t *testing.T) {
			t.Setenv("MONOLITH_QUALITY_FORMAT", format)
			write := "0"
			if i == 0 {
				write = "1"
			}
			t.Setenv("MONOLITH_QUALITY_WRITE_REFERENCE", write)
			t.Setenv("MONOLITH_QUALITY_OUTPUT", filepath.Join(dir, format+".json"))
			TestHFQualityProbe(t)
		})
	}
}

func TestLlama3FrequencyOracle(t *testing.T) {
	var cases []struct {
		Theta       float64      `json:"theta"`
		Head        int          `json:"head_dim"`
		Scaling     hfLlama3RoPE `json:"scaling"`
		Frequencies []float32    `json:"inverse_frequencies"`
	}
	b, err := os.ReadFile("testdata/hf-llama3/frequencies.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		config := hfTestConfig()
		config.Dim, config.Heads, config.KVHeads, config.Hidden = c.Head*2, 2, 1, c.Head*3
		config.HeadDim, config.Theta, config.Context, config.RoPE = c.Head, c.Theta, max(128, c.Scaling.Original), &c.Scaling
		dir := t.TempDir()
		hfWriteJSON(t, filepath.Join(dir, "config.json"), config)
		parsed, _, err := readHFConfig(dir, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(c.Frequencies) != c.Head/2 {
			t.Fatal("oracle frequency count")
		}
		for i, want := range c.Frequencies {
			got := hfInverseFrequency(parsed.Theta, c.Head, i, parsed.RoPE)
			if !finite(float64(got)) || math.Abs(float64(got-want)) > 2e-7*math.Max(1e-12, math.Abs(float64(want))) {
				t.Fatalf("theta=%g index=%d got=%.9g want=%.9g", c.Theta, i, got, want)
			}
		}
	}
}
