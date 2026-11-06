// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMistralFullAttentionBoundary(t *testing.T) {
	b, err := os.ReadFile("testdata/hf-mistral/f32/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, key, value string
	}{
		{"missing window", "sliding_window", ""},
		{"sliding window", "sliding_window", "4096"},
		{"zero window", "sliding_window", "0"},
		{"false window", "sliding_window", "false"},
		{"string window", "sliding_window", `"null"`},
		{"Qwen switch", "use_sliding_window", "false"},
		{"unknown scaling", "rope_scaling", `{"rope_type":"linear","factor":2}`},
		{"MoE", "architectures", `["MixtralForCausalLM"]`},
		{"head shape", "head_dim", "8"},
		{"attention bias", "attention_bias", "true"},
		{"MLP bias", "mlp_bias", "true"},
		{"dropout", "attention_dropout", "0.1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var cfg map[string]json.RawMessage
			if err := json.Unmarshal(b, &cfg); err != nil {
				t.Fatal(err)
			}
			if tc.value == "" {
				delete(cfg, tc.key)
			} else {
				cfg[tc.key] = json.RawMessage(tc.value)
			}
			dir := t.TempDir()
			hfWriteJSON(t, filepath.Join(dir, "config.json"), cfg)
			if _, _, err := readHFConfig(dir, 64); err == nil {
				t.Fatal("accepted unsupported Mistral computation")
			}
		})
	}
	var cfg map[string]json.RawMessage
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["head_dim"] = json.RawMessage("4")
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), cfg)
	if _, c, err := readHFConfig(dir, 64); err != nil || c.Context != 64 {
		t.Fatal("valid explicit head dimension", c, err)
	}
}

func TestMistral7BPlanBeforeAllocation(t *testing.T) {
	h := hfTestConfig()
	h.ModelType, h.Architectures = "mistral", []string{"MistralForCausalLM"}
	h.Vocab, h.Dim, h.Hidden, h.Layers, h.Heads, h.KVHeads = 32768, 4096, 14336, 32, 32, 8
	h.Context, h.Theta, h.Tied = 32768, 1000000, false
	b, _ := json.Marshal(h)
	var cfg map[string]json.RawMessage
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["sliding_window"] = json.RawMessage("null")
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), cfg)
	_, c, err := readHFConfig(dir, 128)
	if err != nil {
		t.Fatal(err)
	}
	if got := hfParameterCount(c, h); got != 7248023552 {
		t.Fatal("parameter accounting", got)
	}
	if c.Validate() == nil {
		t.Fatal("HF profile expanded native training limits")
	}
	if _, err := openHFArchive(dir, 128, 1<<20); err == nil || !strings.Contains(err.Error(), "-weight-mib") {
		t.Fatal("must reject before missing weight files are opened", err)
	}
}

// JSON spelling must not bypass profile-specific defaults and restrictions.
func TestMistralEscapedDiscriminator(t *testing.T) {
	b, err := os.ReadFile("testdata/hf-mistral/f32/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, window := range []string{"", "null", "4096"} {
		t.Run("window="+window, func(t *testing.T) {
			var cfg map[string]json.RawMessage
			if err := json.Unmarshal(b, &cfg); err != nil {
				t.Fatal(err)
			}
			cfg["model_type"] = json.RawMessage(`"mis\u0074ral"`)
			delete(cfg, "head_dim")
			delete(cfg, "sliding_window")
			if window != "" {
				cfg["sliding_window"] = json.RawMessage(window)
			}
			dir := t.TempDir()
			hfWriteJSON(t, filepath.Join(dir, "config.json"), cfg)
			_, _, err := readHFConfig(dir, 64)
			if (err == nil) != (window == "null") {
				t.Fatalf("window=%q: %v", window, err)
			}
		})
	}
}
