// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func kvMistralMetadata(t *testing.T) string {
	t.Helper()
	h := hfTestConfig()
	h.ModelType, h.Architectures = "mistral", []string{"MistralForCausalLM"}
	h.Vocab, h.Dim, h.Hidden, h.Layers, h.Heads, h.KVHeads = 32768, 4096, 14336, 32, 32, 8
	h.Context, h.Theta, h.Tied = 32768, 1000000, false
	b, _ := json.Marshal(h)
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		t.Fatal(err)
	}
	obj["sliding_window"] = json.RawMessage("null")
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), obj)
	return dir
}

func TestKVBudgetMistralAdmissionAndNativeBoundary(t *testing.T) {
	dir := kvMistralMetadata(t)
	for _, n := range []int{2048, 4096, 8192} {
		_, c, err := readHFConfig(dir, n)
		if err != nil {
			t.Fatal(n, err)
		}
		bytes := int64(n) * 256 * 1024
		p, err := planInferenceKV(c, n, bytes)
		if err != nil || !p.FitsBudget || p.RequiredBytes != bytes {
			t.Fatal(n, p, err)
		}
		if p.PageBytes != 8<<20 || p.Pages != n/32 {
			t.Fatal(p)
		}
		p, err = planInferenceKV(c, n, bytes-1)
		if err != nil || p.FitsBudget {
			t.Fatal(p, err)
		}
	}
	// Metadata-only fixture has no shards. A default loader must reject the KV
	// budget before opening weights. Explicit budget gets as far as missing files.
	if _, _, err := loadHFModelFormat(context.Background(), dir, 8192, 8192, "q4-mixed"); err == nil || !strings.Contains(err.Error(), "KV budget exceeded") {
		t.Fatal(err)
	}
	if _, _, err := loadHFModelFormatKV(context.Background(), dir, 8192, 8192, "q4-mixed", 2<<30, true); err == nil || strings.Contains(err.Error(), "KV budget") {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := loadHFModelFormatKV(canceled, dir, 8192, 8192, "q4-mixed", 2<<30, true); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	c := Config{Vocab: 258, Dim: 512, Hidden: 704, Layers: 8, Heads: 8, KVHeads: 8, Context: 32768, RopeTheta: 10000, NormEps: 1e-5}
	if err := c.validateShapeLimits(65536, maxParameters); err != nil {
		t.Fatal(err)
	}
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "KV cache exceeds") {
		t.Fatal("native guard changed", err)
	}
	if _, err := NewModel(c, 1); err == nil {
		t.Fatal("native allocation admitted")
	}
}

func TestKVBudgetRoundedPagesAndInvalidPlans(t *testing.T) {
	c := workerTestModel(t, 2).Config
	page := int64(2 * pageTokens * c.KVDim() * c.Layers * 4)
	for _, pos := range []int{1, 31, 32, 33, 63, 64, 65, 128} {
		want := int64((pos+31)/32) * page
		p, err := planInferenceKV(c, pos, want)
		if err != nil || p.RequiredBytes != want || !p.FitsBudget {
			t.Fatal(pos, p, err)
		}
		p, err = planInferenceKV(c, pos, want-1)
		if err != nil || p.FitsBudget {
			t.Fatal(p, err)
		}
	}
	for _, budget := range []int64{-1, 0, maxInferenceKVBytes + 1, math.MaxInt64} {
		if _, err := planInferenceKV(c, 1, budget); err == nil {
			t.Fatal("invalid budget", budget)
		}
	}
	for _, pos := range []int{-1, 0, c.Context + 1, math.MaxInt} {
		if _, err := planInferenceKV(c, pos, 1<<20); err == nil {
			t.Fatal("invalid position", pos)
		}
	}
	c.Dim = math.MaxInt
	if _, err := planInferenceKV(c, 1, 1<<20); err == nil {
		t.Fatal("overflow shape accepted")
	}
}

func TestKVBudgetRejectsBeforeAnyBatchMutation(t *testing.T) {
	m := workerTestModel(t, 2)
	page := int64(2 * pageTokens * m.Config.KVDim() * m.Config.Layers * 4)
	a, b := m.NewCache(), m.NewCache()
	defer a.Close()
	defer b.Close()
	b.budgetBytes = page
	_, err := m.PrefillBatch(context.Background(), []*KVCache{b}, [][]int{make([]int, 32)})
	if err != nil {
		t.Fatal(err)
	}
	saved := append([]float32(nil), b.Pages[0].Data...)
	var w decodeWorkspace
	_, err = m.prefillBorrowed(context.Background(), []*KVCache{a, b}, [][]int{{1}, {2}}, nil, &w)
	if err == nil || !strings.Contains(err.Error(), "KV budget exceeded") {
		t.Fatal(err)
	}
	if a.Pos != 0 || len(a.Pages) != 0 || b.Pos != 32 || len(b.Pages) != 1 || !reflect.DeepEqual(saved, b.Pages[0].Data) || w.floatBytes() != 0 {
		t.Fatal("failed admission changed batch state")
	}
	assertWorkspaceReferencesCleared(t, &w)
	// A separately admitted full-context cache is independent of the small one.
	c, err := m.NewCacheWithBudget(4 * page)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err = m.PrefillBatch(context.Background(), []*KVCache{c}, [][]int{make([]int, 33)}); err != nil {
		t.Fatal(err)
	}
	if _, err = m.NewCacheWithBudget(4*page - 1); err == nil {
		t.Fatal("underfunded full context")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = m.PrefillBatch(canceled, []*KVCache{a}, [][]int{{1}})
	if !errors.Is(err, context.Canceled) || len(a.Pages) != 0 {
		t.Fatal(err)
	}
	// Generation reserves the whole request before touching model weights.
	empty := &Model{Config: m.Config, directKVBytes: page}
	sampling := defaultSampling()
	sampling.MaxTokens = 33
	_, err = Generate(context.Background(), empty, Tokenizer{}, "", sampling)
	if err == nil || !strings.Contains(err.Error(), "KV budget exceeded") {
		t.Fatal(err)
	}
}

func TestKVBudgetCLIValidationBeforeIO(t *testing.T) {
	for _, action := range []string{"check", "generate", "ids"} {
		for _, value := range []string{"0", "-1", "1048577", "9223372036854775807"} {
			args := []string{"infer", "local", "-repo", "missing/model", action, "-kv-mib", value}
			if action == "ids" {
				args = append(args, "-ids", "1")
			}
			err := run(context.Background(), args)
			if err == nil || !strings.Contains(err.Error(), "kv-mib") {
				t.Fatal(action, value, err)
			}
		}
	}
}

func TestKVBudget8192PositionHFFixture(t *testing.T) {
	// A genuinely executed 8K cache with tiny weights. This is not a 7B speed or
	// quality measurement; it checks the supported position range and page tail.
	h := hfTestConfig()
	h.Dim, h.Hidden, h.Layers, h.Heads, h.KVHeads, h.Context = 2, 2, 1, 1, 1, 8192
	dir := t.TempDir()
	hfWriteJSON(t, filepath.Join(dir, "config.json"), h)
	_, c, err := readHFConfig(dir, 8192)
	if err != nil {
		t.Fatal(err)
	}
	header := map[string]any{}
	var payload []byte
	for _, spec := range hfTensorSpecs(c) {
		n := 1
		for _, d := range spec.Shape {
			n *= d
		}
		start := len(payload)
		payload = append(payload, make([]byte, n*4)...)
		header[spec.Name] = map[string]any{"dtype": "F32", "shape": spec.Shape, "data_offsets": []int{start, len(payload)}}
	}
	hfWriteShard(t, filepath.Join(dir, "model.safetensors"), header, payload)
	m, _, err := loadHFModelFormatKV(context.Background(), dir, 8192, 1, "f32", 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	defer cache.Close()
	var w decodeWorkspace
	for start := 0; start < 8192; start += 128 {
		logits, err := m.prefillBorrowed(context.Background(), []*KVCache{cache}, [][]int{make([]int, 128)}, nil, &w)
		if err != nil {
			t.Fatal(start, err)
		}
		for _, x := range logits {
			if x != 0 {
				t.Fatal("nonzero fixture logit", x)
			}
		}
	}
	if cache.Pos != 8192 || len(cache.Pages) != 256 {
		t.Fatal(cache.Pos, len(cache.Pages))
	}
	if _, err = m.PrefillBatch(context.Background(), []*KVCache{cache}, [][]int{{1}}); err == nil {
		t.Fatal("context overrun accepted")
	}
	for _, p := range cache.Pages {
		for _, x := range p.Data {
			if x != 0 {
				t.Fatal("nonzero fixture KV")
			}
		}
	}
	// Record no model artifacts outside the temporary directory.
	if _, err = os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
}

func TestKVBudgetLoaderAndEnginePolicies(t *testing.T) {
	dir := hfCLIFixture(t)
	_, c, err := readHFConfig(dir, 8192)
	if err != nil {
		t.Fatal(err)
	}
	full, err := planInferenceKV(c, c.Context, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if full.RequiredBytes < 2*full.PageBytes {
		t.Fatal(full)
	}
	// Engine metadata may describe a longer context than the pool can admit
	// per request. Its loaded model still bounds direct, unpooled caches.
	m, _, err := loadHFModelFormatKV(context.Background(), dir, 8192, 1, "f32", full.PageBytes, false)
	if err != nil {
		t.Fatal(err)
	}
	if m.NewCache().budgetBytes != full.PageBytes {
		t.Fatal("loader budget was not captured")
	}
	if _, _, err := loadHFModelFormatKV(context.Background(), dir, 8192, 1, "f32", full.PageBytes, true); err == nil {
		t.Fatal("direct loader skipped full-context admission")
	}
	tok := testHFTokenizer(t, dir)
	e, err := newEngine(m, tok, EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: 2 * full.PageBytes})
	if err != nil {
		t.Fatal(err)
	}
	// Pool reservation governs pooled caches even when model direct budget is
	// one page. This permits the admitted two-page request with no bypass.
	cache := m.NewCache()
	cache.pool = &e.pool
	cache.reserved = 2
	e.pool.Reserved = 2
	defer cache.Close()
	if _, err = m.PrefillBatch(context.Background(), []*KVCache{cache}, [][]int{make([]int, 33)}); err != nil {
		t.Fatal(err)
	}
	if cache.Pos != 33 || len(cache.Pages) != 2 || e.pool.Allocated != 2 {
		t.Fatal("pooled admission changed")
	}
}
