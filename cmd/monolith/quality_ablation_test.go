// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func TestQualityAblation(t *testing.T) {
	dir := os.Getenv("MONOLITH_ABLATION_MODEL")
	if dir == "" {
		t.Skip("manual offline ablation")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	ref, _, err := loadHFModelFormat(ctx, dir, 128, 1024, "f32")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *ref.hf)
	if err != nil {
		t.Fatal(err)
	}
	corpusBytes, err := os.ReadFile("validation/quality/corpus.json")
	if err != nil {
		t.Fatal(err)
	}
	var corpus qualityCorpus
	if err = json.Unmarshal(corpusBytes, &corpus); err != nil || len(corpus.Selection) != 4 {
		t.Fatal("selection corpus", err)
	}
	prompts := corpus.Selection
	ids := make([][]int, len(prompts))
	base := make([][]float32, len(prompts))
	for i, prompt := range prompts {
		ids[i], err = tok.encodePrompt(ctx, prompt, 64)
		if err != nil {
			t.Fatal(err)
		}
		r, err := inferHFIDs(ctx, ref, ids[i], 1, 16, true)
		if err != nil {
			t.Fatal(err)
		}
		base[i] = r.Logits[0]
	}
	m, _, err := loadHFModelFormat(ctx, dir, 128, 1024, "q4")
	if err != nil {
		t.Fatal(err)
	}
	q4s := make([]*q4Weights, len(m.Params))
	q8s := make([]*q8Weights, len(m.Params))
	for i, p := range m.Params {
		q4s[i] = p.Q4
	}
	type result struct {
		Policy string    `json:"policy"`
		Bytes  int64     `json:"weight_bytes"`
		KL     []float64 `json:"kl"`
		Text   []string  `json:"text"`
	}
	var results []result
	for _, policy := range []string{"q4", "q8-embedding", "q8-attention", "q8-mlp", "q8-embedding-attention", "q8"} {
		r := result{Policy: policy}
		for i, p := range m.Params {
			if q4s[i] == nil {
				r.Bytes += int64(len(p.Data)) * 4
				continue
			}
			embedding := p.Name == "wte.weight" || p.Name == "token_embedding" || p.Name == "output"
			mlp := strings.Contains(p.Name, "mlp.") || strings.HasSuffix(p.Name, ".gate") || strings.HasSuffix(p.Name, ".up") || strings.HasSuffix(p.Name, ".down")
			rescue := policy == "q8" || policy == "q8-embedding" && embedding || policy == "q8-attention" && !embedding && !mlp || policy == "q8-mlp" && mlp || policy == "q8-embedding-attention" && !mlp
			p.Q8 = nil
			p.Q4 = q4s[i]
			if rescue {
				if q8s[i] == nil {
					q8s[i] = quantizeTestParam(t, ref.Params[i]).Q8
				}
				p.Q8 = q8s[i]
				p.Q4 = nil
				r.Bytes += int64(len(p.Q8.Values) + len(p.Q8.Scales)*4)
			} else {
				r.Bytes += int64(len(p.Q4.Values) + len(p.Q4.Scales)*4)
			}
		}
		for i := range prompts {
			got, err := inferHFIDs(ctx, m, ids[i], 8, 16, true)
			if err != nil {
				t.Fatal(err)
			}
			r.KL = append(r.KL, qualityKL(base[i], got.Logits[0]))
			r.Text = append(r.Text, tok.Decode(got.TokenIDs))
		}
		results = append(results, r)
		t.Log(policy, r.KL, r.Text)
	}
	out := struct {
		Prompts []string `json:"selection_prompts"`
		Results []result `json:"results"`
	}{prompts, results}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(os.Getenv("MONOLITH_ABLATION_OUTPUT"), append(b, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}
