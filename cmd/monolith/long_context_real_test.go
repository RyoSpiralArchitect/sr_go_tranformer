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

// Opt-in bounded original-model probe. An 8K configured context is not an
// 8K executed prompt; the receipt records the actual Completion prompt length.
func TestHFLongContextReal(t *testing.T) {
	dir := os.Getenv("MONOLITH_KV_MODEL")
	if dir == "" {
		t.Skip("set MONOLITH_KV_MODEL and MONOLITH_KV_OUTPUT for the bounded 7B probe")
	}
	output := os.Getenv("MONOLITH_KV_OUTPUT")
	if output == "" {
		t.Fatal("MONOLITH_KV_OUTPUT is required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	m, plan, err := loadHFModelFormatKV(ctx, dir, 8192, 8192, "q4-mixed", 2048<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	s := defaultSampling()
	s.MaxTokens = 1
	s.Temperature = 0
	s.Seed = 19
	s.RepeatPenalty = 1
	prompt := strings.Repeat("A black cat lives near the lighthouse. The light guides ships safely home. ", 16) + "The cat"
	result, err := Generate(ctx, m, tok, prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	kv, err := planInferenceKV(m.Config, m.Config.Context, m.directKVBudget())
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	err = json.NewEncoder(f).Encode(struct {
		Format     string       `json:"format"`
		Plan       HFPlan       `json:"plan"`
		KVPlan     KVMemoryPlan `json:"kv_plan"`
		Completion Completion   `json:"completion"`
		Note       string       `json:"note"`
	}{"monolith-long-context-bounded-probe-v1", plan, kv, result, "8K configured context, bounded prompt and one output; not a full-8K throughput or quality result."})
	closeErr := f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	t.Logf("configured=%d actual_prompt=%d completion=%d", m.Config.Context, result.PromptTokens, result.CompletionTokens)
}
