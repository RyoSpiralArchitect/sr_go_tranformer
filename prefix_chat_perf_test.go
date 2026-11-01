// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"encoding/json"
	"os"
	"runtime"
	"testing"
	"time"
)

var prefixChatQuestions = []string{
	"A black cat named Kuro lives beside a lighthouse. Its keeper uses a blue lantern when fog rolls in. Remember the cat's name, the lantern's color, and where they live. What is the cat's name?",
	"What color is the lantern the keeper uses in fog?",
	"Where does Kuro live, according to the first message?",
	"Which two details would help a sailor recognize the place?",
	"Recall the cat's name and the lantern's color in one line.",
}

type prefixChatTurn struct {
	PromptIDs         []int      `json:"prompt_ids"`
	Completion        Completion `json:"completion"`
	FirstTokenSeconds float64    `json:"first_token_seconds"`
	CompletionSeconds float64    `json:"completion_seconds"`
	Computed          int64      `json:"computed_input_tokens"`
	Reused            int64      `json:"reused_input_tokens"`
	PrefixBytes       int64      `json:"retained_prefix_bytes"`
	KVBytes           int64      `json:"allocated_kv_bytes"`
}

// Use the same template, stream, decoded assistant text, and release ordering
// as interactiveChat. The timer includes formatting, encoding and queueing.
func measurePrefixChatTurn(ctx context.Context, e *Engine, tok *hfTokenizer, history []ChatMessage, s Sampling) (prefixChatTurn, error) {
	var turn prefixChatTurn
	before := e.Stats()
	start := time.Now()
	prompt, err := tok.formatChat(history)
	if err != nil {
		return turn, err
	}
	stream, err := e.streamInput(ctx, prompt, s, true)
	if err != nil {
		return turn, err
	}
	// The request owns a fresh ID slice, immutable after admission.
	turn.PromptIDs = stream.request.IDs
	for range stream.Events {
		if turn.FirstTokenSeconds == 0 {
			turn.FirstTokenSeconds = time.Since(start).Seconds()
		}
	}
	turn.Completion, err = stream.Wait()
	turn.CompletionSeconds = time.Since(start).Seconds()
	stream.Close()
	<-stream.Released()
	after := e.Stats()
	turn.Computed = after.InputTokens - before.InputTokens
	turn.Reused = after.PrefixTokens - before.PrefixTokens
	turn.PrefixBytes = after.PrefixBytes
	turn.KVBytes = after.KVAllocatedBytes
	return turn, err
}

func TestHFMistralGrowingChat(t *testing.T) {
	dir := os.Getenv("MONOLITH_CHAT_MODEL")
	if dir == "" {
		t.Skip("manual offline five-turn Mistral probe")
	}
	mode := os.Getenv("MONOLITH_CHAT_MODE")
	cfg := EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: 512 << 20, KernelWorkers: 4}
	switch mode {
	case "disabled":
	case "fixed", "growing":
		cfg.PrefixCacheBytes = 128 << 20
	default:
		t.Fatal("unknown chat cache mode", mode)
	}
	if runtime.GOMAXPROCS(0) != 4 {
		t.Fatal("expected four Go threads")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	m, plan, err := loadHFModelFormat(ctx, dir, 512, 8192, "q4-mixed")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	s := defaultSampling()
	s.MaxTokens, s.Temperature, s.RepeatPenalty, s.Seed = 8, 0, 1, 19
	e, err := NewEngine(m, tok, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	var history []ChatMessage
	var turns []prefixChatTurn
	for i, question := range prefixChatQuestions {
		history = append(history, ChatMessage{Role: "user", Content: question})
		turn, err := measurePrefixChatTurn(ctx, e, tok, history, s)
		if err != nil {
			t.Fatal(err)
		}
		if len(turn.PromptIDs) != turn.Completion.PromptTokens || turn.Computed+turn.Reused != int64(len(turn.PromptIDs)) ||
			turn.KVBytes > cfg.CacheBytes || turn.PrefixBytes > cfg.PrefixCacheBytes {
			t.Fatalf("invalid accounting at turn %d: %+v", i+1, turn)
		}
		turns = append(turns, turn)
		history = append(history, ChatMessage{Role: "assistant", Content: turn.Completion.Text})
		t.Logf("turn %d: %d logical, %d reused, %.3fs to first token", i+1, len(turn.PromptIDs), turn.Reused, turn.FirstTokenSeconds)
	}
	e.Close()
	shutdown := e.Stats()
	if shutdown.PrefixBytes != 0 || shutdown.KVReservedBytes != 0 || shutdown.AdmissionUsed != 0 || shutdown.KernelWorkers != 0 {
		t.Fatal("prefix, reservation or active workers survived engine close", shutdown)
	}
	report := map[string]any{"mode": mode, "threads": runtime.GOMAXPROCS(0), "plan": plan, "sampling": s,
		"questions": prefixChatQuestions, "turns": turns, "shutdown": shutdown}
	f, err := os.OpenFile(os.Getenv("MONOLITH_CHAT_OUTPUT"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(report); err != nil {
		t.Fatal(err)
	}
}
