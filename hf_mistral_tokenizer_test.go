// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestHFMistralTokenizerReference(t *testing.T) {
	testHFTokenizerReference(t, "testdata/hf-mistral-tokenizer", "testdata/hf-mistral-tokenizer/reference.json")
}

func TestMistralTextAndChatBounds(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-mistral-tokenizer")
	ctx := context.Background()
	for _, tc := range []struct {
		text  string
		limit int
	}{{"", 0}, {"abc", 1}, {"\xff", 128}, {strings.Repeat("a", 65537), 70000}} {
		if _, err := tok.encodePrompt(ctx, tc.text, tc.limit); err == nil {
			t.Fatal("accepted invalid prompt", tc.limit)
		}
	}
	ids, err := tok.encodePrompt(ctx, "<s>", 2)
	if err != nil || !reflect.DeepEqual(ids, []int{tok.bos, tok.bos}) {
		t.Fatal("literal BOS", ids, err)
	}
	ids, err = tok.encodeText(ctx, "<s>", 1, false)
	if err != nil || !reflect.DeepEqual(ids, []int{tok.bos}) {
		t.Fatal("chat framing", ids, err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := tok.encodePrompt(canceled, "", 1); err != context.Canceled {
		t.Fatal(err)
	}
	for _, m := range [][]ChatMessage{nil, make([]ChatMessage, 129), {{Role: "tool", Content: "x"}}, {{Role: "assistant", Content: "x"}}, {{Role: "user"}, {Role: "user"}}, {{Role: "user"}, {Role: "system"}}, {{Role: "user", Content: "\xff"}}, {{Role: "user", Content: strings.Repeat("x", 1<<20)}}} {
		if _, err := tok.formatChat(m); err == nil {
			t.Fatal("accepted invalid chat")
		}
	}
}

func TestMistralTokenizerRejectsDrift(t *testing.T) {
	for _, change := range []string{"normalizer", "prepend", "split", "bos", "decoder", "fallback", "missing-byte", "byte-alias", "special", "missing-legacy", "legacy", "prefix", "cleanup", "extra", "chat"} {
		t.Run(change, func(t *testing.T) {
			dir := t.TempDir()
			for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
				b, err := os.ReadFile(filepath.Join("testdata/hf-mistral-tokenizer", name))
				if err != nil {
					t.Fatal(err)
				}
				var v map[string]any
				if err = json.Unmarshal(b, &v); err != nil {
					t.Fatal(err)
				}
				if name == "tokenizer.json" {
					switch change {
					case "normalizer":
						v["normalizer"] = map[string]any{"type": "NFC"}
					case "prepend":
						v["pre_tokenizer"].(map[string]any)["prepend_scheme"] = "always"
					case "split":
						v["pre_tokenizer"].(map[string]any)["split"] = true
					case "bos":
						v["post_processor"] = nil
					case "decoder":
						v["decoder"] = map[string]any{"type": "ByteLevel"}
					case "fallback":
						v["model"].(map[string]any)["byte_fallback"] = false
					case "missing-byte":
						x := v["model"].(map[string]any)["vocab"].(map[string]any)
						x["missing-byte"] = x["<0xFF>"]
						delete(x, "<0xFF>")
					case "byte-alias":
						x := v["model"].(map[string]any)["vocab"].(map[string]any)
						x["<0xff>"] = x["<0xFF>"]
						delete(x, "<0xFF>")
					case "special":
						v["added_tokens"].([]any)[0].(map[string]any)["normalized"] = true
					}
				}
				if name == "tokenizer_config.json" {
					switch change {
					case "missing-legacy":
						delete(v, "legacy")
					case "legacy":
						v["legacy"] = true
					case "prefix":
						v["add_prefix_space"] = false
					case "cleanup":
						v["clean_up_tokenization_spaces"] = true
					case "extra":
						v["split_special_tokens"] = true
					case "chat":
						v["chat_template"] = v["chat_template"].(string) + " "
					}
				}
				hfWriteJSON(t, filepath.Join(dir, name), v)
			}
			h, _, err := readHFConfig(dir, 64)
			if err != nil {
				t.Fatal(err)
			}
			tok, err := loadHFTokenizer(dir, h)
			if change == "chat" {
				if err != nil || tok.chat {
					t.Fatal("chat identity", tok, err)
				}
				if _, err = tok.encodePrompt(context.Background(), "Hi", 64); err != nil {
					t.Fatal(err)
				}
				if _, err = tok.formatChat([]ChatMessage{{Role: "user", Content: "Hi"}}); err == nil {
					t.Fatal("arbitrary template accepted")
				}
			} else if err == nil {
				t.Fatal("accepted pipeline drift")
			}
		})
	}
}

func TestMistralStreamingByteRunsAndEOS(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-mistral-tokenizer")
	for _, winner := range []int{tok.bos, tok.eos, tok.bytes[255], tok.symbols['▁']} {
		m := hfServingModel(t, tok, winner)
		for i := range m.Emb.Data {
			m.Emb.Data[i] = 1
		}
		for j := 0; j < m.Config.Dim; j++ {
			m.Emb.Data[winner*m.Config.Dim+j] = 2
		}
		s := defaultSampling()
		s.Temperature = 0
		s.RepeatPenalty = 1
		s.MaxTokens = 3
		want, err := Generate(context.Background(), m, tok, "x", s)
		if err != nil {
			t.Fatal(err)
		}
		if want.TokenIDs[0] != winner {
			t.Fatal("winner", want)
		}
		e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 2, Queue: 2, CacheBytes: 2 << 20, KernelWorkers: 2})
		if err != nil {
			t.Fatal(err)
		}
		stream, err := e.Stream(context.Background(), "x", s)
		if err != nil {
			e.Close()
			t.Fatal(err)
		}
		var text string
		for ev := range stream.Events {
			text += ev.Text
		}
		got, err := stream.Wait()
		<-stream.Released()
		e.Close()
		if err != nil || !reflect.DeepEqual(got, want) || text != want.Text {
			t.Fatal("stream mismatch", got, want, text, err)
		}
		if e.Stats().AdmissionUsed != 0 || e.Stats().KVReservedBytes != 0 || e.Stats().KernelWorkers != 0 {
			t.Fatal("resources retained")
		}
		if winner == tok.bytes[255] && text != "���" {
			t.Fatal("fallback run not flushed", text)
		}
		if winner == tok.eos && (len(got.TokenIDs) != 1 || got.FinishReason != "eos") {
			t.Fatal("EOS", got)
		}
	}
}

func TestMistralHubFetchAndOffline(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
		b, err := os.ReadFile(filepath.Join("testdata/hf-mistral-tokenizer", name))
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	h, c, err := readHFConfig(dir, 64)
	if err != nil {
		t.Fatal(err)
	}
	header := map[string]any{}
	var payload []byte
	for _, spec := range hfProfileSpecs(c, h) {
		n := 1
		for _, d := range spec.Shape {
			n *= d
		}
		start := len(payload)
		payload = append(payload, make([]byte, n*4)...)
		header[spec.Name] = map[string]any{"dtype": "F32", "shape": spec.Shape, "data_offsets": []int{start, len(payload)}}
	}
	hfWriteShard(t, filepath.Join(dir, "model.safetensors"), header, payload)
	hub, source, counts := fakeHFHub(t, false, dir)
	cached, manifest, err := hub.resolve(context.Background(), source)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Revision != strings.Repeat("a", 40) {
		t.Fatal(manifest)
	}
	check, err := checkHFFormat(cached, 64, 1, true, "q4-mixed")
	if err != nil || check.TokenizerProfile != "mistral-v3-metaspace-bytefallback" || check.WeightValuesChecked {
		t.Fatal("Mistral preparation profile", check, err)
	}
	before := counts()
	t.Setenv("HF_HUB_OFFLINE", "1")
	again, _, err := hub.resolve(context.Background(), source)
	if err != nil || again != cached || !reflect.DeepEqual(before, counts()) {
		t.Fatal("offline fetch", again, err)
	}
	m, _, err := loadHFModel(context.Background(), cached, 64, 1)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(cached, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	s := defaultSampling()
	s.MaxTokens = 2
	s.Temperature = 0
	if _, err := GenerateChat(context.Background(), m, tok, []ChatMessage{{Role: "user", Content: "Hi"}}, s); err != nil {
		t.Fatal(err)
	}
}

// Rust accepts a leading plus in ByteFallback hex, unlike Go's ParseUint.
// Keep the complete canonical byte table so this exercises alias rejection,
// rather than a missing-byte or vocabulary-size failure.
func TestMistralRejectsAdditionalByteAliases(t *testing.T) {
	for _, alias := range []string{"<0x+A>", "<0x+0>", "<0xff>"} {
		t.Run(alias, func(t *testing.T) {
			dir := t.TempDir()
			for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
				b, err := os.ReadFile(filepath.Join("testdata/hf-mistral-tokenizer", name))
				if err != nil {
					t.Fatal(err)
				}
				var raw map[string]any
				if err = json.Unmarshal(b, &raw); err != nil {
					t.Fatal(err)
				}
				if name == "config.json" {
					raw["vocab_size"] = raw["vocab_size"].(float64) + 1
				}
				if name == "tokenizer.json" {
					v := raw["model"].(map[string]any)["vocab"].(map[string]any)
					v[alias] = len(v)
				}
				hfWriteJSON(t, filepath.Join(dir, name), raw)
			}
			h, _, err := readHFConfig(dir, 64)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = loadHFTokenizer(dir, h); err == nil || !strings.Contains(err.Error(), "byte fallback alias") {
				t.Fatal("accepted alias or wrong boundary", err)
			}
		})
	}
}
