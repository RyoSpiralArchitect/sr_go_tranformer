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

func TestHFLlama3TokenizerReference(t *testing.T) {
	testHFTokenizerReference(t, "testdata/hf-llama3-tokenizer", "testdata/hf-llama3-tokenizer/reference.json")
}

func TestLlama3TextFramingAndBounds(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-llama3-tokenizer")
	ctx := context.Background()
	ids, err := tok.encodePrompt(ctx, "inference", 2)
	if err != nil || !reflect.DeepEqual(ids, []int{tok.bos, tok.whole["inference"]}) {
		t.Fatal("BOS + ignore_merges word", ids, err)
	}
	ids, err = tok.encodePrompt(ctx, "<|begin_of_text|>inference", 3)
	if err != nil || !reflect.DeepEqual(ids, []int{tok.bos, tok.bos, tok.whole["inference"]}) {
		t.Fatal("literal BOS must not be deduplicated", ids, err)
	}
	ids, err = tok.encodeText(ctx, "<|begin_of_text|>inference", 2, false)
	if err != nil || !reflect.DeepEqual(ids, []int{tok.bos, tok.whole["inference"]}) {
		t.Fatal("explicit preframed input", ids, err)
	}
	for _, tc := range []struct {
		text  string
		limit int
	}{{"", 0}, {"inference", 1}, {"\xff", 4}, {strings.Repeat("x", 65537), 70000}} {
		if _, err := tok.encodePrompt(ctx, tc.text, tc.limit); err == nil {
			t.Fatal("accepted invalid/beyond-budget prompt", tc.limit)
		}
	}
	ids, err = tok.encodePrompt(ctx, "", 1)
	if err != nil || !reflect.DeepEqual(ids, []int{tok.bos}) {
		t.Fatal("empty text still has BOS", ids, err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := tok.encodePrompt(canceled, "", 1); err != context.Canceled {
		t.Fatal(err)
	}
	if tok.chat {
		t.Fatal("base model must not acquire an invented chat template")
	}
}

func TestLlama3PinnedUnicodeClasses(t *testing.T) {
	var oracle struct {
		Tokenizers string
		Classes    map[string][][2]rune
	}
	if err := readBoundedJSON("testdata/hf-llama3-tokenizer/unicode.json", 1<<20, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Tokenizers != "0.22.2" {
		t.Fatal("unexpected Unicode oracle")
	}
	for name, table := range map[string][][2]rune{"Letter": llama3LetterRanges, "Number": llama3NumberRanges, "Space": llama3SpaceRanges} {
		want := oracle.Classes[name]
		i := 0
		for r := rune(0); r <= 0x10ffff; r++ {
			for i < len(want) && r > want[i][1] {
				i++
			}
			inside := i < len(want) && r >= want[i][0]
			if inHFRanges(r, table) != inside {
				t.Fatalf("%s U+%04X", name, r)
			}
		}
	}
}

func TestLlama3TokenizerRejectsPipelineDrift(t *testing.T) {
	for _, change := range []string{"bos-template", "digits", "normalizer", "ignore-merges", "input-names", "extra", "class"} {
		t.Run(change, func(t *testing.T) {
			dir := t.TempDir()
			for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
				b, err := os.ReadFile(filepath.Join("testdata/hf-llama3-tokenizer", name))
				if err != nil {
					t.Fatal(err)
				}
				var raw map[string]any
				if err = json.Unmarshal(b, &raw); err != nil {
					t.Fatal(err)
				}
				if name == "tokenizer.json" {
					switch change {
					case "bos-template":
						raw["post_processor"] = nil
					case "digits":
						pre := raw["pre_tokenizer"].(map[string]any)["pretokenizers"].([]any)
						pre[0].(map[string]any)["pattern"] = map[string]any{"Regex": qwenSplitPattern}
					case "normalizer":
						raw["normalizer"] = map[string]any{"type": "NFC"}
					case "ignore-merges":
						raw["model"].(map[string]any)["ignore_merges"] = false
					}
				}
				if name == "tokenizer_config.json" {
					switch change {
					case "input-names":
						raw["model_input_names"] = []string{"input_ids,attention_mask"}
					case "extra":
						raw["extra_special_tokens"] = map[string]any{"image_token": "<image>"}
					case "class":
						raw["tokenizer_class"] = "LlamaTokenizer"
					}
				}
				hfWriteJSON(t, filepath.Join(dir, name), raw)
			}
			h, _, err := readHFConfig(dir, 64)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = loadHFTokenizer(dir, h); err == nil {
				t.Fatal("accepted drift")
			}
		})
	}
}

func TestLlama3StreamingCleanupOracle(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-llama3-tokenizer")
	var ref hfTokenizerReference
	if err := readBoundedJSON("testdata/hf-llama3-tokenizer/reference.json", 4<<20, &ref); err != nil {
		t.Fatal(err)
	}
	for _, tc := range ref.DecodeCases {
		var raw []byte
		for _, id := range tc.IDs {
			raw = append(raw, tok.pieces[id]...)
		}
		for split := 0; split <= len(raw); split++ {
			d := newTextDecoder(tok)
			got := d.Push(raw[:split], false) + d.Push(raw[split:], false) + d.Push(nil, true)
			if got != tc.Decoded {
				t.Fatalf("split %d: got %q want %q", split, got, tc.Decoded)
			}
		}
		d := newTextDecoder(tok)
		var got strings.Builder
		for _, b := range raw {
			got.WriteString(d.Push([]byte{b}, false))
			for i, s := range d.held {
				if len(s) >= len(hfCleanup[i][0]) {
					t.Fatal("unbounded cleanup buffer")
				}
			}
		}
		got.WriteString(d.Push(nil, true))
		if got.String() != tc.Decoded {
			t.Fatalf("byte stream got %q want %q", got.String(), tc.Decoded)
		}
	}
}

func TestLlama3TextEngineFramingAndFlush(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-llama3-tokenizer")
	for _, winner := range []int{tok.bytes[' '], tok.eos} {
		m := hfServingModel(t, tok, winner)
		s := defaultSampling()
		s.Temperature, s.RepeatPenalty, s.MaxTokens = 0, 1, 3
		want, err := Generate(context.Background(), m, tok, "x", s)
		if err != nil {
			t.Fatal(err)
		}
		if want.PromptTokens != 2 || want.TokenIDs[0] != winner {
			t.Fatal(want)
		}
		if winner == tok.bytes[' '] && want.Text != "   " {
			t.Fatal("trailing whitespace", want)
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
		for event := range stream.Events {
			text += event.Text
		}
		got, err := stream.Wait()
		<-stream.Released()
		e.Close()
		if err != nil || !reflect.DeepEqual(got, want) || text != want.Text {
			t.Fatal("stream/Generate", got, want, text, err)
		}
		if e.Stats().AdmissionUsed != 0 || e.Stats().KVReservedBytes != 0 {
			t.Fatal("retained resources")
		}
	}
}
