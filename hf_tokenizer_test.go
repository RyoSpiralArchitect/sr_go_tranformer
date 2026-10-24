package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

type hfTokenizerReference struct {
	Files map[string]string `json:"files"`
	Cases []struct {
		Text    string
		IDs     []int
		Decoded string
	}
	Chats []struct {
		Messages []ChatMessage
		Prompt   string
		IDs      []int
	}
	DecodeCases []struct {
		IDs     []int
		Decoded string
	} `json:"decode_cases"`
}

func testHFTokenizer(t *testing.T, dir string) *hfTokenizer {
	t.Helper()
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err == nil {
		h, _, err := readHFConfig(dir, 64)
		if err != nil {
			t.Fatal(err)
		}
		tok, err := loadHFTokenizer(dir, h)
		if err != nil {
			t.Fatal(err)
		}
		return tok
	}
	var cfg struct {
		Vocab   int `json:"vocab_size"`
		Context int `json:"model_max_length"`
	}
	if err := readBoundedJSON(filepath.Join(dir, "tokenizer_config.json"), 1<<20, &cfg); err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, hfConfig{Vocab: cfg.Vocab, Context: cfg.Context, BOS: 1, EOS: 2})
	if err != nil {
		t.Fatal(err)
	}
	return tok
}
func testHFTokenizerReference(t *testing.T, dir, path string) {
	t.Helper()
	tok := testHFTokenizer(t, dir)
	var ref hfTokenizerReference
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	for name, want := range ref.Files {
		if name != "tokenizer.json" && name != "tokenizer_config.json" {
			t.Fatal("unexpected reference file")
		}
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(b)
		if hex.EncodeToString(sum[:]) != want {
			t.Fatal("tokenizer/reference identity mismatch", name)
		}
	}
	for _, tc := range ref.Cases {
		got, err := tok.encodePrompt(context.Background(), tc.Text, 8192)
		if err != nil {
			t.Fatalf("%q: %v", tc.Text, err)
		}
		if !reflect.DeepEqual(got, tc.IDs) {
			t.Fatalf("IDs for %q: got %v, want %v", tc.Text, got, tc.IDs)
		}
		if got := tok.Decode(tc.IDs); got != tc.Decoded {
			t.Fatalf("decoded %q: got %q want %q", tc.Text, got, tc.Decoded)
		}
	}
	for _, tc := range ref.Chats {
		prompt, err := tok.formatChat(tc.Messages)
		if err != nil {
			t.Fatal(err)
		}
		if prompt != tc.Prompt {
			t.Fatalf("chat text %q != %q", prompt, tc.Prompt)
		}
		ids, err := tok.encodePrompt(context.Background(), prompt, 8192)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ids, tc.IDs) {
			t.Fatal("chat IDs", ids, tc.IDs)
		}
	}
	for _, tc := range ref.DecodeCases {
		if got := tok.Decode(tc.IDs); got != tc.Decoded {
			t.Fatalf("decode IDs %v got %q want %q", tc.IDs, got, tc.Decoded)
		}
		d := incrementalText{hf: true}
		var out string
		for i, id := range tc.IDs {
			out += d.Push(tok.pieces[id], i == len(tc.IDs)-1)
		}
		if out != tc.Decoded {
			t.Fatalf("stream IDs %v got %q want %q", tc.IDs, out, tc.Decoded)
		}
	}
	t.Logf("matched %d encoding, %d chat and %d decoding cases", len(ref.Cases), len(ref.Chats), len(ref.DecodeCases))
}
func TestHFTokenizerReference(t *testing.T) {
	testHFTokenizerReference(t, "testdata/hf-tokenizer", "testdata/hf-tokenizer/reference.json")
}
func TestHFQwenTokenizerReference(t *testing.T) {
	testHFTokenizerReference(t, "testdata/hf-qwen-tokenizer", "testdata/hf-qwen-tokenizer/reference.json")
}
func TestHFLocalTokenizerReference(t *testing.T) {
	dir := os.Getenv("MONOLITH_HF_MODEL")
	ref := os.Getenv("MONOLITH_HF_TOKENIZER_REFERENCE")
	if dir == "" || ref == "" {
		t.Skip("optional pinned real-model tokenizer comparison")
	}
	testHFTokenizerReference(t, dir, ref)
}
func TestHFTokenizerBoundariesAndConcurrentUse(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	for _, text := range []string{strings.Repeat("x", 65537), "\xff", strings.Repeat("a ", (1<<19)+1)} {
		if _, err := tok.encodePrompt(context.Background(), text, 8192); err == nil {
			t.Fatal("accepted invalid/excessive prompt")
		}
	}
	if _, err := tok.encodePrompt(context.Background(), "Hello", 0); err == nil {
		t.Fatal("ignored token budget")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tok.encodePrompt(ctx, "hello", 8192); err != context.Canceled {
		t.Fatal(err)
	}
	for _, messages := range [][]ChatMessage{nil, {{Role: "tool", Content: "x"}}, {{Role: "user", Content: strings.Repeat("x", 1<<20)}}} {
		if _, err := tok.formatChat(messages); err == nil {
			t.Fatal("accepted invalid chat")
		}
	}
	text := " 123 café 猫 🐈‍⬛ can't\n<|im_end|>"
	want, err := tok.encodePrompt(context.Background(), text, 8192)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				got, err := tok.encodePrompt(context.Background(), text, 8192)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Error("concurrent tokenization", got, err)
				}
			}
		}()
	}
	wg.Wait()
}

func TestHFChatFileRejectsUnknownFields(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	path := filepath.Join(t.TempDir(), "messages.json")
	for _, body := range []string{
		`[{"role":"user","content":"hi","tool_calls":[]}]`,
		`[{"role":"user","role":"assistant","content":"hi"}]`,
	} {
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := inferencePrompt(context.Background(), flags("test"), tok, "", false, path); err == nil {
			t.Fatal("accepted ambiguous/unsupported chat file")
		}
	}
	if err := os.WriteFile(path, []byte(`[{"role":"user","content":"hi"}]`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := inferencePrompt(context.Background(), flags("test"), tok, "", false, path); err != nil {
		t.Fatal(err)
	}
}
func TestHFTokenizerRejectUnsupported(t *testing.T) {
	for _, kind := range []string{"normalizer", "post_processor", "padding", "pipeline", "dropout", "vocab", "special", "config special", "template"} {
		t.Run(kind, func(t *testing.T) {
			dir := t.TempDir()
			for _, name := range []string{"tokenizer.json", "tokenizer_config.json"} {
				b, err := os.ReadFile(filepath.Join("testdata/hf-tokenizer", name))
				if err != nil {
					t.Fatal(err)
				}
				if err = os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			var raw map[string]any
			name := "tokenizer.json"
			if kind == "template" || kind == "config special" {
				name = "tokenizer_config.json"
			}
			if err := readBoundedJSON(filepath.Join(dir, name), 1<<20, &raw); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "normalizer", "post_processor", "padding":
				raw[kind] = map[string]any{"type": "Unknown"}
			case "pipeline":
				raw["pre_tokenizer"] = map[string]any{"type": "ByteLevel"}
			case "dropout":
				raw["model"].(map[string]any)["dropout"] = 0.1
			case "vocab":
				raw["model"].(map[string]any)["vocab"].(map[string]any)["!"] = -1
			case "special":
				raw["added_tokens"].([]any)[0].(map[string]any)["lstrip"] = true
			case "config special":
				raw["added_tokens_decoder"].(map[string]any)["0"].(map[string]any)["content"] = "changed"
			case "template":
				raw["chat_template"] = "unsupported"
			}
			hfWriteJSON(t, filepath.Join(dir, name), raw)
			var cfg struct {
				Vocab   int `json:"vocab_size"`
				Context int `json:"model_max_length"`
			}
			if err := readBoundedJSON(filepath.Join(dir, "tokenizer_config.json"), 1<<20, &cfg); err != nil {
				t.Fatal(err)
			}
			tok, err := loadHFTokenizer(dir, hfConfig{Vocab: cfg.Vocab, Context: cfg.Context, BOS: 1, EOS: 2})
			if kind == "template" {
				if err != nil {
					t.Fatal(err)
				}
				_, err = tok.formatChat([]ChatMessage{{Role: "user", Content: "Hi"}})
			}
			if err == nil {
				t.Fatal("accepted unsupported configuration")
			}
		})
	}
}
