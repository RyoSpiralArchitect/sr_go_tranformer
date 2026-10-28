package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestHFLlama3ChatReference(t *testing.T) {
	testHFTokenizerReference(t, "testdata/hf-llama3-chat", "testdata/hf-llama3-chat/reference.json")
}

func TestLlama3EOSMetadata(t *testing.T) {
	b, err := os.ReadFile("testdata/hf-llama3-chat/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{[]any{}, []any{1, 1}, []any{1, nil}, []any{1, 1.5}, []any{1, -1}, []any{1, 999999}, []any{1, 2, 3, 4, 5, 6, 7, 8, 9}, []any{true}, nil, "2"} {
		var raw map[string]any
		if err := json.Unmarshal(b, &raw); err != nil {
			t.Fatal(err)
		}
		raw["eos_token_id"] = value
		dir := t.TempDir()
		hfWriteJSON(t, filepath.Join(dir, "config.json"), raw)
		if _, _, err := readHFConfig(dir, 64); err == nil {
			t.Fatalf("accepted EOS %v", value)
		}
	}
	h, _, err := readHFConfig("testdata/hf-llama3-chat", 64)
	if err != nil || len(h.ExtraEOS) != 2 {
		t.Fatal(h, err)
	}
	for _, id := range append([]int{h.EOS}, h.ExtraEOS...) {
		if !h.isEOS(id) {
			t.Fatal("lost stop", id)
		}
	}
	tok := testHFTokenizer(t, "testdata/hf-llama3-chat")
	m := hfServingModel(t, tok, -1)
	m.hf.ExtraEOS = nil
	if err := validateInferenceTokenizer(m, tok); err == nil {
		t.Fatal("accepted mismatched stop set")
	}
}

func TestLlama3ChatBoundsAndTemplateIdentity(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-llama3-chat")
	for _, m := range [][]ChatMessage{nil, make([]ChatMessage, 129), {{Role: "tool", Content: "x"}}, {{Role: "user", Content: "\xff"}}, {{Role: "user", Content: strings.Repeat("x", 1<<20)}}} {
		if _, err := tok.formatChat(m); err == nil {
			t.Fatal("accepted invalid chat")
		}
	}
	dir := t.TempDir()
	for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
		b, err := os.ReadFile(filepath.Join("testdata/hf-llama3-chat", name))
		if err != nil {
			t.Fatal(err)
		}
		if name == "tokenizer_config.json" {
			var cfg map[string]any
			json.Unmarshal(b, &cfg)
			cfg["chat_template"] = cfg["chat_template"].(string) + " "
			b, _ = json.Marshal(cfg)
		}
		if err = os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	changed := testHFTokenizer(t, dir)
	if changed.chat {
		t.Fatal("accepted changed template")
	}
	if _, err := changed.encodePrompt(context.Background(), "Hello", 64); err != nil {
		t.Fatal("template change disabled plain text", err)
	}
	if _, err := changed.formatChat([]ChatMessage{{Role: "user", Content: "Hi"}}); err == nil {
		t.Fatal("rendered unrecognized template")
	}
}

func TestLlama3ChatFramingAndAllStops(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-llama3-chat")
	messages := []ChatMessage{{Role: "user", Content: "Hi"}}
	prompt, err := tok.formatChatAt(messages, time.Date(2026, 10, 25, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	s := defaultSampling()
	s.Temperature, s.RepeatPenalty, s.MaxTokens = 0, 1, 3
	for _, winner := range append([]int{tok.eos}, tok.extraEOS...) {
		m := hfServingModel(t, tok, winner)
		for i := range m.Emb.Data {
			m.Emb.Data[i] = 1
		}
		for j := 0; j < m.Config.Dim; j++ {
			m.Emb.Data[winner*m.Config.Dim+j] = 2
		}
		ids, err := prepareInput(context.Background(), m, tok, prompt, s, true)
		if err != nil {
			t.Fatal(err)
		}
		bos := 0
		for _, id := range ids {
			if id == tok.bos {
				bos++
			}
		}
		if bos != 1 {
			t.Fatal("chat BOS count", bos)
		}
		plain, err := preparePrompt(context.Background(), m, tok, prompt, s)
		if err != nil || len(plain) != len(ids)+1 {
			t.Fatal("plain framing", plain, err)
		}
		want, err := generateInput(context.Background(), m, tok, prompt, s, true)
		if err != nil || !reflect.DeepEqual(want.TokenIDs, []int{winner}) || want.FinishReason != "eos" || want.Text != "" {
			t.Fatal(winner, want, err)
		}
		probe, err := inferHFIDs(context.Background(), m, ids, 3, 4, false)
		if err != nil || !reflect.DeepEqual(probe.TokenIDs, want.TokenIDs) || probe.FinishReason != "eos" {
			t.Fatal("ID inference stop", probe, err)
		}
		e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 2, Queue: 2, CacheBytes: 2 << 20, KernelWorkers: 2})
		if err != nil {
			t.Fatal(err)
		}
		stream, err := e.streamInput(context.Background(), prompt, s, true)
		if err != nil {
			e.Close()
			t.Fatal(err)
		}
		var streamed string
		for event := range stream.Events {
			streamed += event.Text
		}
		got, err := stream.Wait()
		<-stream.Released()
		e.Close()
		if err != nil || !reflect.DeepEqual(got, want) || streamed != want.Text {
			t.Fatal("stream stop/framing", got, want, err)
		}
		if e.Stats().AdmissionUsed != 0 || e.Stats().KVReservedBytes != 0 {
			t.Fatal("unreleased chat resources")
		}
	}
}

func TestLlama3HTTPChatFraming(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-llama3-chat")
	winner := tok.extraEOS[1]
	m := hfServingModel(t, tok, winner)
	for i := range m.Emb.Data {
		m.Emb.Data[i] = 1
	}
	for j := 0; j < m.Config.Dim; j++ {
		m.Emb.Data[winner*m.Config.Dim+j] = 2
	}
	e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 2, Queue: 2, CacheBytes: 2 << 20, KernelWorkers: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	server := httptest.NewServer(e.Handler())
	defer server.Close()
	messages := []ChatMessage{{Role: "user", Content: "猫"}}
	s := defaultSampling()
	s.Temperature, s.RepeatPenalty, s.MaxTokens = 0, 1, 3
	want, err := GenerateChat(context.Background(), m, tok, messages, s)
	if err != nil {
		t.Fatal(err)
	}
	for _, stream := range []bool{false, true} {
		body, _ := json.Marshal(map[string]any{"messages": messages, "stream": stream, "temperature": 0, "repeat_penalty": 1, "max_tokens": 3})
		resp, err := http.Post(server.URL+"/v1/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil || resp.StatusCode != 200 {
			t.Fatal(string(b), err)
		}
		var got Completion
		if !stream {
			if err = json.Unmarshal(b, &got); err != nil {
				t.Fatal(err)
			}
		} else {
			found := false
			for _, event := range strings.Split(string(b), "\n\n") {
				if strings.HasPrefix(event, "event: done\ndata: ") {
					found = true
					if err = json.Unmarshal([]byte(strings.TrimPrefix(event, "event: done\ndata: ")), &got); err != nil {
						t.Fatal(err)
					}
				}
			}
			if !found {
				t.Fatal("missing SSE completion", string(b))
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatal("HTTP framing/stop", got, want)
		}
	}
}

// Exercise the actual flag dispatch, stdin framing and interactive reset. The
// tiny zero-weight fixture intentionally predicts the same token at each step.
func TestLlama3CLIChatFraming(t *testing.T) {
	if args := os.Getenv("MONOLITH_TEST_LLAMA_CHAT_ARGS"); args != "" {
		var cli []string
		if err := json.Unmarshal([]byte(args), &cli); err != nil {
			t.Fatal(err)
		}
		os.Args = append([]string{"monolith"}, cli...)
		main()
		os.Exit(0)
	}
	testHFChatCLIFraming(t, "testdata/hf-llama3-chat")
}
func TestMistralCLIChatFraming(t *testing.T) {
	testHFChatCLIFraming(t, "testdata/hf-mistral-tokenizer")
}
func testHFChatCLIFraming(t *testing.T, fixture string) {
	dir := t.TempDir()
	for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
		b, err := os.ReadFile(filepath.Join(fixture, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	h, c, err := readHFConfig(dir, 512)
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
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	runCLI := func(input string, args ...string) []byte {
		t.Helper()
		args = append(args, "-hf-model", dir, "-context", "512")
		b, err := json.Marshal(args)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "-test.run=^TestLlama3CLIChatFraming$")
		cmd.Env = append(os.Environ(), "MONOLITH_TEST_LLAMA_CHAT_ARGS="+string(b))
		cmd.Stdin = strings.NewReader(input)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("CLI %v: %v: %s", args, err, stderr.String())
		}
		return out
	}
	var encoded struct {
		Prompt string `json:"prompt"`
		IDs    []int  `json:"token_ids"`
	}
	if err := json.Unmarshal(runCLI("", "tokenize-hf", "-chat", "-prompt", "Hi"), &encoded); err != nil {
		t.Fatal(err)
	}
	var want Completion
	for i, source := range []struct {
		args  []string
		input string
	}{
		{[]string{"-chat", "-prompt", "Hi"}, ""},
		{[]string{"-chat", "-prompt-file", "-"}, "Hi"},
		{[]string{"-messages", "-"}, `[{"role":"user","content":"Hi"}]`},
	} {
		args := append([]string{"generate", "-max-tokens", "3", "-temperature", "0", "-repeat-penalty", "1", "-json"}, source.args...)
		var got Completion
		if err := json.Unmarshal(runCLI(source.input, args...), &got); err != nil {
			t.Fatal(err)
		}
		if got.PromptTokens != len(encoded.IDs) || got.CompletionTokens != 3 {
			t.Fatal("CLI chat framing", got, encoded.IDs)
		}
		if i == 0 {
			want = got
		} else if !reflect.DeepEqual(got, want) {
			t.Fatal("stdin chat mismatch", got, want)
		}
	}
	var plain Completion
	if err := json.Unmarshal(runCLI(encoded.Prompt, "generate", "-prompt-file", "-", "-max-tokens", "3", "-temperature", "0", "-repeat-penalty", "1", "-json"), &plain); err != nil {
		t.Fatal(err)
	}
	if plain.PromptTokens != want.PromptTokens+1 {
		t.Fatal("plain input lost automatic BOS", plain, want)
	}
	interactive := runCLI("Hi\n/reset\nHi\n/exit\n", "chat", "-max-tokens", "3", "-temperature", "0", "-repeat-penalty", "1", "-kernel-workers", "2")
	if string(interactive) != strings.Repeat(want.Text+"\n", 2) {
		t.Fatalf("interactive reset mismatch: %q", interactive)
	}
}
