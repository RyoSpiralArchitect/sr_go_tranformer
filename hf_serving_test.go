package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func hfServingModel(t *testing.T, tok *hfTokenizer, winner int) *Model {
	t.Helper()
	c, err := preset("demo")
	if err != nil {
		t.Fatal(err)
	}
	c.Vocab = tok.Vocab()
	c.Context = 512
	m, err := NewModel(c, 4)
	if err != nil {
		t.Fatal(err)
	}
	m.hf = &hfConfig{BOS: tok.bos, EOS: tok.eos}
	if winner < 0 {
		return m
	}
	for _, p := range m.Params {
		clear(p.Data)
		if !p.Decay {
			for i := range p.Data {
				p.Data[i] = 1
			}
		}
	}
	id := tok.bytes['x']
	for j := 0; j < c.Dim; j++ {
		m.Emb.Data[id*c.Dim+j] = 1
		m.Emb.Data[winner*c.Dim+j] = 2
	}
	return m
}
func TestHFGenerationSpecialIDsAndStreaming(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	for _, winner := range []int{2, 256} {
		m := hfServingModel(t, tok, winner)
		s := defaultSampling()
		s.Temperature = 0
		s.RepeatPenalty = 1
		s.MaxTokens = 3
		want, err := Generate(context.Background(), m, tok, "x", s)
		if err != nil {
			t.Fatal(err)
		}
		if want.TokenIDs[0] != winner {
			t.Fatal("masked valid HF ID", want)
		}
		if winner == 2 && (want.FinishReason != "eos" || len(want.TokenIDs) != 1 || want.Text != "") {
			t.Fatal("wrong model EOS behavior", want)
		}
		e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 2 << 20, KernelWorkers: 2})
		if err != nil {
			t.Fatal(err)
		}
		stream, err := e.Stream(context.Background(), "x", s)
		if err != nil {
			t.Fatal(err)
		}
		var text string
		var ids []int
		for event := range stream.Events {
			text += event.Text
			ids = append(ids, event.TokenID)
		}
		got, err := stream.Wait()
		if err != nil {
			t.Fatal(err)
		}
		<-stream.Released()
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(ids, want.TokenIDs) || text != want.Text {
			t.Fatal("HF streaming/completion mismatch", got, want, text)
		}
		stats := e.Stats()
		if stats.AdmissionUsed != 0 || stats.KVReservedBytes != 0 || stats.Computing != 0 {
			t.Fatal("unreleased HF resources", stats)
		}
		e.Close()
		if e.Stats().KernelWorkers != 0 {
			t.Fatal("workers retained")
		}
	}
}
func TestHFInferenceProfileValidation(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	m := hfServingModel(t, tok, -1)
	native := Tokenizer{Merges: make([]Pair, tok.Vocab()-258)}
	if _, err := Generate(context.Background(), m, native, "x", defaultSampling()); err == nil {
		t.Fatal("accepted native tokenizer for HF")
	}
	m.hf = nil
	if _, err := Generate(context.Background(), m, tok, "x", defaultSampling()); err == nil {
		t.Fatal("accepted HF tokenizer for native")
	}
}
func TestHFHTTPMessagesAndStream(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	m := hfServingModel(t, tok, -1)
	s := defaultSampling()
	s.Temperature = 0
	s.RepeatPenalty = 1
	s.MaxTokens = 4
	messages := []ChatMessage{{Role: "user", Content: "Hello, 猫!"}}
	prompt, err := tok.formatChat(messages)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Generate(context.Background(), m, tok, prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 2 << 20, KernelWorkers: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	server := httptest.NewServer(e.Handler())
	defer server.Close()
	for _, stream := range []bool{false, true} {
		body, _ := json.Marshal(map[string]any{"messages": messages, "max_tokens": s.MaxTokens, "temperature": 0, "repeat_penalty": 1, "stream": stream})
		resp, err := http.Post(server.URL+"/v1/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 200 {
			t.Fatalf("%d %s", resp.StatusCode, b)
		}
		var got Completion
		if !stream {
			if err = json.Unmarshal(b, &got); err != nil {
				t.Fatal(err)
			}
		} else {
			var text string
			found := false
			for _, event := range strings.Split(string(b), "\n\n") {
				if strings.HasPrefix(event, "event: token\ndata: ") {
					var token TokenEvent
					if err = json.Unmarshal([]byte(strings.TrimPrefix(event, "event: token\ndata: ")), &token); err != nil {
						t.Fatal(err)
					}
					text += token.Text
				}
				if strings.HasPrefix(event, "event: done\ndata: ") {
					if err = json.Unmarshal([]byte(strings.TrimPrefix(event, "event: done\ndata: ")), &got); err != nil {
						t.Fatal(err)
					}
					found = true
				}
			}
			if !found || text != want.Text {
				t.Fatal("missing terminal or inconsistent SSE text", string(b))
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatal(got, want)
		}
	}
	for _, body := range []string{
		`{"messages":[]}`,
		`{"messages":[{"role":"user","content":"hi"}],"prompt":"x"}`,
		`{"messages":[{"role":"tool","content":"hi"}]}`,
		`{"prompt":""}`,
	} {
		resp, err := http.Post(server.URL+"/v1/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 400 {
			t.Fatalf("expected 400, got %d for %s", resp.StatusCode, body)
		}
	}
}
