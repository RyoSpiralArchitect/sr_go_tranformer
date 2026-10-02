// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestIncrementalTextMatchesWholeDecoding(t *testing.T) {
	cases := [][]byte{
		{}, []byte("cat 猫 🌀 日本語"),
		{'a', 0xff, 0xfe, 'b'},
		{0xe7, 0x8c}, // An incomplete final rune is one invalid run.
		{0xff, 0xe7, 0x8c, 0xab, 0xfe},
		{0xed, 0xa0, 0x80, 0xc0, 0xaf, 'x', 0x80, 0x81},
		{0xe7, 0xff, 0x8c, 0xab, 'x', 0xf0, 0x9f},
	}
	allBytes := make([]byte, 256)
	for i := range allBytes {
		allBytes[i] = byte(i)
	}
	cases = append(cases, allBytes)
	for caseID, raw := range cases {
		want := strings.ToValidUTF8(string(raw), "\ufffd")
		// Every two-way split includes splits within valid runes and invalid runs.
		for split := 0; split <= len(raw); split++ {
			var d incrementalText
			first := d.Push(raw[:split], false)
			last := d.Push(raw[split:], true)
			if !utf8.ValidString(first) || !utf8.ValidString(last) {
				t.Fatalf("case %d split %d emitted invalid UTF-8", caseID, split)
			}
			if got := first + last; got != want {
				t.Fatalf("case %d split %d: got %q, want %q", caseID, split, got, want)
			}
			if got := d.Push(nil, true); got != "" {
				t.Fatalf("case %d repeated finalization emitted %q", caseID, got)
			}
		}
		var d incrementalText
		var got strings.Builder
		for _, b := range raw {
			chunk := d.Push([]byte{b}, false)
			if !utf8.ValidString(chunk) {
				t.Fatalf("case %d byte-at-a-time emitted invalid UTF-8", caseID)
			}
			got.WriteString(chunk)
		}
		got.WriteString(d.Push(nil, true))
		if got.String() != want {
			t.Fatalf("case %d byte-at-a-time: got %q, want %q", caseID, got.String(), want)
		}
	}
}

func streamTestContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func collectTestStream(t *testing.T, ctx context.Context, stream *CompletionStream) ([]TokenEvent, Completion, error) {
	t.Helper()
	var events []TokenEvent
	for {
		select {
		case event, open := <-stream.Events:
			if !open {
				out, err := stream.Wait()
				return events, out, err
			}
			events = append(events, event)
		case <-ctx.Done():
			stream.Close()
			t.Fatal("stream did not terminate before its context expired")
		}
	}
}

func assertStreamParity(t *testing.T, events []TokenEvent, got, want Completion) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream completion %+v != independent completion %+v", got, want)
	}
	var text strings.Builder
	ids := make([]int, 0, len(events))
	for _, event := range events {
		if !utf8.ValidString(event.Text) {
			t.Fatalf("event emitted invalid UTF-8: %q", event.Text)
		}
		ids = append(ids, event.TokenID)
		text.WriteString(event.Text)
	}
	if !reflect.DeepEqual(ids, want.TokenIDs) || text.String() != want.Text {
		t.Fatalf("events differ from completion: IDs %v / %v, text %q / %q", ids, want.TokenIDs, text.String(), want.Text)
	}
}

func TestEngineStreamMatchesIndependentGeneration(t *testing.T) {
	m := testModel(t)
	tok := Tokenizer{}
	e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 4, Queue: 8, CacheBytes: 1 << 20, StreamBuffer: 64, PrefillChunk: 7, TokenBudget: 16})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ctx := streamTestContext(t)
	prompts := []string{"a", "猫", strings.Repeat("x", 47), "mixed prompt"}
	streams := make([]*CompletionStream, len(prompts))
	wants := make([]Completion, len(prompts))
	for i, prompt := range prompts {
		s := defaultSampling()
		s.MaxTokens = 32
		s.Seed = uint64(i + 1)
		wants[i], err = Generate(ctx, m, tok, prompt, s)
		if err != nil {
			t.Fatal(err)
		}
		streams[i], err = e.Stream(ctx, prompt, s)
		if err != nil {
			t.Fatal(err)
		}
		defer streams[i].Close()
	}
	for i, stream := range streams {
		events, got, err := collectTestStream(t, ctx, stream)
		if err != nil {
			t.Fatalf("stream %d: %v", i, err)
		}
		assertStreamParity(t, events, got, wants[i])
	}
	await(t, "stream reservations released", func() bool {
		s := e.Stats()
		return s.Active == 0 && s.KVReservedBytes == 0 && len(e.admission) == 0
	})
}

func zeroStreamModel(t *testing.T, contextSize int) *Model {
	t.Helper()
	c := testModel(t).Config
	c.Context = contextSize
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	// Equal logits and greedy sampling always select byte 0, never EOS.
	for _, p := range m.Params {
		clear(p.Data)
	}
	return m
}

func TestUnconsumedStreamIsBoundedAndDoesNotBlockEngine(t *testing.T) {
	m := zeroStreamModel(t, 128)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 1 << 20, StreamBuffer: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ctx := streamTestContext(t)
	s := defaultSampling()
	s.MaxTokens, s.Temperature = 32, 0
	stream, err := e.Stream(ctx, "slow", s)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	// Deliberately never receive Events until the actor has terminated this
	// session. An accidentally blocking send also stalls this healthy request.
	got, err := e.Generate(ctx, "healthy", s)
	if err != nil || got.CompletionTokens != s.MaxTokens {
		t.Fatalf("slow client stalled healthy generation: %+v, %v", got, err)
	}
	await(t, "slow consumer terminated", func() bool { return e.Stats().Failed == 1 && e.Stats().Active == 0 })
	if _, err = stream.Wait(); !errors.Is(err, ErrSlowConsumer) {
		t.Fatalf("slow consumer terminal error: %v", err)
	}
	count := 0
	for range stream.Events {
		count++
	}
	if count > e.Config.StreamBuffer {
		t.Fatalf("buffer exceeded capacity: %d > %d", count, e.Config.StreamBuffer)
	}
	if stats := e.Stats(); stats.KVReservedBytes != 0 || len(e.admission) != 0 || stats.Completed != 1 {
		t.Fatalf("slow client leaked capacity: %+v, admission %d", stats, len(e.admission))
	}
}

func TestEngineStreamCancellationAndShutdown(t *testing.T) {
	for _, mode := range []string{"context", "stream-close", "engine-close"} {
		t.Run(mode, func(t *testing.T) {
			m := zeroStreamModel(t, 8192)
			e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 4, CacheBytes: 2 << 20, StreamBuffer: 64, PrefillChunk: 1, TokenBudget: 1})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			ctx, cancel := context.WithCancel(streamTestContext(t))
			defer cancel()
			s := defaultSampling()
			s.MaxTokens, s.Temperature = 32, 0
			stream, err := e.Stream(ctx, strings.Repeat("x", 4096), s)
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			await(t, "stream owns KV reservation", func() bool { return e.Stats().Active == 1 && e.Stats().KVReservedBytes > 0 })
			var queued *CompletionStream
			if mode == "engine-close" {
				queued, err = e.Stream(ctx, "queued", s)
				if err != nil {
					t.Fatal(err)
				}
				defer queued.Close()
			}
			wantErr := error(context.Canceled)
			switch mode {
			case "context":
				cancel()
			case "stream-close":
				stream.Close()
			case "engine-close":
				e.Close()
				wantErr = ErrClosed
			}
			_, _, err = collectTestStream(t, streamTestContext(t), stream)
			if !errors.Is(err, wantErr) {
				t.Fatalf("terminal error %v, want %v", err, wantErr)
			}
			if queued != nil {
				_, _, err = collectTestStream(t, streamTestContext(t), queued)
				if !errors.Is(err, ErrClosed) {
					t.Fatalf("queued stream survived shutdown: %v", err)
				}
			}
			await(t, "all stream ownership released", func() bool {
				stats := e.Stats()
				return stats.Active == 0 && stats.KVReservedBytes == 0 && len(e.admission) == 0 && stats.Queue == 0
			})
		})
	}
}

// readTestSSEFrame consumes one SSE frame, keeping the network response open.
func readTestSSEFrame(reader *bufio.Reader) (string, []byte, error) {
	var event string
	var data []byte
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return "", nil, err
		}
		line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
		if line == "" {
			if event != "" || data != nil {
				return event, data, nil
			}
			continue
		}
		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		}
		if strings.HasPrefix(line, "data:") {
			if data != nil {
				data = append(data, '\n')
			}
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " ")...)
		}
	}
}

func TestHTTPStreamingMatchesIndependentGeneration(t *testing.T) {
	m := testModel(t)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 1 << 20, StreamBuffer: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	server := httptest.NewServer(e.Handler())
	defer server.Close()
	ctx := streamTestContext(t)
	s := defaultSampling()
	s.MaxTokens, s.Seed = 24, 13
	prompt := "猫 and spiral"
	want, err := Generate(ctx, m, Tokenizer{}, prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(struct {
		Prompt string `json:"prompt"`
		Stream bool   `json:"stream"`
		Sampling
	}{prompt, true, s})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("stream response %d %q: %s", resp.StatusCode, resp.Header.Get("Content-Type"), raw)
	}
	reader := bufio.NewReader(resp.Body)
	var events []TokenEvent
	var got Completion
	for {
		event, data, err := readTestSSEFrame(reader)
		if err != nil {
			t.Fatalf("stream ended before done event: %v", err)
		}
		switch event {
		case "token":
			var token TokenEvent
			if err := json.Unmarshal(data, &token); err != nil {
				t.Fatal(err)
			}
			events = append(events, token)
		case "done":
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			assertStreamParity(t, events, got, want)
			if extra, err := io.ReadAll(reader); err != nil || len(bytes.TrimSpace(extra)) != 0 {
				t.Fatalf("unexpected data after terminal done: %q, %v", extra, err)
			}
			return
		default:
			t.Fatalf("unexpected SSE event %q: %s", event, data)
		}
	}
}

func TestHTTPStreamDisconnectReleasesCapacity(t *testing.T) {
	m := zeroStreamModel(t, 8192)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 2, CacheBytes: 4 << 20, StreamBuffer: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	server := httptest.NewServer(e.Handler())
	defer server.Close()
	ctx := streamTestContext(t)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/completions", strings.NewReader(`{"prompt":"a","stream":true,"temperature":0,"max_tokens":4096}`))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("status %d", resp.StatusCode)
	}
	event, _, err := readTestSSEFrame(bufio.NewReader(resp.Body))
	if err != nil || event != "token" {
		resp.Body.Close()
		t.Fatalf("first streamed event %q, %v", event, err)
	}
	// Close the actual socket-backed response body while generation is active.
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	await(t, "disconnected HTTP request released", func() bool {
		s := e.Stats()
		return s.Active == 0 && s.KVReservedBytes == 0 && len(e.admission) == 0 && len(e.httpSlots) == 0
	})
	if s := e.Stats(); s.Canceled != 1 || s.Completed != 0 {
		t.Fatalf("disconnect did not cancel generation: %+v", s)
	}
}

func TestHTTPStreamingErrorsBeforeHeaders(t *testing.T) {
	m := testModel(t)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 2, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	for _, tc := range []struct {
		name   string
		body   string
		status int
	}{
		{"malformed", `{"stream":true,`, http.StatusBadRequest},
		{"stream-type", `{"stream":"yes"}`, http.StatusBadRequest},
		{"unknown-field", `{"stream":true,"unknown":1}`, http.StatusBadRequest},
		{"sampling", `{"stream":true,"top_p":0}`, http.StatusBadRequest},
		{"context-limit", `{"stream":true,"max_tokens":128,"prompt":"x"}`, http.StatusBadRequest},
		{"body-limit", `{"stream":true,"prompt":"` + strings.Repeat("x", 1<<20) + `"}`, http.StatusRequestEntityTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertStreamingHTTPError(t, e, tc.body, tc.status)
		})
	}
	t.Run("admission-full", func(t *testing.T) {
		for i := 0; i < cap(e.admission); i++ {
			e.admission <- struct{}{}
		}
		defer func() {
			for len(e.admission) > 0 {
				<-e.admission
			}
		}()
		assertStreamingHTTPError(t, e, `{"stream":true,"max_tokens":8}`, http.StatusTooManyRequests)
	})
	t.Run("budget", func(t *testing.T) {
		pageBytes := int64(2 * pageTokens * m.Config.KVDim() * m.Config.Layers * 4)
		limited, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: pageBytes})
		if err != nil {
			t.Fatal(err)
		}
		defer limited.Close()
		assertStreamingHTTPError(t, limited, fmt.Sprintf(`{"stream":true,"max_tokens":%d}`, pageTokens+1), http.StatusBadRequest)
	})
	e.Close()
	assertStreamingHTTPError(t, e, `{"stream":true,"max_tokens":8}`, http.StatusServiceUnavailable)
}

func assertStreamingHTTPError(t *testing.T, e *Engine, body string, status int) {
	t.Helper()
	w := httptest.NewRecorder()
	e.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body)))
	if w.Code != status || !strings.HasPrefix(w.Header().Get("Content-Type"), "application/json") {
		t.Fatalf("got status %d, type %q, want JSON status %d: %s", w.Code, w.Header().Get("Content-Type"), status, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil || payload["error"] == nil {
		t.Fatalf("invalid error payload %s: %v", w.Body.String(), err)
	}
}

func TestHTTPStreamingReportsAsynchronousCapacityError(t *testing.T) {
	m := zeroStreamModel(t, 8192)
	s := defaultSampling()
	s.MaxTokens = 32
	prompt := strings.Repeat("x", 4096)
	pages := (len(prompt) + 1 + s.MaxTokens + pageTokens - 1) / pageTokens
	pageBytes := int64(2 * pageTokens * m.Config.KVDim() * m.Config.Layers * 4)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: int64(pages) * pageBytes, PrefillChunk: 1, TokenBudget: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ctx := streamTestContext(t)
	blocker, err := e.Stream(ctx, prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	defer blocker.Close()
	await(t, "first request reserves full cache budget", func() bool { return e.Stats().KVReservedBytes == e.Config.CacheBytes })
	server := httptest.NewServer(e.Handler())
	defer server.Close()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/completions", strings.NewReader(`{"stream":true,"max_tokens":8}`))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("expected admitted SSE stream, got %d %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	reader := bufio.NewReader(resp.Body)
	event, data, err := readTestSSEFrame(reader)
	if err != nil || event != "error" {
		t.Fatalf("expected terminal SSE error, got %q %s (%v)", event, data, err)
	}
	var payload struct {
		Error string `json:"error"`
		Code  string `json:"code"`
	}
	if err = json.Unmarshal(data, &payload); err != nil || payload.Error != ErrBusy.Error() || payload.Code == "" {
		t.Fatalf("invalid SSE error payload %s: %v", data, err)
	}
	if extra, err := io.ReadAll(reader); err != nil || len(bytes.TrimSpace(extra)) != 0 {
		t.Fatalf("unexpected events after terminal error: %q, %v", extra, err)
	}
}

type deadlineTestWriter struct {
	header   http.Header
	deadline time.Time
	entered  chan time.Time
	unblock  <-chan struct{}
}

func (w *deadlineTestWriter) Header() http.Header { return w.header }
func (w *deadlineTestWriter) WriteHeader(int)     {}
func (w *deadlineTestWriter) Flush()              {}
func (w *deadlineTestWriter) SetWriteDeadline(d time.Time) error {
	w.deadline = d
	return nil
}
func (w *deadlineTestWriter) Write(p []byte) (int, error) {
	w.entered <- w.deadline
	<-w.unblock
	return 0, context.DeadlineExceeded
}

func TestHTTPStreamWriteDeadlineFailureCancelsGeneration(t *testing.T) {
	m := zeroStreamModel(t, 8192)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 2, CacheBytes: 2 << 20, PrefillChunk: 1, TokenBudget: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ctx := streamTestContext(t)
	unblock := make(chan struct{})
	defer func() {
		select {
		case <-unblock:
		default:
			close(unblock)
		}
	}()
	w := &deadlineTestWriter{header: make(http.Header), entered: make(chan time.Time, 1), unblock: unblock}
	body := `{"prompt":"` + strings.Repeat("x", 4096) + `","stream":true,"max_tokens":32}`
	r := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body)).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		e.Handler().ServeHTTP(w, r)
	}()
	select {
	case deadline := <-w.entered:
		if deadline.IsZero() {
			t.Fatal("streaming write has no deadline")
		}
	case <-ctx.Done():
		t.Fatal("HTTP stream never reached its writer")
	}
	await(t, "blocked writer's active request", func() bool { return e.Stats().Active == 1 })
	// Release the writer with the same error a timed-out network write returns.
	// The gate makes this deterministic without waiting for a real five-second
	// write deadline or relying on a platform-specific socket buffer size.
	close(unblock)
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("handler did not return after write timeout")
	}
	if !w.deadline.IsZero() {
		t.Fatal("failed write left a deadline armed")
	}
	await(t, "write failure releases engine and HTTP admission", func() bool {
		s := e.Stats()
		return s.Active == 0 && s.KVReservedBytes == 0 && len(e.admission) == 0 && len(e.httpSlots) == 0
	})
	if s := e.Stats(); s.Canceled != 1 {
		t.Fatalf("write timeout did not cancel generation: %+v", s)
	}
}

type streamDeadlineAction struct {
	kind     string
	deadline time.Time
}

type streamDeadlineRecorder struct {
	header   http.Header
	deadline time.Time
	actions  chan streamDeadlineAction
	flushErr error
}

func (w *streamDeadlineRecorder) Header() http.Header { return w.header }
func (w *streamDeadlineRecorder) WriteHeader(int)     {}
func (w *streamDeadlineRecorder) Flush()              { _ = w.FlushError() }
func (w *streamDeadlineRecorder) SetWriteDeadline(d time.Time) error {
	w.deadline = d
	w.actions <- streamDeadlineAction{"deadline", d}
	return nil
}
func (w *streamDeadlineRecorder) Write(p []byte) (int, error) {
	w.actions <- streamDeadlineAction{"write", w.deadline}
	return len(p), nil
}
func (w *streamDeadlineRecorder) FlushError() error {
	w.actions <- streamDeadlineAction{"flush", w.deadline}
	return w.flushErr
}

func TestHTTPStreamClearsDeadlineWhileWaiting(t *testing.T) {
	for _, flushFails := range []bool{false, true} {
		t.Run(fmt.Sprintf("flush-error=%t", flushFails), func(t *testing.T) {
			// Hold an admitted request in the mailbox without running the actor.
			// This models arbitrarily long queue/prefill time without a wall-clock
			// delay, and guarantees that no token can arrive during the assertion.
			e := &Engine{
				Model: testModel(t), Tokenizer: Tokenizer{},
				Config: EngineConfig{StreamBuffer: 1},
				queue:  make(chan *engineRequest, 1), admission: make(chan struct{}, 1),
				done: make(chan struct{}), pool: pagePool{Limit: 100},
			}
			ctx, cancel := context.WithCancel(streamTestContext(t))
			defer cancel()
			w := &streamDeadlineRecorder{header: make(http.Header), actions: make(chan streamDeadlineAction, 8)}
			if flushFails {
				w.flushErr = context.DeadlineExceeded
			}
			r := httptest.NewRequest(http.MethodPost, "/v1/completions", nil).WithContext(ctx)
			done := make(chan struct{})
			go func() {
				defer close(done)
				e.serveStream(w, r, ctx, "pending", defaultSampling())
			}()
			var request *engineRequest
			select {
			case request = <-e.queue:
				defer request.release()
				defer request.Cancel()
			case <-ctx.Done():
				t.Fatal("stream request was not admitted")
			}
			for i, kind := range []string{"deadline", "write", "flush", "deadline"} {
				select {
				case action := <-w.actions:
					if action.kind != kind {
						t.Fatalf("deadline operation %d: got %q, want %q", i, action.kind, kind)
					}
					if shouldBeClear := i == 3; action.deadline.IsZero() != shouldBeClear {
						t.Fatalf("operation %d (%s): zero deadline=%t, want %t", i, kind, action.deadline.IsZero(), shouldBeClear)
					}
				case <-ctx.Done():
					t.Fatalf("missing deadline operation %d (%s)", i, kind)
				}
			}
			// The successful heartbeat is flushed, but the request is still
			// pending: an HTTP/2 idle stream must now have no write timer armed.
			if !flushFails {
				select {
				case <-done:
					t.Fatal("healthy pending stream returned prematurely")
				default:
				}
			}
			cancel()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("handler did not finish after controlled cancellation")
			}
			if !w.deadline.IsZero() {
				t.Fatal("handler left an outstanding write deadline")
			}
		})
	}
}
