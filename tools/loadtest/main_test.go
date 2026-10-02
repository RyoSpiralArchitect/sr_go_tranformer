// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testOptions(endpoint string) options {
	return options{URL: endpoint, Requests: 12, Concurrency: 3, ShortPrompt: "x", LongPrompt: "longer prompt", MaxTokens: 4, CancelAfter: 2, SlowDelay: time.Millisecond, Timeout: 2 * time.Second, SampleInterval: time.Millisecond}
}

func emit(w http.ResponseWriter, event string, data any) {
	b, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func TestReadEvents(t *testing.T) {
	input := ": keepalive\r\nevent: token\r\ndata: {\r\ndata: \"token_id\": 3, \"text\": \"猫\"}\r\n\r\n"
	count := 0
	err := readEvents(strings.NewReader(input), func(event string, data []byte) error {
		count++
		var token tokenEvent
		if event != "token" || json.Unmarshal(data, &token) != nil || token.TokenID == nil || *token.TokenID != 3 || token.Text != "猫" {
			t.Errorf("unexpected event %q %s", event, data)
		}
		return nil
	})
	if err != nil || count != 1 {
		t.Fatalf("readEvents: count=%d err=%v", count, err)
	}
	if err := readEvents(strings.NewReader("event: token\ndata: {}"), func(string, []byte) error { return nil }); err == nil {
		t.Fatal("accepted incomplete event")
	}
	if err := readEvents(strings.NewReader("data: "+strings.Repeat("x", maxEventBytes)+"\n\n"), func(string, []byte) error { return nil }); err == nil {
		t.Fatal("accepted oversized event")
	}
}

func TestExecuteRequestChecksProtocolAndUTF8(t *testing.T) {
	tests := []struct {
		name    string
		serve   func(http.ResponseWriter)
		outcome string
	}{
		{"utf8", func(w http.ResponseWriter) {
			emit(w, "token", map[string]any{"token_id": 1, "text": ""})
			emit(w, "token", map[string]any{"token_id": 2, "text": "猫"})
			emit(w, "done", completion{Text: "猫", TokenIDs: []int{1, 2}, PromptTokens: 1, CompletionTokens: 2, FinishReason: "length"})
		}, "completed"},
		{"wrong_text", func(w http.ResponseWriter) {
			emit(w, "token", map[string]any{"token_id": 1, "text": "a"})
			emit(w, "done", completion{Text: "b", TokenIDs: []int{1}, PromptTokens: 1, CompletionTokens: 1, FinishReason: "length"})
		}, "protocol_error"},
		{"wrong_ids", func(w http.ResponseWriter) {
			emit(w, "token", map[string]any{"token_id": 1, "text": "a"})
			emit(w, "done", completion{Text: "a", TokenIDs: []int{2}, PromptTokens: 1, CompletionTokens: 1, FinishReason: "length"})
		}, "protocol_error"},
		{"missing_done", func(w http.ResponseWriter) {
			emit(w, "token", map[string]any{"token_id": 1, "text": "a"})
		}, "protocol_error"},
		{"event_after_done", func(w http.ResponseWriter) {
			emit(w, "done", completion{TokenIDs: []int{}, PromptTokens: 1, FinishReason: "eos"})
			emit(w, "token", map[string]any{"token_id": 1, "text": "a"})
		}, "protocol_error"},
		{"missing_token_id", func(w http.ResponseWriter) {
			emit(w, "token", map[string]any{"text": "a"})
		}, "protocol_error"},
		{"server_error", func(w http.ResponseWriter) {
			emit(w, "error", map[string]string{"error": "reader is slow", "code": "slow_consumer"})
		}, "server_error"},
		{"overloaded", func(w http.ResponseWriter) {
			emit(w, "error", map[string]string{"error": "queue full", "code": "busy"})
		}, "overloaded"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				tc.serve(w)
			}))
			defer srv.Close()
			o := testOptions(srv.URL)
			got := executeRequest(context.Background(), srv.Client(), o, planRequest(o, 1))
			if got.Outcome != tc.outcome {
				t.Fatalf("outcome=%s want=%s error=%s", got.Outcome, tc.outcome, got.Error)
			}
			if tc.name == "utf8" && (got.TokenEvents != 2 || got.TTFTMillis == nil || len(got.gaps) != 1) {
				t.Fatalf("unexpected timings: %+v", got)
			}
		})
	}
}

func TestTTFTWaitsForTokenNotHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		time.Sleep(25 * time.Millisecond)
		emit(w, "token", map[string]any{"token_id": 1, "text": ""})
		emit(w, "done", completion{TokenIDs: []int{1}, PromptTokens: 1, CompletionTokens: 1, FinishReason: "eos"})
	}))
	defer srv.Close()
	o := testOptions(srv.URL)
	got := executeRequest(context.Background(), srv.Client(), o, planRequest(o, 1))
	if got.Outcome != "completed" || got.TTFTMillis == nil || *got.TTFTMillis < 20 {
		t.Fatalf("TTFT measured before first token: %+v", got)
	}
}

func TestPlannedCancelClosesRequest(t *testing.T) {
	disconnected := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		emit(w, "token", map[string]any{"token_id": 1, "text": "a"})
		emit(w, "token", map[string]any{"token_id": 2, "text": "b"})
		<-r.Context().Done()
		close(disconnected)
	}))
	defer srv.Close()
	o := testOptions(srv.URL)
	o.CancelEvery = 1
	got := executeRequest(context.Background(), srv.Client(), o, planRequest(o, 1))
	if got.Outcome != "canceled" || got.TokenEvents != 2 {
		t.Fatalf("unexpected cancellation: %+v", got)
	}
	select {
	case <-disconnected:
	case <-time.After(time.Second):
		t.Fatal("server did not observe cancellation")
	}
}

func TestTimeoutAndHTTPOutcomes(t *testing.T) {
	for _, code := range []int{400, 429, 503} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(code) }))
		o := testOptions(srv.URL)
		got := executeRequest(context.Background(), srv.Client(), o, planRequest(o, 1))
		srv.Close()
		if got.Outcome != classifyHTTP(code) || got.HTTPStatus != code {
			t.Fatalf("unexpected HTTP classification: %+v", got)
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	o := testOptions(srv.URL)
	o.Timeout = 15 * time.Millisecond
	got := executeRequest(context.Background(), srv.Client(), o, planRequest(o, 1))
	if got.Outcome != "timeout" {
		t.Fatalf("unexpected timeout: %+v", got)
	}
}

func TestMixedRunBoundedConcurrencyAndSeparateGroups(t *testing.T) {
	var active, peak, completed atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			fmt.Fprintf(w, "monolith_active %d\nmonolith_heap_alloc_bytes 1000\nmonolith_completed_total %d\nmonolith_gc_cycles_total 2\nmonolith_gc_pause_seconds_total 0.1\n", active.Load(), completed.Load())
			return
		}
		n := active.Add(1)
		defer active.Add(-1)
		for old := peak.Load(); n > old; old = peak.Load() {
			if peak.CompareAndSwap(old, n) {
				break
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		var ids []int
		for i := 0; i < 4; i++ {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(time.Millisecond):
			}
			ids = append(ids, i)
			emit(w, "token", map[string]any{"token_id": i, "text": "a"})
		}
		emit(w, "done", completion{Text: "aaaa", TokenIDs: ids, PromptTokens: 1, CompletionTokens: 4, FinishReason: "length"})
		completed.Add(1)
	}))
	defer srv.Close()
	o := testOptions(srv.URL)
	o.SlowEvery, o.CancelEvery, o.Details = 3, 5, true
	got := run(context.Background(), o)
	if got.Started != 12 || got.Outcomes["completed"] != 10 || got.Outcomes["canceled"] != 2 || len(got.Results) != 12 {
		t.Fatalf("unexpected outcomes: %+v", got.Outcomes)
	}
	// Canceled handlers may overlap the replacement request briefly while the
	// server observes connection closure; exercise the strict worker bound again
	// without cancellations before asserting server-side overlap.
	o.CancelEvery = 0
	peak.Store(0)
	deadline := time.Now().Add(time.Second)
	for active.Load() != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	bounded := run(context.Background(), o)
	if peak.Load() > int64(o.Concurrency) || bounded.Outcomes["completed"] != o.Requests {
		t.Fatalf("concurrency=%d limit=%d outcomes=%v", peak.Load(), o.Concurrency, bounded.Outcomes)
	}
	if got.Metrics.Samples < 2 || got.Metrics.Errors != 0 || got.Metrics.SampledPeaks["monolith_heap_alloc_bytes"] != 1000 || got.Metrics.CounterDeltas["monolith_gc_cycles_total"] != 0 {
		t.Fatalf("unexpected metrics: %+v", got.Metrics)
	}
	var normal, slow, canceled bool
	for _, group := range got.Workloads {
		if group.PlannedCancel {
			canceled = true
			if group.CompletedLatency != nil {
				t.Fatal("canceled requests entered completed latency distribution")
			}
			continue
		}
		if group.Reader == "slow" {
			slow = true
		} else {
			normal = true
		}
		if group.CompletedLatency == nil || group.CompletedTTFT == nil || group.CompletedGaps == nil {
			t.Fatalf("missing completed timing: %+v", group)
		}
	}
	if !normal || !slow || !canceled {
		t.Fatal("mixed workloads were not separated")
	}
	if _, err := json.Marshal(got); err != nil {
		t.Fatalf("report is not JSON encodable: %v", err)
	}
}

func TestObservationsAndMetrics(t *testing.T) {
	var o observations
	if o.summary() != nil {
		t.Fatal("empty observations must be null")
	}
	for i := 100; i >= 1; i-- {
		o.add(float64(i))
	}
	s := o.summary()
	if s.Count != 100 || s.Mean != 50.5 || s.P50 != 50 || s.P95 != 95 || s.Min != 1 || s.Max != 100 || s.Approximate {
		t.Fatalf("incorrect distribution: %+v", s)
	}
	for i := 0; i < retainedSamples; i++ {
		o.add(20)
	}
	if !o.summary().Approximate || len(o.data) != retainedSamples {
		t.Fatal("reservoir did not bound memory")
	}
	metrics, err := parseMetrics(strings.NewReader("# HELP test help\nmonolith_gc_pause_seconds_total 0.25\nmetric{le=\"2\"} 5\nmonolith_active 2\n"))
	if err != nil || len(metrics) != 2 || metrics["monolith_active"] != 2 {
		t.Fatalf("metrics=%v err=%v", metrics, err)
	}
	if _, err := parseMetrics(strings.NewReader("monolith_active NaN\n")); err == nil {
		t.Fatal("accepted nonfinite metric")
	}
	m := metricReport{Before: map[string]float64{"monolith_gc_cycles_total": 8, "monolith_gc_pause_seconds_total": .25}, After: map[string]float64{"monolith_gc_cycles_total": 2, "monolith_gc_pause_seconds_total": .5}, CounterDeltas: map[string]float64{}}
	m.finish()
	if len(m.CounterResets) != 1 || m.CounterDeltas["monolith_gc_pause_seconds_total"] != .25 {
		t.Fatalf("incorrect deltas: %+v", m)
	}
	if _, ok := m.CounterDeltas["monolith_gc_cycles_total"]; ok {
		t.Fatal("reported negative GC delta after counter reset")
	}
}

func TestOptionsValidation(t *testing.T) {
	for _, args := range [][]string{{"-concurrency=0"}, {"-requests=-1"}, {"-timeout=0"}, {"-url=ftp://example.test"}, {"-url=http://user:pass@example.test"}, {"-cancel-every=2", "-cancel-after=16"}} {
		if _, err := parseOptions(args); err == nil {
			t.Fatalf("accepted invalid options %v", args)
		}
	}
	if _, err := parseOptions([]string{"-requests=20", "-cancel-every=5", "-slow-every=3"}); err != nil {
		t.Fatal(err)
	}
}

func TestServerTerminalOutcomeClassification(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, tc := range []struct {
			code, want string
			status     int
		}{
			{"busy", "overloaded", 429}, {"timeout", "timeout", 408}, {"canceled", "server_canceled", 408},
		} {
			t.Run(fmt.Sprintf("%v-%s", streaming, tc.code), func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprintf(w, "event: error\ndata: {\"error\":\"stopped\",\"code\":%q}\n\n", tc.code)
					} else {
						w.WriteHeader(tc.status)
						fmt.Fprintf(w, "{\"error\":\"stopped\",\"code\":%q}", tc.code)
					}
				}))
				defer srv.Close()
				o := testOptions(srv.URL)
				got := executeRequest(context.Background(), srv.Client(), o, planRequest(o, 1))
				if got.Outcome != tc.want || got.ErrorCode != tc.code {
					t.Fatalf("outcome %+v want %s/%s", got, tc.want, tc.code)
				}
			})
		}
	}
}
