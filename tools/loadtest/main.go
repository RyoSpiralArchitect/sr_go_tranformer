// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

// loadtest exercises the streaming HTTP API with a bounded number of clients.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const maxEventBytes = 8 << 20
const retainedSamples = 100000

type options struct {
	URL            string        `json:"url"`
	Requests       int           `json:"requests"`
	Concurrency    int           `json:"concurrency"`
	ShortPrompt    string        `json:"-"`
	LongPrompt     string        `json:"-"`
	MaxTokens      int           `json:"max_tokens"`
	CancelEvery    int           `json:"cancel_every"`
	CancelAfter    int           `json:"cancel_after_token_events"`
	SlowEvery      int           `json:"slow_every"`
	SlowDelay      time.Duration `json:"-"`
	Timeout        time.Duration `json:"-"`
	SampleInterval time.Duration `json:"-"`
	Details        bool          `json:"-"`
}

func parseOptions(args []string) (options, error) {
	var o options
	f := flag.NewFlagSet("loadtest", flag.ContinueOnError)
	f.StringVar(&o.URL, "url", "http://127.0.0.1:8080", "server base URL")
	f.IntVar(&o.Requests, "requests", 32, "total number of requests")
	f.IntVar(&o.Concurrency, "concurrency", 4, "maximum simultaneous requests")
	f.StringVar(&o.ShortPrompt, "short-prompt", "the ", "prompt for odd-numbered requests")
	f.StringVar(&o.LongPrompt, "long-prompt", strings.Repeat("the cat rests. ", 4), "prompt for even-numbered requests; must fit the model context")
	f.IntVar(&o.MaxTokens, "max-tokens", 16, "maximum generated tokens per request")
	f.IntVar(&o.CancelEvery, "cancel-every", 0, "cancel every Nth request (0 disables)")
	f.IntVar(&o.CancelAfter, "cancel-after", 2, "cancel selected requests after this many token events")
	f.IntVar(&o.SlowEvery, "slow-every", 0, "delay reads for every Nth request (0 disables)")
	f.DurationVar(&o.SlowDelay, "slow-delay", 50*time.Millisecond, "read delay after each token event for slow clients")
	f.DurationVar(&o.Timeout, "timeout", 30*time.Second, "timeout for each request")
	f.DurationVar(&o.SampleInterval, "sample-interval", 100*time.Millisecond, "interval between server metrics samples")
	f.BoolVar(&o.Details, "details", false, "include individual request results")
	if err := f.Parse(args); err != nil {
		return o, err
	}
	if f.NArg() != 0 {
		return o, fmt.Errorf("unexpected arguments: %v", f.Args())
	}
	u, err := url.Parse(o.URL)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return o, errors.New("url must be an HTTP(S) base URL without credentials, query, or fragment")
	}
	o.URL = strings.TrimRight(o.URL, "/")
	if o.Requests < 1 || o.Requests > 1000000 || o.Concurrency < 1 || o.Concurrency > 4096 || o.MaxTokens < 1 || o.MaxTokens > 65536 {
		return o, errors.New("requests must be 1..1000000, concurrency 1..4096, and max-tokens 1..65536")
	}
	if o.CancelEvery < 0 || o.CancelAfter < 1 || o.SlowEvery < 0 || o.SlowDelay < 0 || o.Timeout <= 0 || o.SampleInterval <= 0 {
		return o, errors.New("invalid cancellation, delay, timeout, or sample interval")
	}
	if o.CancelEvery > 0 && o.CancelAfter >= o.MaxTokens {
		return o, errors.New("cancel-after must be less than max-tokens when cancellation is enabled")
	}
	if len(o.ShortPrompt) > 1<<20 || len(o.LongPrompt) > 1<<20 {
		return o, errors.New("each prompt must fit within 1 MiB")
	}
	return o, nil
}

type requestPlan struct {
	ID     int    `json:"id"`
	Prompt string `json:"prompt_kind"`
	Reader string `json:"reader"`
	Cancel bool   `json:"planned_cancel"`
	text   string
}

func planRequest(o options, id int) requestPlan {
	p := requestPlan{ID: id, Prompt: "short", Reader: "normal", text: o.ShortPrompt}
	if id%2 == 0 {
		p.Prompt, p.text = "long", o.LongPrompt
	}
	if o.SlowEvery > 0 && id%o.SlowEvery == 0 {
		p.Reader = "slow"
	}
	p.Cancel = o.CancelEvery > 0 && id%o.CancelEvery == 0
	return p
}

type requestResult struct {
	requestPlan
	Outcome      string   `json:"outcome"`
	HTTPStatus   int      `json:"http_status,omitempty"`
	ErrorCode    string   `json:"error_code,omitempty"`
	Error        string   `json:"error,omitempty"`
	TokenEvents  int      `json:"token_events"`
	TTFTMillis   *float64 `json:"first_token_event_ms,omitempty"`
	LatencyMS    float64  `json:"latency_ms"`
	PromptTokens int      `json:"prompt_tokens,omitempty"`
	gaps         []float64
}

type tokenEvent struct {
	TokenID *int   `json:"token_id"`
	Text    string `json:"text"`
}

type completion struct {
	Text             string `json:"text"`
	TokenIDs         []int  `json:"token_ids"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	FinishReason     string `json:"finish_reason"`
}

// readEvents handles SSE framing, including comments and multiline data. Event
// payloads are bounded; an incomplete event at EOF is a protocol error.
func readEvents(r io.Reader, visit func(string, []byte) error) error {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), maxEventBytes)
	var name string
	var data []byte
	for s.Scan() {
		line := s.Text()
		if line == "" {
			if len(data) > 0 {
				if err := visit(name, data[:len(data)-1]); err != nil {
					return err
				}
			}
			name, data = "", data[:0]
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, found := strings.Cut(line, ":")
		if !found {
			value = ""
		}
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			name = value
		case "data":
			if len(data)+len(value)+1 > maxEventBytes {
				return errors.New("SSE event exceeds 8 MiB")
			}
			data = append(data, value...)
			data = append(data, '\n')
		}
	}
	if err := s.Err(); err != nil {
		return err
	}
	if name != "" || len(data) != 0 {
		return errors.New("incomplete SSE event at EOF")
	}
	return nil
}

var errPlannedCancel = errors.New("planned cancellation")
var errServerEvent = errors.New("server error event")

func classifyHTTP(code int) string {
	switch {
	case code == http.StatusTooManyRequests:
		return "overloaded"
	case code == http.StatusRequestTimeout:
		return "timeout"
	case code == http.StatusRequestTimeout || code == http.StatusGatewayTimeout:
		return "timeout"
	case code >= 500:
		return "server_error"
	default:
		return "request_error"
	}
}

func classifyServerCode(code, fallback string) string {
	switch code {
	case "busy", "overloaded":
		return "overloaded"
	case "timeout":
		return "timeout"
	case "canceled":
		return "server_canceled"
	default:
		return fallback
	}
}

func executeRequest(parent context.Context, client *http.Client, o options, p requestPlan) (result requestResult) {
	result.requestPlan = p
	result.Outcome = "transport_failure"
	body, _ := json.Marshal(map[string]any{"prompt": p.text, "max_tokens": o.MaxTokens, "temperature": 0, "repeat_penalty": 1, "stream": true})
	ctx, cancel := context.WithTimeout(parent, o.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.URL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		result.Error = err.Error()
		return result
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	start := time.Now()
	defer func() { result.LatencyMS = float64(time.Since(start)) / float64(time.Millisecond) }()
	resp, err := client.Do(req)
	if err != nil {
		result.Error = err.Error()
		result.Outcome = contextOutcome(ctx)
		return result
	}
	defer resp.Body.Close()
	result.HTTPStatus = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		result.Outcome, result.Error = classifyHTTP(resp.StatusCode), strings.TrimSpace(string(b))
		var e struct {
			Error string `json:"error"`
			Code  string `json:"code"`
		}
		if json.Unmarshal(b, &e) == nil {
			result.ErrorCode = e.Code
			result.Outcome = classifyServerCode(e.Code, result.Outcome)
			if e.Error != "" {
				result.Error = e.Error
			}
		}
		return result
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		result.Outcome, result.Error = "protocol_error", "expected text/event-stream response"
		return result
	}
	var ids []int
	var text strings.Builder
	var previous time.Time
	done := false
	err = readEvents(resp.Body, func(event string, data []byte) error {
		if done {
			return errors.New("event after done")
		}
		switch event {
		case "token":
			var token tokenEvent
			if err := json.Unmarshal(data, &token); err != nil || token.TokenID == nil || *token.TokenID < 0 {
				return errors.New("invalid token event")
			}
			if len(ids) >= o.MaxTokens || text.Len()+len(token.Text) > maxEventBytes {
				return errors.New("stream exceeds configured token or text limit")
			}
			now := time.Now()
			if previous.IsZero() {
				ms := float64(now.Sub(start)) / float64(time.Millisecond)
				result.TTFTMillis = &ms
			} else {
				result.gaps = append(result.gaps, float64(now.Sub(previous))/float64(time.Millisecond))
			}
			previous = now
			ids = append(ids, *token.TokenID)
			text.WriteString(token.Text)
			result.TokenEvents++
			if p.Cancel && len(ids) >= o.CancelAfter {
				cancel()
				return errPlannedCancel
			}
			if p.Reader == "slow" && o.SlowDelay > 0 {
				timer := time.NewTimer(o.SlowDelay)
				defer timer.Stop()
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-timer.C:
				}
			}
		case "done":
			var c completion
			if err := json.Unmarshal(data, &c); err != nil {
				return fmt.Errorf("invalid done event: %w", err)
			}
			if c.CompletionTokens != len(ids) || len(c.TokenIDs) != len(ids) || c.Text != text.String() || c.PromptTokens < 1 || c.FinishReason == "" {
				return errors.New("done metadata/text does not match token events")
			}
			for i, id := range ids {
				if c.TokenIDs[i] != id {
					return errors.New("done token IDs do not match token events")
				}
			}
			result.PromptTokens = c.PromptTokens
			done = true
		case "error":
			var e struct {
				Error string `json:"error"`
				Code  string `json:"code"`
			}
			if err := json.Unmarshal(data, &e); err != nil || e.Error == "" || e.Code == "" {
				return errors.New("invalid error event")
			}
			result.Error, result.ErrorCode = e.Error, e.Code
			return errServerEvent
		default:
			return fmt.Errorf("unexpected SSE event %q", event)
		}
		return nil
	})
	switch {
	case errors.Is(err, errPlannedCancel):
		result.Outcome = "canceled"
	case errors.Is(err, errServerEvent):
		result.Outcome = classifyServerCode(result.ErrorCode, "server_error")
	case ctx.Err() != nil:
		result.Outcome, result.Error = contextOutcome(ctx), ctx.Err().Error()
	case err != nil:
		result.Outcome, result.Error = "protocol_error", err.Error()
		var netErr interface{ Timeout() bool }
		if errors.As(err, &netErr) || errors.Is(err, io.ErrUnexpectedEOF) {
			result.Outcome = "transport_failure"
		}
	case !done:
		result.Outcome, result.Error = "protocol_error", "stream ended without a done event"
	default:
		result.Outcome = "completed"
	}
	return result
}

func contextOutcome(ctx context.Context) string {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return "timeout"
	}
	if ctx.Err() != nil {
		return "interrupted"
	}
	return "transport_failure"
}

// Reservoir sampling bounds summary memory for very long load runs. Typical
// small runs retain every observation and report exact nearest-rank quantiles.
type observations struct {
	count int64
	sum   float64
	min   float64
	max   float64
	data  []float64
	rng   *rand.Rand
}

type distribution struct {
	Count       int64   `json:"count"`
	Samples     int     `json:"retained_samples"`
	Approximate bool    `json:"quantiles_sampled"`
	Min         float64 `json:"min_ms"`
	Mean        float64 `json:"mean_ms"`
	P50         float64 `json:"p50_ms"`
	P95         float64 `json:"p95_ms"`
	Max         float64 `json:"max_ms"`
}

func (s *observations) add(v float64) {
	s.count++
	s.sum += v
	if s.count == 1 || v < s.min {
		s.min = v
	}
	if v > s.max {
		s.max = v
	}
	if len(s.data) < retainedSamples {
		s.data = append(s.data, v)
		return
	}
	if s.rng == nil {
		s.rng = rand.New(rand.NewSource(1))
	}
	if n := s.rng.Int63n(s.count); n < int64(len(s.data)) {
		s.data[n] = v
	}
}

func (s *observations) summary() *distribution {
	if s.count == 0 {
		return nil
	}
	data := append([]float64(nil), s.data...)
	sort.Float64s(data)
	q := func(p float64) float64 { return data[int(math.Ceil(float64(len(data))*p))-1] }
	return &distribution{s.count, len(data), s.count > int64(len(data)), s.min, s.sum / float64(s.count), q(.5), q(.95), s.max}
}

type workloadSummary struct {
	Prompt              string            `json:"prompt_kind"`
	Reader              string            `json:"reader"`
	PlannedCancel       bool              `json:"planned_cancel"`
	Outcomes            map[string]int    `json:"outcomes"`
	CompletedTTFT       *distribution     `json:"completed_first_token_event"`
	CompletedGaps       *distribution     `json:"completed_token_event_gap"`
	CompletedLatency    *distribution     `json:"completed_latency"`
	ObservedTTFT        *distribution     `json:"all_observed_first_token_event"`
	ErrorCodes          map[string]int    `json:"error_codes,omitempty"`
	Examples            map[string]string `json:"error_examples,omitempty"`
	ttft, gaps, latency observations
	observed            observations
}

func (s *workloadSummary) add(r requestResult) {
	s.Outcomes[r.Outcome]++
	if r.ErrorCode != "" {
		s.ErrorCodes[r.ErrorCode]++
	}
	if r.Error != "" {
		if _, ok := s.Examples[r.Outcome]; !ok {
			s.Examples[r.Outcome] = r.Error
		}
	}
	if r.TTFTMillis != nil {
		s.observed.add(*r.TTFTMillis)
	}
	if r.Outcome == "completed" {
		if r.TTFTMillis != nil {
			s.ttft.add(*r.TTFTMillis)
		}
		for _, gap := range r.gaps {
			s.gaps.add(gap)
		}
		s.latency.add(r.LatencyMS)
	}
}

var peakMetricNames = []string{
	"monolith_heap_alloc_bytes", "monolith_heap_sys_bytes", "monolith_goroutines",
	"monolith_active", "monolith_queue_depth", "monolith_kv_reserved_bytes", "monolith_kv_allocated_bytes",
}
var counterMetricNames = []string{
	"monolith_gc_cycles_total", "monolith_gc_pause_seconds_total", "monolith_submitted_total",
	"monolith_completed_total", "monolith_canceled_total", "monolith_failed_total", "monolith_rejected_total",
	"monolith_input_tokens_total", "monolith_generated_tokens_total", "monolith_ticks_total", "monolith_slow_consumers_total",
}

type metricReport struct {
	Samples       int                `json:"samples"`
	Errors        int                `json:"sample_errors"`
	FirstError    string             `json:"first_error,omitempty"`
	Before        map[string]float64 `json:"before"`
	After         map[string]float64 `json:"after"`
	SampledPeaks  map[string]float64 `json:"sampled_peaks"`
	CounterDeltas map[string]float64 `json:"counter_deltas"`
	CounterResets []string           `json:"counter_resets,omitempty"`
	Missing       []string           `json:"missing_metrics,omitempty"`
}

func parseMetrics(r io.Reader) (map[string]float64, error) {
	values := map[string]float64{}
	s := bufio.NewScanner(io.LimitReader(r, 1<<20))
	for s.Scan() {
		f := strings.Fields(s.Text())
		if len(f) < 2 || strings.HasPrefix(f[0], "#") || strings.Contains(f[0], "{") {
			continue
		}
		v, err := strconv.ParseFloat(f[1], 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("invalid metric %q", f[0])
		}
		values[f[0]] = v
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return nil, errors.New("metrics response contains no scalar metrics")
	}
	return values, nil
}

func (m *metricReport) sample(ctx context.Context, client *http.Client, endpoint string) map[string]float64 {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/metrics", nil)
	var values map[string]float64
	if err == nil {
		var resp *http.Response
		resp, err = client.Do(req)
		if err == nil {
			if resp.StatusCode != http.StatusOK {
				err = fmt.Errorf("metrics returned HTTP %d", resp.StatusCode)
			} else {
				values, err = parseMetrics(resp.Body)
			}
			resp.Body.Close()
		}
	}
	if err != nil {
		m.Errors++
		if m.FirstError == "" {
			m.FirstError = err.Error()
		}
		return nil
	}
	m.Samples++
	for _, name := range peakMetricNames {
		v, ok := values[name]
		if old, seen := m.SampledPeaks[name]; ok && (!seen || v > old) {
			m.SampledPeaks[name] = v
		}
	}
	return values
}

func (m *metricReport) finish() {
	for _, name := range counterMetricNames {
		before, b := m.Before[name]
		after, a := m.After[name]
		if a && b {
			if after >= before {
				m.CounterDeltas[name] = after - before
			} else {
				m.CounterResets = append(m.CounterResets, name)
			}
		}
	}
	for _, name := range append(append([]string(nil), peakMetricNames...), counterMetricNames...) {
		if _, ok := m.After[name]; !ok {
			m.Missing = append(m.Missing, name)
		}
	}
}

type report struct {
	Options          options            `json:"options"`
	ShortPromptBytes int                `json:"short_prompt_bytes"`
	LongPromptBytes  int                `json:"long_prompt_bytes"`
	SlowDelayMS      float64            `json:"slow_delay_ms"`
	TimeoutMS        float64            `json:"request_timeout_ms"`
	SampleIntervalMS float64            `json:"metrics_sample_interval_ms"`
	ElapsedSeconds   float64            `json:"elapsed_seconds"`
	Started          int                `json:"started_requests"`
	CompletedTokens  int64              `json:"completed_token_events"`
	TokensPerSecond  float64            `json:"completed_token_events_per_second"`
	Outcomes         map[string]int     `json:"outcomes"`
	Workloads        []*workloadSummary `json:"workloads"`
	Metrics          metricReport       `json:"server_metrics"`
	Results          []requestResult    `json:"requests,omitempty"`
}

func run(ctx context.Context, o options) report {
	transport := &http.Transport{MaxConnsPerHost: o.Concurrency, MaxIdleConns: o.Concurrency, MaxIdleConnsPerHost: o.Concurrency, DisableCompression: true}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	metricsTransport := &http.Transport{MaxConnsPerHost: 1, MaxIdleConns: 1, MaxIdleConnsPerHost: 1}
	defer metricsTransport.CloseIdleConnections()
	metricsClient := &http.Client{Transport: metricsTransport, Timeout: min(5*time.Second, o.Timeout)}
	out := report{Options: o, ShortPromptBytes: len(o.ShortPrompt), LongPromptBytes: len(o.LongPrompt), SlowDelayMS: float64(o.SlowDelay) / float64(time.Millisecond), TimeoutMS: float64(o.Timeout) / float64(time.Millisecond), SampleIntervalMS: float64(o.SampleInterval) / float64(time.Millisecond), Outcomes: map[string]int{}}
	out.Metrics.SampledPeaks, out.Metrics.CounterDeltas = map[string]float64{}, map[string]float64{}
	out.Metrics.Before = out.Metrics.sample(ctx, metricsClient, o.URL)
	stopSample, sampleDone := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(sampleDone)
		ticker := time.NewTicker(o.SampleInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stopSample:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				out.Metrics.sample(ctx, metricsClient, o.URL)
			}
		}
	}()
	start := time.Now()
	results := make(chan requestResult, o.Concurrency)
	var next atomic.Int64
	var workers sync.WaitGroup
	for i := 0; i < min(o.Requests, o.Concurrency); i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for ctx.Err() == nil {
				id := int(next.Add(1))
				if id > o.Requests {
					return
				}
				results <- executeRequest(ctx, client, o, planRequest(o, id))
			}
		}()
	}
	go func() { workers.Wait(); close(results) }()
	groups := map[string]*workloadSummary{}
	for result := range results {
		out.Started++
		out.Outcomes[result.Outcome]++
		key := fmt.Sprintf("%s/%s/%t", result.Prompt, result.Reader, result.Cancel)
		group := groups[key]
		if group == nil {
			group = &workloadSummary{Prompt: result.Prompt, Reader: result.Reader, PlannedCancel: result.Cancel, Outcomes: map[string]int{}, ErrorCodes: map[string]int{}, Examples: map[string]string{}}
			groups[key] = group
		}
		group.add(result)
		if result.Outcome == "completed" {
			out.CompletedTokens += int64(result.TokenEvents)
		}
		if o.Details {
			result.gaps = nil
			out.Results = append(out.Results, result)
		}
	}
	out.ElapsedSeconds = time.Since(start).Seconds()
	if out.ElapsedSeconds > 0 {
		out.TokensPerSecond = float64(out.CompletedTokens) / out.ElapsedSeconds
	}
	close(stopSample)
	<-sampleDone
	out.Metrics.After = out.Metrics.sample(ctx, metricsClient, o.URL)
	out.Metrics.finish()
	var keys []string
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		group := groups[key]
		group.CompletedTTFT, group.CompletedGaps, group.CompletedLatency = group.ttft.summary(), group.gaps.summary(), group.latency.summary()
		group.ObservedTTFT = group.observed.summary()
		out.Workloads = append(out.Workloads, group)
	}
	sort.Slice(out.Results, func(i, j int) bool { return out.Results[i].ID < out.Results[j].ID })
	return out
}

func main() {
	o, err := parseOptions(os.Args[1:])
	if errors.Is(err, flag.ErrHelp) {
		return
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	result := run(ctx, o)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// Planned cancellations and overload are workload outcomes, not harness
	// failures. Surface malformed streams, timeouts, or server/network failures.
	for outcome, count := range result.Outcomes {
		if count > 0 && outcome != "completed" && outcome != "canceled" && outcome != "overloaded" {
			os.Exit(1)
		}
	}
}
