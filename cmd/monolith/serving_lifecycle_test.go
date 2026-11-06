// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"math"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Gate the real PrefillBatch at its first layer, after KV pages are allocated.
// The actor's loop, prefill entry, and post-validation checks precede this one.
// Only the actor uses Err after staging the cohort; no kernel is substituted.
type servingBatchGate struct {
	context.Context
	calls   atomic.Int64
	entered chan struct{}
	resume  chan struct{}
	once    sync.Once
}

func (g *servingBatchGate) Err() error {
	if g.calls.Add(1) == 4 {
		close(g.entered)
		<-g.resume
	}
	return g.Context.Err()
}
func (g *servingBatchGate) open() { g.once.Do(func() { close(g.resume) }) }

func startGatedEngine(t *testing.T, e *Engine) *servingBatchGate {
	t.Helper()
	g := &servingBatchGate{Context: e.ctx, entered: make(chan struct{}), resume: make(chan struct{})}
	e.ctx = g
	go e.run()
	t.Cleanup(func() { g.open(); e.Close() })
	waitLifecycle(t, g.entered)
	return g
}

func waitLifecycle(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("lifecycle transition timed out")
	}
}

func assertNotReleased(t *testing.T, s *CompletionStream) {
	t.Helper()
	select {
	case <-s.Released():
		t.Fatal("released resources while their batch was still executing")
	default:
	}
}

func assertEngineDrained(t *testing.T, e *Engine, released int64) {
	t.Helper()
	s := e.Stats()
	if s.Active != 0 || s.Queue != 0 || s.KVReservedBytes != 0 || s.Computing != 0 || s.AdmissionUsed != 0 || s.Released != released {
		t.Fatalf("engine not drained: %+v", s)
	}
}

func TestReleaseWaitsForMixedBatchCompute(t *testing.T) {
	for _, source := range []string{"close", "parent"} {
		t.Run(source, func(t *testing.T) {
			m := workerTestModel(t, 2)
			e, err := newEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 1 << 20, StreamBuffer: 64, PrefillChunk: 32, KernelWorkers: 4})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(streamTestContext(t))
			defer cancel()
			sampling := defaultSampling()
			sampling.MaxTokens = 8
			want, err := Generate(context.Background(), m, Tokenizer{}, "healthy", sampling)
			if err != nil {
				t.Fatal(err)
			}
			canceled, err := e.Stream(ctx, strings.Repeat("x", 40), sampling)
			if err != nil {
				t.Fatal(err)
			}
			healthy, err := e.Stream(streamTestContext(t), "healthy", sampling)
			if err != nil {
				t.Fatal(err)
			}
			queued, err := e.Stream(streamTestContext(t), "queued", sampling)
			if err != nil {
				t.Fatal(err)
			}
			gate := startGatedEngine(t, e)
			before := e.Stats()
			if before.Active != 2 || before.Computing != 2 || before.Queue != 1 || before.KVReservedBytes == 0 {
				t.Fatalf("fixture did not gate a mixed batch: %+v", before)
			}
			if source == "close" {
				canceled.Close()
			} else {
				cancel()
			}
			queued.Close()
			// Ensure the parent callback has run while the compute gate is closed.
			await(t, "cancellation observation", func() bool {
				r := canceled.request
				r.lifecycleMu.Lock()
				defer r.lifecycleMu.Unlock()
				return !r.cancelObserved.IsZero()
			})
			if _, err := canceled.Wait(); !errors.Is(err, context.Canceled) {
				t.Fatalf("Wait did not return promptly on cancellation: %v", err)
			}
			assertNotReleased(t, canceled)
			assertNotReleased(t, queued)
			if got := e.Stats(); got.KVReservedBytes != before.KVReservedBytes || got.AdmissionUsed != 3 || got.Released != 0 {
				t.Fatalf("premature resource return: %+v", got)
			}
			gate.open()
			waitLifecycle(t, canceled.Released())
			waitLifecycle(t, queued.Released())
			events, got, err := collectTestStream(t, streamTestContext(t), healthy)
			if err != nil {
				t.Fatal(err)
			}
			assertStreamParity(t, events, got, want)
			waitLifecycle(t, healthy.Released())
			assertEngineDrained(t, e, 3)
			if e.Stats().Canceled != 2 || e.cancelRelease.count.Load() != 2 || e.cancelCompute.count.Load() != 2 {
				t.Fatal("cancellation observations were lost or counted twice")
			}
			if e.cancelCompute.nanos.Load() == 0 || e.cancelRelease.nanos.Load() < e.cancelCompute.nanos.Load() {
				t.Fatal("compute completion did not precede resource return")
			}
			if !queued.request.computeDone.IsZero() {
				t.Fatal("canceled queued request performed computation")
			}
			// A returned permit can immediately admit more work, and cancellation
			// of an already completed handle cannot create a new observation.
			healthy.Close()
			canceled.Close()
			again, err := e.Generate(streamTestContext(t), "healthy", sampling)
			if err != nil || !reflect.DeepEqual(again, want) {
				t.Fatalf("capacity reuse changed output: %v", err)
			}
			if e.cancelRelease.count.Load() != 2 {
				t.Fatal("normal cleanup or late Close was counted as cancellation")
			}
		})
	}
}

func TestShutdownReleaseWaitsForComputeAndDrainsQueue(t *testing.T) {
	m := testModel(t)
	e, err := newEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	sampling := defaultSampling()
	sampling.MaxTokens = 8
	streams := make([]*CompletionStream, 3)
	for i := range streams {
		streams[i], err = e.Stream(context.Background(), strings.Repeat("x", 32), sampling)
		if err != nil {
			t.Fatal(err)
		}
	}
	gate := startGatedEngine(t, e)
	e.cancel()
	for _, stream := range streams {
		await(t, "shutdown observed", func() bool {
			r := stream.request
			r.lifecycleMu.Lock()
			defer r.lifecycleMu.Unlock()
			return !r.cancelObserved.IsZero()
		})
		assertNotReleased(t, stream)
	}
	gate.open()
	waitLifecycle(t, e.done)
	for _, stream := range streams {
		waitLifecycle(t, stream.Released())
		if _, err := stream.Wait(); !errors.Is(err, ErrClosed) {
			t.Fatalf("shutdown result: %v", err)
		}
	}
	assertEngineDrained(t, e, 3)
	if e.Stats().Canceled != 3 || e.cancelRelease.count.Load() != 3 || e.cancelCompute.count.Load() != 3 {
		t.Fatal("shutdown did not account for active and queued cancellations")
	}
}

func TestReleasedCoversSlowConsumerAndMetrics(t *testing.T) {
	m := zeroStreamModel(t, 128)
	e, err := newEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 4, CacheBytes: 1 << 20, StreamBuffer: 1})
	if err != nil {
		t.Fatal(err)
	}
	sampling := defaultSampling()
	sampling.MaxTokens, sampling.Temperature = 8, 0
	slow, err := e.Stream(context.Background(), "slow", sampling)
	if err != nil {
		t.Fatal(err)
	}
	queued, err := e.Stream(context.Background(), "canceled before execution", sampling)
	if err != nil {
		t.Fatal(err)
	}
	queued.Close()
	go e.run()
	defer e.Close()
	waitLifecycle(t, slow.Released()) // No event consumption is needed to release.
	waitLifecycle(t, queued.Released())
	if _, err := slow.Wait(); !errors.Is(err, ErrSlowConsumer) {
		t.Fatalf("slow consumer result: %v", err)
	}
	if _, err := queued.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("queued cancellation: %v", err)
	}
	assertEngineDrained(t, e, 2)
	if e.cancelCompute.nanos.Load() != 0 || e.cancelCompute.count.Load() != 1 {
		t.Fatal("request canceled before execution should have zero compute tail")
	}
	if e.Stats().SlowConsumers != 1 || e.Stats().Canceled != 1 {
		t.Fatal("slow consumer cleanup was misclassified as a cancellation")
	}
	w := httptest.NewRecorder()
	e.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/metrics", nil))
	for _, metric := range []string{
		"monolith_released_total 2\n", "monolith_computing_sessions 0\n", "monolith_admission_used 0\n",
		"monolith_cancel_observed_to_compute_done_seconds_count 1\n",
		"monolith_cancel_observed_to_release_seconds_count 1\n",
		"monolith_batch_compute_seconds_count 2\n",
	} {
		if !strings.Contains(w.Body.String(), metric) {
			t.Fatalf("missing metric %q in %s", metric, w.Body.String())
		}
	}
}

func TestServingTracePreservesFilesAndProducesTrace(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "serve.trace")
	recorder, err := startInferenceTrace(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if recorder != nil {
			_ = recorder.close()
		}
	})
	// A competing recorder must leave both the existing file and trace alone.
	if _, err := startInferenceTrace(path); err == nil {
		t.Fatal("trace output was overwritten")
	}
	conflict := filepath.Join(dir, "conflict.trace")
	if _, err := startInferenceTrace(conflict); err == nil {
		t.Fatal("concurrent process-wide tracing accepted")
	}
	if _, err := os.Stat(conflict); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("failed trace startup left a reserved output path")
	}
	m := testModel(t)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 2, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(e.Close)
	sampling := defaultSampling()
	sampling.MaxTokens = 4
	want, err := Generate(context.Background(), m, Tokenizer{}, "trace", sampling)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Generate(streamTestContext(t), "trace", sampling)
	e.Close()
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("traced inference changed output: %v", err)
	}
	closeErr := recorder.close()
	recorder = nil
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) < 16 || !strings.Contains(string(raw[:16]), "trace") {
		t.Fatalf("invalid trace file: bytes=%d error=%v", len(raw), err)
	}
	// Trace shutdown must surface write failures, even though runtime/trace
	// does not return them from Stop.
	broken, err := startInferenceTrace(filepath.Join(dir, "broken.trace"))
	if err != nil {
		t.Fatal(err)
	}
	if err := broken.file.Close(); err != nil {
		t.Fatal(err)
	}
	if err := broken.close(); err == nil {
		t.Fatal("trace shutdown hid output failure")
	}
}

func TestReleasedOnAsynchronousRejectionAndModelFailure(t *testing.T) {
	for _, failure := range []string{"kv_budget", "sampling"} {
		t.Run(failure, func(t *testing.T) {
			m := testModel(t)
			pageBytes := int64(2 * pageTokens * m.Config.KVDim() * m.Config.Layers * 4)
			budget := pageBytes
			if failure == "sampling" {
				budget *= 2
				m.Emb.Data[0] = float32(math.NaN())
			}
			e, err := newEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 2, Queue: 2, CacheBytes: budget, StreamBuffer: 16})
			if err != nil {
				t.Fatal(err)
			}
			sampling := defaultSampling()
			sampling.MaxTokens = 8
			streams := make([]*CompletionStream, 2)
			for i := range streams {
				streams[i], err = e.Stream(context.Background(), "x", sampling)
				if err != nil {
					t.Fatal(err)
				}
			}
			go e.run()
			defer e.Close()
			for i, stream := range streams {
				waitLifecycle(t, stream.Released())
				_, err := stream.Wait()
				if failure == "sampling" {
					if err == nil || !strings.Contains(err.Error(), "nonfinite") {
						t.Fatalf("model failure was not delivered: %v", err)
					}
				} else if i == 1 && !errors.Is(err, ErrBusy) {
					t.Fatalf("reservation rejection was not delivered: %v", err)
				} else if i == 0 && err != nil {
					t.Fatal(err)
				}
			}
			assertEngineDrained(t, e, 2)
			if e.cancelRelease.count.Load() != 0 {
				t.Fatal("failure cleanup was counted as cancellation")
			}
		})
	}
}
