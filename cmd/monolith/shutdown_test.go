// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// Holding done open models a kernel that cannot observe cancellation until its
// current computation ends. No large allocations or slow computation are needed.
func heldShutdownEngine(t *testing.T) (*Engine, func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(done) }) }
	t.Cleanup(func() { cancel(); release() })
	return &Engine{ctx: ctx, cancel: cancel, done: done}, release
}

func TestServerShutdownStopsHTTPBeforeEngineFinishes(t *testing.T) {
	e, _ := heldShutdownEngine(t)
	entered := make(chan struct{})
	disconnected := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		<-r.Context().Done()
		close(disconnected)
	}))
	defer srv.Close()
	clientResult := make(chan error, 1)
	go func() {
		response, err := srv.Client().Get(srv.URL)
		if response != nil {
			response.Body.Close()
		}
		clientResult <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("HTTP request never entered handler")
	}
	stopping := make(chan struct{})
	srv.Config.RegisterOnShutdown(func() { close(stopping) })
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	result := make(chan error, 1)
	go func() { result <- shutdownInferenceServer(ctx, srv.Config, e) }()
	select {
	case <-stopping:
	case <-time.After(5 * time.Second):
		t.Fatal("HTTP shutdown waited for the unfinished engine")
	}
	if e.ctx.Err() == nil {
		t.Fatal("engine cancellation was not signaled before HTTP shutdown")
	}
	select {
	case err := <-result:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("shutdown result: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("shutdown ignored its deadline")
	}
	select {
	case <-e.done:
		t.Fatal("test engine should still be executing its held kernel")
	default:
	}
	select {
	case <-disconnected:
	case <-time.After(5 * time.Second):
		t.Fatal("deadline left the HTTP connection open")
	}
	select {
	case err := <-clientResult:
		if err == nil {
			t.Fatal("held request unexpectedly completed normally")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("client remained blocked after forced HTTP close")
	}
}

func TestServerShutdownBoundsEngineWaitAfterHTTPDrains(t *testing.T) {
	e, _ := heldShutdownEngine(t)
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	result := make(chan error, 1)
	go func() { result <- shutdownInferenceServer(ctx, srv.Config, e) }()
	select {
	case err := <-result:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("engine wait result: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("engine wait exceeded shutdown deadline after HTTP drained")
	}
	if e.ctx.Err() == nil {
		t.Fatal("engine was not canceled")
	}
}

func TestServerShutdownDrainsCooperativeRequest(t *testing.T) {
	e, release := heldShutdownEngine(t)
	entered := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		select {
		case <-e.done:
			_, _ = io.WriteString(w, "drained")
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	responseResult := make(chan string, 1)
	go func() {
		response, err := srv.Client().Get(srv.URL)
		if err != nil {
			responseResult <- err.Error()
			return
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		if err != nil {
			responseResult <- err.Error()
			return
		}
		responseResult <- string(body)
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("HTTP request never entered handler")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	go func() { <-e.ctx.Done(); release() }()
	if err := shutdownInferenceServer(ctx, srv.Config, e); err != nil {
		t.Fatal(err)
	}
	select {
	case body := <-responseResult:
		if body != "drained" {
			t.Fatalf("cooperative request did not drain: %q", body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cooperative response was stranded")
	}
}
