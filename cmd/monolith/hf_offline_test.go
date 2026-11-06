// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestHFOfflineEnvironmentValues(t *testing.T) {
	for _, value := range []string{"1", "ON", "yes", "tRuE", "", "0", "false", "off", "2", " true "} {
		t.Run(fmt.Sprintf("%q", value), func(t *testing.T) {
			t.Setenv("HF_HUB_OFFLINE", value)
			want := value == "1" || value == "ON" || value == "yes" || value == "tRuE"
			if got := hfOffline(); got != want {
				t.Fatalf("offline=%v want %v", got, want)
			}
		})
	}
}

// A failed connection is still a network attempt. Count at the dial boundary,
// including clients cloned by the CLI, without contacting external services.
func denyHFNetwork(t *testing.T) *atomic.Int32 {
	t.Helper()
	original := http.DefaultTransport
	transport := original.(*http.Transport).Clone()
	var dials atomic.Int32
	transport.DialContext = func(context.Context, string, string) (net.Conn, error) {
		dials.Add(1)
		return nil, errors.New("unexpected network attempt")
	}
	http.DefaultTransport = transport
	t.Cleanup(func() {
		http.DefaultTransport = original
		transport.CloseIdleConnections()
		if n := dials.Load(); n != 0 {
			t.Errorf("offline made %d network attempts", n)
		}
	})
	return &dials
}

func TestHFOfflineEnvironmentCache(t *testing.T) {
	h, s, counts := fakeHFHub(t, false)
	dir, _, err := h.resolve(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	before := fmt.Sprint(counts())
	t.Setenv("HF_HUB_OFFLINE", "YES")
	denyHFNetwork(t)
	// The flag defaults to false; environment alone must select the cache.
	for _, revision := range []string{"main", strings.Repeat("a", 40)} {
		s.Revision = revision
		got, _, err := h.resolve(context.Background(), s)
		if err != nil || got != dir {
			t.Fatal(got, err)
		}
		args := []string{"infer", "local", "-repo", s.Repo, "-revision", revision,
			"-cache-dir", s.Cache, "-offline=false", "-context", "64", "-weight-mib", "1", "check"}
		if err := run(context.Background(), args); err != nil {
			t.Fatal("CLI did not use offline cache", err)
		}
	}
	for _, revision := range []string{"uncached-tag", strings.Repeat("b", 40)} {
		s.Revision = revision
		if _, _, err := h.resolve(context.Background(), s); err == nil {
			t.Fatal("missing snapshot accepted")
		}
	}
	s.Revision = "main"
	file := filepath.Join(dir, "model.safetensors")
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-1] ^= 1
	if err := os.WriteFile(file, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := h.resolve(context.Background(), s); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatal("corrupt cache accepted", err)
	}
	if got := fmt.Sprint(counts()); got != before {
		t.Fatal("cache read or failure contacted Hub", before, got)
	}
	// A miss must not create even a staging/cache directory.
	s.Cache = filepath.Join(t.TempDir(), "absent")
	if _, _, err := h.resolve(context.Background(), s); err == nil {
		t.Fatal("absent cache accepted")
	}
	if _, err := os.Stat(s.Cache); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("offline miss created cache", err)
	}
}

func TestHFOfflineEnvironmentRemoteBoundaries(t *testing.T) {
	t.Setenv("HF_HUB_OFFLINE", "1")
	t.Setenv("HF_TOKEN", "test-token")
	denyHFNetwork(t)
	client := newHFClient(time.Second)
	defer client.CloseIdleConnections()
	req, _ := http.NewRequest(http.MethodGet, "https://huggingface.co/api/models/author/toy", nil)
	if _, err := hfDo(context.Background(), client, req); !errors.Is(err, errHFOffline) {
		t.Fatal("direct transport boundary", err)
	}
	if _, err := listHFRemote(context.Background(), client, "https://router.huggingface.co/v1", ""); !errors.Is(err, errHFOffline) {
		t.Fatal("direct catalog boundary", err)
	}
	r := hfRemote{Base: "https://router.huggingface.co/v1", Token: "test-token", Model: "author/toy",
		API: "chat", Client: client, MaxTokens: 8, TopP: 1}
	if _, err := r.complete(context.Background(), []ChatMessage{{Role: "user", Content: "hello"}}, nil); !errors.Is(err, errHFOffline) {
		t.Fatal("direct generation boundary", err)
	}
	in, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	defer writer.Close() // Keep empty stdin open to detect accidental input waits.
	original := os.Stdin
	os.Stdin = in
	defer func() { os.Stdin = original }()
	for _, args := range [][]string{
		{"models"},
		{"generate", "-model", "author/toy", "-prompt-file", "-"},
		{"generate", "-model", "author/toy", "-messages", "-"},
		{"chat", "-model", "author/toy"},
	} {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		err := runInferRemote(ctx, args)
		cancel()
		if !errors.Is(err, errHFOffline) {
			t.Fatal(args, err)
		}
	}
}

func TestHFOfflineEnvironmentLocalAndHelpRemainAvailable(t *testing.T) {
	t.Setenv("HF_HUB_OFFLINE", "true")
	denyHFNetwork(t)
	checkpoint := filepath.Join(t.TempDir(), "model.mglm")
	if err := SaveCheckpoint(checkpoint, testModel(t), Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"infer", "models"}, {"infer", "remote", "-h"},
		{"infer", "local", "-repo", "author/toy", "generate", "-h"},
		{"infer", "local", "-dir", hfCLIFixture(t), "check"},
		{"infer", "local", "-checkpoint", checkpoint, "inspect"},
		{"learn", "inspect", "-model", checkpoint},
	} {
		if err := run(context.Background(), args); err != nil && !(args[len(args)-1] == "-h" && errors.Is(err, flag.ErrHelp)) {
			t.Fatal(args, err)
		}
	}
}
