// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHFFetchPreparedReportAndOffline(t *testing.T) {
	hub, source, _ := fakeHFHub(t, false)
	dir, manifest, err := hub.resolve(context.Background(), source)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_HUB_OFFLINE", "1")
	denyHFNetwork(t)
	out, err := os.CreateTemp(t.TempDir(), "fetch-output-")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	stdout := os.Stdout
	os.Stdout = out
	defer func() { os.Stdout = stdout }()
	args := []string{"infer", "local", "-repo", source.Repo, "-revision", source.Revision, "-cache-dir", source.Cache, "-context", "32", "-weight-format", "q4-mixed", "fetch", "-json"}
	if err = run(context.Background(), args); err != nil {
		t.Fatal(err)
	}
	if _, err = out.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	var report struct {
		Format, Path string
		SourceBytes  int64 `json:"source_bytes"`
		Snapshot     hfSnapshot
		Check        hfCheck
	}
	if err = json.NewDecoder(out).Decode(&report); err != nil {
		t.Fatal(err)
	}
	var size int64
	for _, f := range manifest.Files {
		size += f.Size
	}
	if report.Format != "monolith-hf-fetch-v1" || report.Path != dir || report.SourceBytes != size || report.Snapshot.Revision != manifest.Revision || len(report.Snapshot.Files) != 4 {
		t.Fatalf("incorrect receipt: %+v", report)
	}
	if report.Check.WeightValuesChecked || report.Check.Weights.WeightFormat != "q4-mixed" || report.Check.Weights.Config.Context != 32 || !report.Check.ChatSupported {
		t.Fatalf("incorrect preparation: %+v", report.Check)
	}
	// Cached files do not bypass the current operation's model constraints.
	if err = run(context.Background(), []string{"infer", "local", "-repo", source.Repo, "-cache-dir", source.Cache, "-context", "8193", "fetch"}); err == nil {
		t.Fatal("warm cache bypassed context validation")
	}
	// An offline preparation still rejects damaged cached files.
	path := filepath.Join(dir, "model.safetensors")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	b[len(b)-1] ^= 1
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	if err = run(context.Background(), args); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatal("fetch accepted corrupted cache", err)
	}
}

func TestHFFetchPreflightAndHelp(t *testing.T) {
	t.Setenv("HF_HUB_OFFLINE", "0")
	denyHFNetwork(t)
	for _, args := range [][]string{
		{"fetch", "-unknown"}, {"fetch", "extra"}, {"fetch", "-json=invalid"},
		{"-download-timeout", "0", "fetch"}, {"-download-timeout", "25h", "fetch"},
		{"-download-timeout", "invalid", "fetch"}, {"-context", "-1", "fetch"},
		{"-download-mib", "0", "fetch"}, {"-weight-format", "unknown", "fetch"},
		{"fetch", "-context", "64"},
	} {
		t.Run(strings.Join(args, "/"), func(t *testing.T) {
			cache := filepath.Join(t.TempDir(), "cache")
			full := append([]string{"infer", "local", "-repo", "author/toy", "-cache-dir", cache}, args...)
			if err := run(context.Background(), full); err == nil {
				t.Fatal("invalid fetch accepted")
			}
			if _, err := os.Stat(cache); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("invalid fetch created cache", err)
			}
		})
	}
	for _, args := range [][]string{
		{"fetch"}, {"-dir", "missing", "fetch"}, {"-checkpoint", "missing", "fetch"},
		{"-dir", "missing", "-download-timeout", "1h", "check"},
		{"-checkpoint", "missing", "-download-timeout", "1h", "inspect"},
	} {
		if err := runInferLocal(context.Background(), args); err == nil {
			t.Fatal("invalid fetch source accepted", args)
		}
	}
	if err := runInferLocal(context.Background(), []string{"fetch", "-h"}); !errors.Is(err, flag.ErrHelp) {
		t.Fatal("help needs source", err)
	}
	cache := filepath.Join(t.TempDir(), "cache")
	if err := runInferLocal(context.Background(), []string{"-repo", "author/toy", "-cache-dir", cache, "-offline", "fetch"}); err == nil {
		t.Fatal("offline miss accepted")
	}
	if _, err := os.Stat(cache); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("offline miss created cache", err)
	}
}
