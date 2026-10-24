// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func q8BudgetFixture(t *testing.T) string {
	t.Helper()
	dir := hfCLIFixture(t)
	h, _, err := readHFConfig(dir, 64)
	if err != nil {
		t.Fatal(err)
	}
	h.Dim, h.Hidden, h.Heads, h.KVHeads = 128, 256, 8, 4
	hfWriteJSON(t, filepath.Join(dir, "config.json"), h)
	_, c, err := readHFConfig(dir, 64)
	if err != nil {
		t.Fatal(err)
	}
	header := map[string]any{}
	var data []byte
	for _, s := range hfProfileSpecs(c, h) {
		count := 1
		for _, d := range s.Shape {
			count *= d
		}
		start := len(data)
		for i := 0; i < count; i++ {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(float32(i%19-9)/64))
		}
		header[s.Name] = map[string]any{"dtype": "F32", "shape": s.Shape, "data_offsets": []int{start, len(data)}}
	}
	hfWriteShard(t, filepath.Join(dir, "model.safetensors"), header, data)
	return dir
}

func TestQ8SourceBudgetAndOfflineCLI(t *testing.T) {
	dir := q8BudgetFixture(t)
	if _, err := checkHF(dir, 64, 1, false); err == nil {
		t.Fatal("FP32 accepted insufficient weight budget")
	}
	plan, err := checkHFFormat(dir, 64, 1, false, "q8")
	if err != nil || plan.Weights.WeightFormat != "q8" || plan.Weights.WeightBytes >= 1<<20 || plan.Weights.FP32WeightBytes <= 1<<20 {
		t.Fatal("q8 capacity not applied", plan, err)
	}
	h, s, counts := fakeHFHub(t, false, dir)
	s.WeightMiB = 1
	if _, _, err := h.resolve(context.Background(), s); err == nil {
		t.Fatal("FP32 source accepted insufficient budget")
	}
	for path := range counts() {
		if filepath.Ext(path) == ".safetensors" {
			t.Fatal("over-budget source fetched weights")
		}
	}
	s.WeightFormat = "q8"
	if _, _, err := h.resolve(context.Background(), s); err != nil {
		t.Fatal("q8 source budget not applied", err)
	}
	before := counts()
	t.Setenv("HF_HUB_OFFLINE", "1")
	denyHFNetwork(t)
	for _, action := range [][]string{
		{"check", "-json"}, {"inspect"}, {"tokens", "-prompt", "x"},
		{"ids", "-ids", "1,3", "-max-tokens", "2"},
		{"generate", "-prompt", "x", "-max-tokens", "2", "-temperature", "0"},
	} {
		args := []string{"infer", "local", "-repo", s.Repo, "-cache-dir", s.Cache, "-context", "64", "-weight-format", "q8", "-weight-mib", "1"}
		if err := run(context.Background(), append(args, action...)); err != nil {
			t.Fatal(action, err)
		}
	}
	if !reflect.DeepEqual(before, counts()) {
		t.Fatal("offline q8 used network")
	}
}

func TestQ8FlagPreflight(t *testing.T) {
	t.Setenv("HF_HUB_OFFLINE", "0")
	denyHFNetwork(t)
	for _, args := range [][]string{
		{"-weight-format", "int4", "generate", "-prompt", "x"},
		{"-weight-format", "", "inspect"},
		{"generate", "-weight-format", "q8", "-prompt", "x"},
	} {
		cache := filepath.Join(t.TempDir(), "cache")
		all := append([]string{"infer", "local", "-repo", "author/toy", "-cache-dir", cache}, args...)
		if err := run(context.Background(), all); err == nil {
			t.Fatal("invalid flags accepted")
		}
		if _, err := os.Stat(cache); !errors.Is(err, os.ErrNotExist) {
			t.Fatal("preflight created cache", err)
		}
	}
	for _, args := range [][]string{
		{"infer", "local", "-checkpoint", "missing", "-weight-format", "q8", "generate", "-prompt", "x"},
		{"generate", "-model", "missing", "-weight-format", "q8", "-prompt", "x"},
	} {
		if err := run(context.Background(), args); err == nil || errors.Is(err, os.ErrNotExist) {
			t.Fatal("native q8 was not rejected before checkpoint open", err)
		}
	}
}

func TestQ8EngineSharedModelAndStreams(t *testing.T) {
	dir := hfCLIFixture(t)
	m, _, err := loadHFModelFormat(context.Background(), dir, 64, 1, "q8")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := loadHFTokenizer(dir, *m.hf)
	if err != nil {
		t.Fatal(err)
	}
	s := defaultSampling()
	s.MaxTokens = 4
	want, err := Generate(context.Background(), m, tok, "hi", s)
	if err != nil {
		t.Fatal(err)
	}
	a, err := NewEngine(m, tok, EngineConfig{MaxBatch: 4, Queue: 4, CacheBytes: 1 << 20, KernelWorkers: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := NewEngine(m, tok, EngineConfig{MaxBatch: 4, Queue: 4, CacheBytes: 1 << 20, KernelWorkers: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	results := make(chan engineReply, 8)
	for i := 0; i < 8; i++ {
		e := a
		if i%2 != 0 {
			e = b
		}
		go func() {
			stream, err := e.Stream(context.Background(), "hi", s)
			if err != nil {
				results <- engineReply{Err: err}
				return
			}
			var ids []int
			for event := range stream.Events {
				ids = append(ids, event.TokenID)
			}
			got, err := stream.Wait()
			<-stream.Released()
			if !reflect.DeepEqual(ids, got.TokenIDs) {
				err = errors.New("stream token mismatch")
			}
			results <- engineReply{got, err}
		}()
	}
	for i := 0; i < 8; i++ {
		r := <-results
		if r.Err != nil || !reflect.DeepEqual(r.Completion, want) {
			t.Fatal("quantized engine result", r, want)
		}
	}
	a.Close()
	if a.Stats().WorkspaceFloatBytes != 0 || b.Stats().WorkspaceFloatBytes == 0 {
		t.Fatal("workspace ownership")
	}
}
