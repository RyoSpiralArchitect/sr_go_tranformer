// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLocalModelPathSelection(t *testing.T) {
	model := hfCLIFixture(t)
	alias := filepath.Join(t.TempDir(), "model 猫 with spaces")
	if err := os.Symlink(model, alias); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{model, alias, filepath.Join(model, "config.json"), filepath.Join(model, "model.safetensors")} {
		dir, checkpoint, err := localModelPath(value)
		want := model
		if value == alias {
			want = alias
		}
		if err != nil || dir != want || checkpoint != "" {
			t.Fatal(value, dir, checkpoint, err)
		}
	}
	// The index is authoritative: choosing a stray shard must never silently
	// load an unrelated model.safetensors from the same folder.
	hfWriteJSON(t, filepath.Join(model, "model.safetensors.index.json"), map[string]any{"weight_map": map[string]string{"weight": "part-01.safetensors"}})
	shard := filepath.Join(model, "part-01.safetensors")
	if err := os.WriteFile(shard, []byte("synthetic shard"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{shard, filepath.Join(model, "model.safetensors.index.json")} {
		dir, checkpoint, err := localModelPath(value)
		if err != nil || dir != model || checkpoint != "" {
			t.Fatal(dir, checkpoint, err)
		}
	}
	if _, _, err := localModelPath(filepath.Join(model, "model.safetensors")); err == nil || !strings.Contains(err.Error(), "not referenced") {
		t.Fatal("stray shard accepted", err)
	}
	native := filepath.Join(t.TempDir(), "checkpoint with no extension")
	if err := SaveCheckpoint(native, testModel(t), Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	dir, checkpoint, err := localModelPath(native)
	if err != nil || dir != "" || checkpoint != native {
		t.Fatal(dir, checkpoint, err)
	}
	userHome, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{"~", "~/.", "~/"} {
		dir, checkpoint, err := localModelPath(input)
		if err != nil || dir != filepath.Clean(userHome) || checkpoint != "" {
			t.Fatal(input, dir, checkpoint, err)
		}
	}
}

func TestLocalModelPathRejectsUnsupportedFiles(t *testing.T) {
	for _, name := range []string{"weights.gguf", "tokenizer.json", "unknown.mglm", "stray.safetensors"} {
		path := filepath.Join(t.TempDir(), name)
		if err := os.WriteFile(path, []byte("not an archive"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, _, err := localModelPath(path); err == nil {
			t.Fatal("unsupported path accepted", name)
		}
	}
	if _, _, err := localModelPath("https://huggingface.co/author/model"); err == nil {
		t.Fatal("URL accepted")
	}
	if _, _, err := localModelPath(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("missing path accepted")
	}
}

func TestLocalModelPathCLIIsOffline(t *testing.T) {
	t.Setenv("HF_HUB_OFFLINE", "1")
	denyHFNetwork(t)
	model := hfCLIFixture(t)
	native := filepath.Join(t.TempDir(), "model.mglm")
	if err := SaveCheckpoint(native, testModel(t), Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"-path", model, "-context", "32", "-weight-format", "q4-mixed", "check", "-chat"},
		{"-path", filepath.Join(model, "config.json"), "tokens", "-prompt", "hello 猫"},
		{"-path", filepath.Join(model, "model.safetensors"), "inspect"},
		{"-path", native, "inspect"},
	} {
		if err := runInferLocal(context.Background(), args); err != nil {
			t.Fatal(args, err)
		}
	}
	for _, args := range [][]string{
		{"-path", model, "-dir", model, "check"},
		{"-path", model, "-repo", "author/toy", "check"},
		{"-path", model, "-checkpoint", native, "check"},
		{"-path", model, "-cache-dir", t.TempDir(), "check"},
		{"-path", model, "-download-timeout", "1h", "check"},
		{"-path", model, "fetch"},
		{"-path", native, "tokens"},
		{"-path", native, "-weight-format", "q8", "inspect"},
	} {
		if err := runInferLocal(context.Background(), args); err == nil {
			t.Fatal("invalid path source accepted", args)
		}
	}
	// Action preflight must win before inspecting the selected local path.
	err := runInferLocal(context.Background(), []string{"-path", filepath.Join(t.TempDir(), "missing"), "generate", "-max-tokens", "0"})
	if err == nil || strings.Contains(err.Error(), "no such file") {
		t.Fatal("path read before action validation", err)
	}
}
