// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestDatasetEvaluationReportMatchesTraining(t *testing.T) {
	path, d := preparedDataset(t, Tokenizer{}, "first\nsecond\nthird\n", "held out\nlast", 14)
	m := testModel(t)
	model := filepath.Join(t.TempDir(), "model.mglm")
	if err := SaveCheckpoint(model, m, Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	for _, limit := range []int{0, 1, 2, 100} {
		o := evaluationOptions{Model: model, Dataset: path, Seq: 5, MaxBatches: limit}
		r, err := evaluationReport(context.Background(), o)
		if err != nil {
			t.Fatal(err)
		}
		loss, count, err := evaluateDataset(context.Background(), m, d, 5, limit)
		if err != nil {
			t.Fatal(err)
		}
		if r.Loss != loss || r.Targets != count || r.TargetRange != (TargetRange{0, count}) {
			t.Fatal("CLI/training evaluation mismatch", r, loss, count)
		}
		if r.Source != (EvaluationSource{Kind: "dataset", SHA256: d.id, Split: "validation"}) || r.Scoring != "packed-shards-bos-window-v1" || r.TokenizerSHA256 != tokenizerHash(Tokenizer{}) {
			t.Fatal("incomplete evaluation identity", r)
		}
		if r.CheckpointSHA256 != corpusHash(string(checkpointBytes(t, model))) {
			t.Fatal("wrong complete checkpoint digest")
		}
		again, err := evaluationReport(context.Background(), o)
		if err != nil || !reflect.DeepEqual(r, again) {
			t.Fatal("repeated receipt changed", err)
		}
	}
	r, err := evaluationReport(context.Background(), evaluationOptions{Model: model, Dataset: path, Split: "train", Seq: 5, MaxBatches: 2})
	if err != nil {
		t.Fatal(err)
	}
	loss, count, err := evaluateDatasetSplit(context.Background(), m, d, "train", 5, 2)
	if err != nil || r.Loss != loss || r.Targets != count || r.Source.Split != "train" {
		t.Fatal("train split report mismatch", r, err)
	}
}

func TestEvaluationTextReceiptAndCLI(t *testing.T) {
	dir := t.TempDir()
	model, data := filepath.Join(dir, "model"), filepath.Join(dir, "data")
	m := testModel(t)
	text := "independent held-out text\n"
	if err := SaveCheckpoint(model, m, Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(data, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := evaluationReport(context.Background(), evaluationOptions{Model: model, Data: data, Seq: 4, MaxBatches: 2})
	if err != nil {
		t.Fatal(err)
	}
	loss, n, err := evaluate(context.Background(), m, Tokenizer{}.Encode(text, true, true), 4, 2)
	if err != nil || r.Loss != loss || r.Targets != n || r.Source.SHA256 != corpusHash(text) || r.Scoring != "text-bos-window-v1" {
		t.Fatal("text scoring changed", r, err)
	}
	// Real CLI output keeps the original loss/perplexity/targets fields.
	out, err := os.CreateTemp(dir, "stdout")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	old := os.Stdout
	os.Stdout = out
	defer func() { os.Stdout = old }()
	err = runEval(context.Background(), []string{"-model", model, "-data", data, "-seq", "4", "-batches", "2"})
	os.Stdout = old
	if err != nil {
		t.Fatal(err)
	}
	if _, err = out.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	var got EvaluationReport
	if err = json.NewDecoder(out).Decode(&got); err != nil || !reflect.DeepEqual(got, r) {
		t.Fatal("CLI receipt mismatch", got, err)
	}
}

func TestCheckpointIdentitySurvivesAtomicReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "checkpoint")
	a := testModel(t)
	if err := SaveCheckpoint(path, a, Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	original := checkpointBytes(t, path)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b := testModel(t)
	b.Params[0].Data[0] += 1
	if err = SaveCheckpoint(path, b, Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	loaded, _, _, digest, err := loadCheckpointFile(f)
	if err != nil {
		t.Fatal(err)
	}
	if digest != corpusHash(string(original)) || loaded.Params[0].Data[0] != a.Params[0].Data[0] {
		t.Fatal("receipt refers to replacement instead of loaded model")
	}
	if bytes.Equal(original, checkpointBytes(t, path)) {
		t.Fatal("test did not replace checkpoint")
	}
}

func TestEvaluationRejectsMismatchesAndInvalidInputs(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	tok := Tokenizer{Merges: []Pair{{'a', 'b'}}}
	c := testModel(t).Config
	c.Vocab = tok.Vocab()
	m, err := NewModel(c, 7)
	if err != nil {
		t.Fatal(err)
	}
	if err = SaveCheckpoint(model, m, tok, nil); err != nil {
		t.Fatal(err)
	}
	path, d := preparedDataset(t, Tokenizer{Merges: []Pair{{'b', 'c'}}}, "abc\n", "cab\n", 16)
	for _, o := range []evaluationOptions{
		{Model: model, Dataset: path}, // Same vocabulary size, different merges.
		{Model: model},
		{Model: model, Dataset: path, Data: "unused"},
		{Model: model, Data: "unused", Split: "validation"},
		{Model: model, Dataset: path, Split: "unknown"},
		{Model: model, Dataset: path, Seq: -1},
		{Model: model, Dataset: path, MaxBatches: -1},
	} {
		if _, err := evaluationReport(context.Background(), o); err == nil {
			t.Fatal("invalid evaluation accepted", o)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = evaluationReport(ctx, evaluationOptions{Model: model, Dataset: path}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	// Valid tokenizer cannot make corrupt shards acceptable.
	if err = SaveCheckpoint(model, m, d.manifest.Tokenizer, nil); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(d.root, d.manifest.Shards[0].File)
	if err = os.Truncate(file, shardHeaderBytes); err != nil {
		t.Fatal(err)
	}
	if _, err = evaluationReport(context.Background(), evaluationOptions{Model: model, Dataset: path}); err == nil {
		t.Fatal("corrupt dataset evaluated")
	}
}
