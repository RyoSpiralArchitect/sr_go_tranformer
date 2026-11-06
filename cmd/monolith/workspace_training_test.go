// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func TestWorkspaceDatasetCanceledAccumulationReuse(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "abcdefghijklmnopqrstuvwxyz\n", "held\n", 40)
	m, reference := testModel(t), testModel(t)
	s, control := datasetTestState(d), datasetTestState(d)
	ensureMoments(m)
	ensureMoments(reference)
	w, err := newBackwardWorkspace(m, s.Spec.Batch, s.Spec.Seq)
	if err != nil {
		t.Fatal(err)
	}
	prefetch := func(ctx context.Context, state *TrainState) *TokenPrefetch {
		t.Helper()
		p, err := NewTokenPrefetch(ctx, d, "train", PrefetchConfig{Batch: state.Spec.Batch, Seq: state.Spec.Seq, Workers: 1, Depth: 1, Seed: state.Dataset.OrderSeed, Shuffle: true}, state.Dataset.Cursor)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	update := func(model *Model, state *TrainState, workspace *backwardWorkspace) {
		t.Helper()
		p := prefetch(context.Background(), state)
		defer p.Close()
		if _, _, err := observedDatasetUpdate(context.Background(), model, state, p, nil, workspace); err != nil {
			t.Fatal(err)
		}
	}
	update(m, s, w)
	update(reference, control, nil)
	dir := t.TempDir()
	before, after, expected := filepath.Join(dir, "before"), filepath.Join(dir, "after"), filepath.Join(dir, "expected")
	save := func(path string, model *Model, state *TrainState) []byte {
		t.Helper()
		if err := SaveCheckpoint(path, model, Tokenizer{}, state); err != nil {
			t.Fatal(err)
		}
		return checkpointBytes(t, path)
	}
	prefix := save(before, m, s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var reads atomic.Int64
	read := func(ctx context.Context, shard int, offset int64, dst []int) error {
		if reads.Add(1) == 2 {
			cancel()
			return ctx.Err()
		}
		return d.ReadTokens(ctx, shard, offset, dst)
	}
	p, err := newTokenPrefetch(ctx, d, "train", PrefetchConfig{Batch: s.Spec.Batch, Seq: s.Spec.Seq, Workers: 1, Depth: 1, Seed: s.Dataset.OrderSeed, Shuffle: true}, s.Dataset.Cursor, read)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = observedDatasetUpdate(ctx, m, s, p, nil, w)
	p.Close()
	if !errors.Is(err, context.Canceled) || reads.Load() != 2 {
		t.Fatalf("did not cancel within accumulation: %v, reads=%d", err, reads.Load())
	}
	if !bytes.Equal(prefix, save(after, m, s)) {
		t.Fatal("cancellation changed the committed checkpoint")
	}
	// Reuse the same model and dirty workspace with a fresh depth-one lease.
	// The next update must discard partial gradients and replay from its cursor.
	for s.Step < s.Spec.Steps {
		update(m, s, w)
		update(reference, control, nil)
	}
	if !bytes.Equal(save(after, m, s), save(expected, reference, control)) {
		t.Fatal("same-workspace retry differs from uninterrupted allocating control")
	}
}
