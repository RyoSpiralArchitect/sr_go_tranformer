// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"testing"
)

func TestLongDecodeCrossesAttentionLaneThreshold(t *testing.T) {
	c := Config{Vocab: 258, Dim: 512, Hidden: 100, Heads: 8, KVHeads: 2, Layers: 1, Context: 288, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	serial, parallel := newKernelExecutor(1), newKernelExecutor(4)
	defer serial.close()
	defer parallel.close()
	a, b := m.NewCache(), m.NewCache()
	defer a.Close()
	defer b.Close()
	var x, y decodeWorkspace
	parallelCalls := 0
	for step := 0; step < 272; step++ {
		chunks := [][]int{{(step*17 + 3) % 256}}
		got, err := m.prefillBorrowed(context.Background(), []*KVCache{a}, chunks, parallel, &x)
		if err != nil {
			t.Fatal(err)
		}
		want, err := m.prefillBorrowed(context.Background(), []*KVCache{b}, chunks, serial, &y)
		if err != nil {
			t.Fatal(err)
		}
		exactWorkerFloats(t, got, want)
		if planInferenceAttention(parallel, c, 1, step+1).lanes > 1 {
			parallelCalls++
		}
		assertWorkspaceReferencesCleared(t, &x)
	}
	if parallelCalls != 17 {
		t.Fatalf("multi-lane calls=%d, want 17 across the 256-token threshold", parallelCalls)
	}
	for i, page := range a.Pages {
		exactWorkerFloats(t, page.Data, b.Pages[i].Data)
	}
}
