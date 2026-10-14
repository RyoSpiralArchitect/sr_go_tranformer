// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"testing"
)

// Cancel at a deterministic serial boundary without relying on wall-clock
// scheduling. Kernels do not consult this context from their worker goroutines.
type evaluationCancelAtBoundary struct {
	context.Context
	cancel context.CancelFunc
	calls  int
}

func (c *evaluationCancelAtBoundary) Err() error {
	c.calls++
	if c.calls == 3 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestEvaluationForwardCanceledWindowCanReuseWorkspace(t *testing.T) {
	m := testModel(t)
	m.ZeroGrad()
	ensureMoments(m)
	before := evaluationBoundarySnapshot(evaluationBoundaryModelBuffers(m))
	w, err := newEvaluationWorkspace(m, 2, 7)
	if err != nil {
		t.Fatal(err)
	}
	evaluationExactDirty(w, true)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stopping := &evaluationCancelAtBoundary{Context: ctx, cancel: cancel}
	x, _ := workspaceExactInputs(2, 7, 0)
	logits, err := m.forwardEvaluation(stopping, x, 2, 7, w)
	if !errors.Is(err, context.Canceled) || logits != nil || stopping.calls != 3 {
		t.Fatalf("canceled window returned logits or wrong status: %v, calls=%d", err, stopping.calls)
	}
	// The first layer ran, so this is a partially written workspace, rather
	// than the pre-canceled rejection covered by boundary tests.
	for _, value := range w.x {
		if !finite(float64(value)) {
			t.Fatal("cancellation did not reach a populated residual buffer")
		}
	}
	evaluationBoundaryUnchanged(t, "model", evaluationBoundaryModelBuffers(m), before)
	x, _ = workspaceExactInputs(1, 3, 2)
	reference, err := m.Forward(x, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	logits, err = m.forwardEvaluation(context.Background(), x, 1, 3, w)
	if err != nil {
		t.Fatal(err)
	}
	workspaceExactBits(t, "retry logits", logits, reference.Logits)
	evaluationBoundaryUnchanged(t, "model", evaluationBoundaryModelBuffers(m), before)
}
