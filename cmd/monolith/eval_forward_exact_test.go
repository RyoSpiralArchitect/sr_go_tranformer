// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
)

func evaluationExactBuffers(w *evaluationWorkspace) [][]float32 {
	return [][]float32{w.x, w.norm, w.q, w.k, w.v, w.attention, w.gate, w.up, w.hidden, w.temp, w.logits, w.scores}
}

func evaluationExactDirty(w *evaluationWorkspace, nan bool) {
	for buffer, values := range evaluationExactBuffers(w) {
		for i := range values {
			if nan {
				values[i] = math.Float32frombits(0x7fc00001)
			} else {
				values[i] = float32((i+buffer)%19+1) / 16
				if (i+buffer)%2 != 0 {
					values[i] = -values[i]
				}
			}
		}
	}
}

func TestEvaluationForwardExactDirtyReuse(t *testing.T) {
	previous := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(previous)
	for _, threads := range []int{1, 4} {
		runtime.GOMAXPROCS(threads)
		for _, kvHeads := range []int{1, 2, 4} { // MQA, GQA, MHA.
			for _, nan := range []bool{false, true} {
				dirt := "finite"
				if nan {
					dirt = "NaN"
				}
				t.Run(fmt.Sprintf("threads%d/kv%d/%s", threads, kvHeads, dirt), func(t *testing.T) {
					m, _ := workspaceExactModels(t, kvHeads)
					w, err := newEvaluationWorkspace(m, 2, 65)
					if err != nil {
						t.Fatal(err)
					}
					initialBuffers := evaluationExactBuffers(w)
					// Both full and tail shapes have incomplete row/output tiles.
					// The full shape also dispatches linear and attention workers.
					for call, shape := range [][2]int{{2, 65}, {1, 5}, {2, 3}, {2, 65}} {
						batch, seq := shape[0], shape[1]
						x, y := workspaceExactInputs(batch, seq, call)
						tape, err := m.Forward(x, batch, seq)
						if err != nil {
							t.Fatal(err)
						}
						want, _, err := CrossEntropy(tape.Logits, y, m.Config.Vocab, 1, false)
						if err != nil {
							t.Fatal(err)
						}
						evaluationExactDirty(w, nan)
						logits, err := m.forwardEvaluation(context.Background(), x, batch, seq, w)
						if err != nil {
							t.Fatal(err)
						}
						workspaceExactBits(t, "evaluation logits", logits, tape.Logits)
						if cap(logits) != len(logits) || &logits[0] != &w.logits[0] {
							t.Fatal("logits must borrow only the active workspace region")
						}
						got, grad, err := CrossEntropy(logits, y, m.Config.Vocab, 1, false)
						if err != nil {
							t.Fatal(err)
						}
						if grad != nil || !finite(got) || math.Float64bits(got) != math.Float64bits(want) {
							t.Fatalf("evaluation loss bits %016x != %016x, gradient=%v", math.Float64bits(got), math.Float64bits(want), grad != nil)
						}
						for i, values := range evaluationExactBuffers(w) {
							if len(values) != len(initialBuffers[i]) || cap(values) != cap(initialBuffers[i]) || &values[0] != &initialBuffers[i][0] {
								t.Fatalf("buffer %d changed maximum storage after active shape %v", i, shape)
							}
						}
					}
				})
			}
		}
	}
}

func TestEvaluationForwardPreservesModelAndTrainingTapes(t *testing.T) {
	m := testModel(t)
	m.ZeroGrad()
	ensureMoments(m)
	m.Revision = 17
	before := make([][][]float32, len(m.Params))
	for i, p := range m.Params {
		for j := range p.Grad {
			p.Grad[j] = float32((i+j)%13-6) / 32
			p.M[j] = float32((i+j)%11-5) / 64
			p.V[j] = float32((i+j)%7+1) / 128
		}
		for _, values := range [][]float32{p.Data, p.Grad, p.M, p.V} {
			before[i] = append(before[i], append([]float32(nil), values...))
		}
	}
	cos, sin := append([]float32(nil), m.Cos...), append([]float32(nil), m.Sin...)
	var tapes []*Tape
	var snapshots []workspaceBoundaryTapeSnapshot
	for i := 0; i < 2; i++ {
		x, _ := workspaceExactInputs(1, 7+i, i)
		tape, err := m.Forward(x, 1, 7+i)
		if err != nil {
			t.Fatal(err)
		}
		tapes = append(tapes, tape)
		snapshots = append(snapshots, snapshotWorkspaceBoundaryTape(tape))
	}
	w, err := newEvaluationWorkspace(m, 2, 11)
	if err != nil {
		t.Fatal(err)
	}
	for call, shape := range [][2]int{{2, 11}, {1, 3}, {2, 11}} {
		x, _ := workspaceExactInputs(shape[0], shape[1], call)
		if _, err := m.forwardEvaluation(context.Background(), x, shape[0], shape[1], w); err != nil {
			t.Fatal(err)
		}
		if m.Revision != 17 {
			t.Fatal("evaluation changed model revision")
		}
		for i, p := range m.Params {
			for j, values := range [][]float32{p.Data, p.Grad, p.M, p.V} {
				workspaceExactBits(t, p.Name+" state", values, before[i][j])
			}
		}
		workspaceExactBits(t, "RoPE cosine", m.Cos, cos)
		workspaceExactBits(t, "RoPE sine", m.Sin, sin)
		for i, tape := range tapes {
			snapshots[i].check(t, tape)
			if tape.Used || tape.Model != m || tape.Revision != 17 {
				t.Fatal("evaluation changed retained training tape ownership or lifecycle")
			}
		}
	}
}

func TestEvaluationForwardConcurrentIndependentWorkspaces(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	m, _ := workspaceExactModels(t, 2)
	type call struct {
		x, y       []int
		batch, seq int
		logits     []float32
		loss       float64
	}
	const workers = 4
	workspaces := make([]*evaluationWorkspace, workers)
	calls := make([][]call, workers)
	for worker := 0; worker < workers; worker++ {
		var err error
		workspaces[worker], err = newEvaluationWorkspace(m, 2, 65)
		if err != nil {
			t.Fatal(err)
		}
		for i, shape := range [][2]int{{2, 65}, {1, 5}, {2, 65}} {
			x, y := workspaceExactInputs(shape[0], shape[1], worker+i)
			tape, err := m.Forward(x, shape[0], shape[1])
			if err != nil {
				t.Fatal(err)
			}
			loss, _, err := CrossEntropy(tape.Logits, y, m.Config.Vocab, 1, false)
			if err != nil {
				t.Fatal(err)
			}
			calls[worker] = append(calls[worker], call{x: x, y: y, batch: shape[0], seq: shape[1], logits: tape.Logits, loss: loss})
		}
	}
	errors := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		worker := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := workspaces[worker]
			for i, c := range calls[worker] {
				evaluationExactDirty(w, i%2 != 0)
				got, err := m.forwardEvaluation(context.Background(), c.x, c.batch, c.seq, w)
				if err != nil {
					errors <- err
					return
				}
				if len(got) != len(c.logits) {
					errors <- fmt.Errorf("worker %d changed logit shape", worker)
					return
				}
				for j := range got {
					if math.Float32bits(got[j]) != math.Float32bits(c.logits[j]) {
						errors <- fmt.Errorf("worker %d call %d logit %d differs", worker, i, j)
						return
					}
				}
				loss, _, err := CrossEntropy(got, c.y, m.Config.Vocab, 1, false)
				if err != nil || math.Float64bits(loss) != math.Float64bits(c.loss) {
					errors <- fmt.Errorf("worker %d changed loss: %v", worker, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

// Initialization is outside timing. The allocating case keeps the ordinary
// backward tape API; the warm case borrows logits from evaluation-owned scratch.
// Report retained FP32 payload separately from per-call allocation volume.
func BenchmarkEvaluationForward(b *testing.B) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	for _, name := range []string{"tiny", "small"} {
		for _, reuse := range []bool{false, true} {
			kind := "allocating_tape"
			if reuse {
				kind = "warm_workspace"
			}
			b.Run(name+"/threads4/"+kind, func(b *testing.B) {
				c, err := preset(name)
				if err != nil {
					b.Fatal(err)
				}
				m, err := NewModel(c, 73)
				if err != nil {
					b.Fatal(err)
				}
				x, _ := workspaceExactInputs(1, 64, 0)
				var w *evaluationWorkspace
				var retained int64
				if reuse {
					w, err = newEvaluationWorkspace(m, 1, 64)
					if err != nil {
						b.Fatal(err)
					}
					retained = 4 * evaluationWorkspaceElements(c, 1, 64)
					if _, err = m.forwardEvaluation(context.Background(), x, 1, 64, w); err != nil {
						b.Fatal(err)
					}
				}
				var logits []float32
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if reuse {
						logits, err = m.forwardEvaluation(context.Background(), x, 1, 64, w)
					} else {
						var tape *Tape
						tape, err = m.Forward(x, 1, 64)
						if err == nil {
							logits = tape.Logits
						}
					}
					if err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				b.ReportMetric(float64(retained), "retained-B")
				runtime.KeepAlive(logits)
			})
		}
	}
}

func BenchmarkEvaluationWorkspaceConstruction(b *testing.B) {
	for _, name := range []string{"tiny", "small"} {
		b.Run(name, func(b *testing.B) {
			c, err := preset(name)
			if err != nil {
				b.Fatal(err)
			}
			m, err := NewModel(c, 73)
			if err != nil {
				b.Fatal(err)
			}
			var w *evaluationWorkspace
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w, err = newEvaluationWorkspace(m, 1, 64)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(4*evaluationWorkspaceElements(c, 1, 64)), "payload-B")
			runtime.KeepAlive(w)
		})
	}
}
