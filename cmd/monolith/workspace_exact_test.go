// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"testing"
)

func workspaceExactBuffers(w *backwardWorkspace) [][]float32 {
	return [][]float32{w.loss, w.residual, w.projection, w.temp, w.dq, w.dh, w.dg, w.du, w.dk, w.dv}
}

func workspaceExactDirty(w *backwardWorkspace, nan bool) {
	for buffer, values := range workspaceExactBuffers(w) {
		for i := range values {
			if nan {
				values[i] = math.Float32frombits(0x7fc00001)
			} else {
				values[i] = float32((i+buffer)%17+1) / 8
				if (i+buffer)%2 != 0 {
					values[i] = -values[i]
				}
			}
		}
	}
}

func workspaceExactBits(t testing.TB, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", label, len(got), len(want))
	}
	for i := range got {
		if !finite(float64(got[i])) || !finite(float64(want[i])) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: bits %08x != %08x (%.9g != %.9g)", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]), got[i], want[i])
		}
	}
}

func workspaceExactInputs(batch, seq, micro int) ([]int, []int) {
	// Repeated IDs exercise accumulation into the tied input/output embedding.
	tokens := []int{7, 7, 19, 7, 3, 19, 3}
	x, y := make([]int, batch*seq), make([]int, batch*seq)
	for row := 0; row < batch; row++ {
		for pos := 0; pos < seq; pos++ {
			x[row*seq+pos] = tokens[(pos+row+micro)%len(tokens)]
			y[row*seq+pos] = tokens[(pos+row+micro+1)%len(tokens)]
		}
		x[row*seq] = BOS
		y[(row+1)*seq-1] = EOS
	}
	return x, y
}

func workspaceExactModels(t testing.TB, kvHeads int) (*Model, *Model) {
	t.Helper()
	c := Config{Vocab: baseVocab, Dim: 16, Hidden: 23, Layers: 3, Heads: 4, KVHeads: kvHeads, Context: 128, RopeTheta: 10000, NormEps: 1e-5}
	baseline, err := NewModel(c, 73)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := NewModel(c, 73)
	if err != nil {
		t.Fatal(err)
	}
	return baseline, candidate
}

func workspaceExactMicrobatch(t testing.TB, baseline, candidate *Model, w *backwardWorkspace, x, y []int, scale float64, nan bool) {
	t.Helper()
	a, err := baseline.Forward(x, w.batch, w.seq)
	if err != nil {
		t.Fatal(err)
	}
	b, err := candidate.Forward(x, w.batch, w.seq)
	if err != nil {
		t.Fatal(err)
	}
	workspaceExactDirty(w, nan)
	want, err := baseline.Backward(a, y, scale)
	if err != nil {
		t.Fatal(err)
	}
	got, err := candidate.backward(b, y, scale, w)
	if err != nil {
		t.Fatal(err)
	}
	if !finite(got) || !finite(want) || math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("loss bits %016x != %016x", math.Float64bits(got), math.Float64bits(want))
	}
	for i, p := range candidate.Params {
		workspaceExactBits(t, p.Name+" gradient", p.Grad, baseline.Params[i].Grad)
	}
}

func TestBackwardWorkspaceExactGradients(t *testing.T) {
	previous := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(previous)
	for _, threads := range []int{1, 4} {
		runtime.GOMAXPROCS(threads)
		for _, kvHeads := range []int{1, 2, 4} { // MQA, GQA, MHA.
			for _, seq := range []int{5, 65} {
				for _, nan := range []bool{false, true} {
					dirt := "finite"
					if nan {
						dirt = "NaN"
					}
					t.Run(fmt.Sprintf("threads%d/kv%d/seq%d/%s", threads, kvHeads, seq, dirt), func(t *testing.T) {
						baseline, candidate := workspaceExactModels(t, kvHeads)
						baseline.ZeroGrad()
						candidate.ZeroGrad()
						// Nonzero initial gradients must survive every backward call.
						for i, p := range baseline.Params {
							for j := range p.Grad {
								value := float32((i+j)%13+1) / 1024
								p.Grad[j], candidate.Params[i].Grad[j] = value, value
							}
						}
						w, err := newBackwardWorkspace(candidate, 2, seq)
						if err != nil {
							t.Fatal(err)
						}
						// Both shapes have incomplete row/hidden tiles. seq=65 also
						// crosses the parallel threshold in linear and attention work.
						for micro := 0; micro < 3; micro++ {
							x, y := workspaceExactInputs(2, seq, micro)
							workspaceExactMicrobatch(t, baseline, candidate, w, x, y, 1.0/3, nan)
						}
					})
				}
			}
		}
	}
}

func TestBackwardWorkspaceExactOptimizerCheckpoints(t *testing.T) {
	previous := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(previous)
	for _, threads := range []int{1, 4} {
		runtime.GOMAXPROCS(threads)
		for _, kvHeads := range []int{1, 2, 4} {
			t.Run(fmt.Sprintf("threads%d/kv%d", threads, kvHeads), func(t *testing.T) {
				baseline, candidate := workspaceExactModels(t, kvHeads)
				spec := defaultTrainSpec()
				spec.Batch, spec.Seq, spec.Accum, spec.Steps, spec.Warmup = 2, 9, 3, 3, 1
				a := &TrainState{Spec: spec, RNG: RNG{79}, CorpusSHA256: corpusHash("workspace exact optimizer fixture")}
				b := *a
				w, err := newBackwardWorkspace(candidate, spec.Batch, spec.Seq)
				if err != nil {
					t.Fatal(err)
				}
				dir := t.TempDir()
				baselinePath, candidatePath := filepath.Join(dir, "baseline.mglm"), filepath.Join(dir, "workspace.mglm")
				for step := 1; step <= spec.Steps; step++ {
					baseline.ZeroGrad()
					candidate.ZeroGrad()
					for micro := 0; micro < spec.Accum; micro++ {
						x, y := workspaceExactInputs(spec.Batch, spec.Seq, step*spec.Accum+micro)
						workspaceExactMicrobatch(t, baseline, candidate, w, x, y, 1/float64(spec.Accum), (step+micro)%2 != 0)
					}
					wantNorm, err := baseline.AdamW(spec, step)
					if err != nil {
						t.Fatal(err)
					}
					gotNorm, err := candidate.AdamW(spec, step)
					if err != nil {
						t.Fatal(err)
					}
					if math.Float64bits(gotNorm) != math.Float64bits(wantNorm) {
						t.Fatal("optimizer gradient norm differs")
					}
					for i, p := range candidate.Params {
						ref := baseline.Params[i]
						workspaceExactBits(t, p.Name+" weights", p.Data, ref.Data)
						workspaceExactBits(t, p.Name+" first moment", p.M, ref.M)
						workspaceExactBits(t, p.Name+" second moment", p.V, ref.V)
					}
					a.Step, b.Step = step, step
					a.TokensSeen += int64(spec.Batch * spec.Seq * spec.Accum)
					b.TokensSeen = a.TokensSeen
					if err := SaveCheckpoint(baselinePath, baseline, Tokenizer{}, a); err != nil {
						t.Fatal(err)
					}
					if err := SaveCheckpoint(candidatePath, candidate, Tokenizer{}, &b); err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(checkpointBytes(t, baselinePath), checkpointBytes(t, candidatePath)) {
						t.Fatalf("complete checkpoint differs after optimizer step %d", step)
					}
				}
			})
		}
	}
}

// This isolates backward: model, forward tape, gradients, and workspace are
// initialized before timing. Reusing the fixed tape is benchmark-only; normal
// callers may consume a tape once. Gradients accumulate identically in both
// variants without an optimizer or gradient reset inside the measured loop.
func BenchmarkBackwardWorkspace(b *testing.B) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	for _, name := range []string{"tiny", "small"} {
		for _, reuse := range []bool{false, true} {
			kind := "allocating"
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
				m.ZeroGrad()
				x, y := workspaceExactInputs(2, 64, 0)
				tape, err := m.Forward(x, 2, 64)
				if err != nil {
					b.Fatal(err)
				}
				var w *backwardWorkspace
				var retained int
				if reuse {
					w, err = newBackwardWorkspace(m, 2, 64)
					if err != nil {
						b.Fatal(err)
					}
					for _, values := range workspaceExactBuffers(w) {
						retained += cap(values) * 4
					}
				}
				if _, err = m.backward(tape, y, 1, w); err != nil {
					b.Fatal(err)
				}
				m.ZeroGrad()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					tape.Used = false
					if _, err = m.backward(tape, y, 1, w); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				// Only workspace scratch retention, excluding model/tape/Grad.
				b.ReportMetric(float64(retained), "workspace-retained-B")
			})
		}
	}
}

func BenchmarkBackwardWorkspaceConstruction(b *testing.B) {
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
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w, err := newBackwardWorkspace(m, 2, 64)
				if err != nil {
					b.Fatal(err)
				}
				runtime.KeepAlive(w)
			}
			b.StopTimer()
			b.ReportMetric(float64(4*backwardWorkspaceElements(c, 2, 64)), "workspace-retained-B")
		})
	}
}
