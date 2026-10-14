// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
)

func evaluationBoundaryBuffers(w *evaluationWorkspace) [][]float32 {
	return [][]float32{w.x, w.norm, w.q, w.k, w.v, w.attention, w.gate, w.up, w.hidden, w.temp, w.logits, w.scores}
}

func evaluationBoundaryModelBuffers(m *Model) [][]float32 {
	values := [][]float32{m.Cos, m.Sin}
	for _, p := range m.Params {
		values = append(values, p.Data, p.Grad, p.M, p.V)
	}
	return values
}

func evaluationBoundarySnapshot(buffers [][]float32) [][]float32 {
	copyOf := make([][]float32, len(buffers))
	for i, b := range buffers {
		copyOf[i] = append([]float32(nil), b...)
	}
	return copyOf
}

func evaluationBoundaryUnchanged(t *testing.T, name string, buffers, before [][]float32) {
	t.Helper()
	for i, b := range buffers {
		workspaceExactBits(t, fmt.Sprintf("%s[%d]", name, i), b, before[i])
	}
}

func TestEvaluationForwardRejectsBoundaryBeforeMutation(t *testing.T) {
	type request struct {
		ctx        context.Context
		workspace  *evaluationWorkspace
		ids        []int
		batch, seq int
	}
	type mutation func(*request, *Model)
	cases := map[string]mutation{
		"nil_workspace": func(r *request, _ *Model) { r.workspace = nil },
		"nil_context":   func(r *request, _ *Model) { r.ctx = nil },
		"canceled_context": func(r *request, _ *Model) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			r.ctx = ctx
		},
		"nil_owner":        func(r *request, _ *Model) { r.workspace.model = nil },
		"foreign_owner":    func(r *request, _ *Model) { r.workspace.model = &Model{Config: r.workspace.config} },
		"workspace_config": func(r *request, _ *Model) { r.workspace.config.NormEps *= 2 },
		"model_config":     func(_ *request, m *Model) { m.Config.RopeTheta *= 2 },
		"max_batch_zero":   func(r *request, _ *Model) { r.workspace.maxBatch = 0 },
		"max_batch_large":  func(r *request, _ *Model) { r.workspace.maxBatch = 4097 },
		"max_seq_zero":     func(r *request, _ *Model) { r.workspace.maxSeq = 0 },
		"max_seq_large":    func(r *request, m *Model) { r.workspace.maxSeq = m.Config.Context + 1 },
		"max_shape_storage": func(r *request, _ *Model) {
			r.workspace.maxBatch++
		},
		"batch_zero":     func(r *request, _ *Model) { r.batch = 0; r.ids = nil },
		"batch_negative": func(r *request, _ *Model) { r.batch = -1 },
		"batch_over_max": func(r *request, _ *Model) { r.batch = 3; r.ids = make([]int, 9) },
		"seq_zero":       func(r *request, _ *Model) { r.seq = 0; r.ids = nil },
		"seq_negative":   func(r *request, _ *Model) { r.seq = -1 },
		"seq_over_max":   func(r *request, _ *Model) { r.seq = 4; r.ids = make([]int, 8) },
		"ids_short":      func(r *request, _ *Model) { r.ids = r.ids[:5] },
		"ids_long":       func(r *request, _ *Model) { r.ids = append(r.ids, 7) },
		// Put bad IDs last so a one-pass validate-and-copy implementation would
		// already have overwritten the start of x before noticing the error.
		"negative_last_id": func(r *request, _ *Model) { r.ids[len(r.ids)-1] = -1 },
		"large_last_id":    func(r *request, m *Model) { r.ids[len(r.ids)-1] = m.Config.Vocab },
	}
	for name, selectBuffer := range map[string]func(*evaluationWorkspace) *[]float32{
		"x": func(w *evaluationWorkspace) *[]float32 { return &w.x }, "norm": func(w *evaluationWorkspace) *[]float32 { return &w.norm },
		"q": func(w *evaluationWorkspace) *[]float32 { return &w.q }, "k": func(w *evaluationWorkspace) *[]float32 { return &w.k },
		"v": func(w *evaluationWorkspace) *[]float32 { return &w.v }, "attention": func(w *evaluationWorkspace) *[]float32 { return &w.attention },
		"gate": func(w *evaluationWorkspace) *[]float32 { return &w.gate }, "up": func(w *evaluationWorkspace) *[]float32 { return &w.up },
		"hidden": func(w *evaluationWorkspace) *[]float32 { return &w.hidden }, "temp": func(w *evaluationWorkspace) *[]float32 { return &w.temp },
		"logits": func(w *evaluationWorkspace) *[]float32 { return &w.logits }, "scores": func(w *evaluationWorkspace) *[]float32 { return &w.scores },
	} {
		selectBuffer := selectBuffer
		cases[name+"/nil"] = func(r *request, _ *Model) { *selectBuffer(r.workspace) = nil }
		cases[name+"/short_length"] = func(r *request, _ *Model) {
			p := selectBuffer(r.workspace)
			*p = (*p)[:len(*p)-1]
		}
		cases[name+"/long_length"] = func(r *request, _ *Model) {
			p := selectBuffer(r.workspace)
			*p = make([]float32, len(*p)+1)
		}
		cases[name+"/extra_capacity"] = func(r *request, _ *Model) {
			p := selectBuffer(r.workspace)
			*p = make([]float32, len(*p), len(*p)+1)
		}
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			m := testModel(t)
			m.ZeroGrad()
			for _, p := range m.Params {
				p.M, p.V = make([]float32, len(p.Data)), make([]float32, len(p.Data))
				for i := range p.Data {
					p.Grad[i], p.M[i], p.V[i] = float32(i%7-3)/8, float32(i%5)/4, float32(i%3)/2
				}
			}
			w, err := newEvaluationWorkspace(m, 2, 3)
			if err != nil {
				t.Fatal(err)
			}
			for j, buffer := range evaluationBoundaryBuffers(w) {
				for i := range buffer {
					buffer[i] = float32((i+j)%17+1) / 8
				}
			}
			r := request{context.Background(), w, []int{BOS, 7, 3, BOS, 9, 2}, 2, 3}
			mutate(&r, m)
			modelBefore := evaluationBoundarySnapshot(evaluationBoundaryModelBuffers(m))
			workspaceBefore := evaluationBoundarySnapshot(evaluationBoundaryBuffers(w))
			configBefore, revisionBefore := m.Config, m.Revision
			logits, err := m.forwardEvaluation(r.ctx, r.ids, r.batch, r.seq, r.workspace)
			if err == nil || logits != nil {
				t.Fatal("rejected evaluation must return an error and no logits")
			}
			evaluationBoundaryUnchanged(t, "model", evaluationBoundaryModelBuffers(m), modelBefore)
			evaluationBoundaryUnchanged(t, "workspace", evaluationBoundaryBuffers(w), workspaceBefore)
			if m.Config != configBefore || m.Revision != revisionBefore {
				t.Fatal("rejected evaluation changed model configuration or revision")
			}
		})
	}
}

func TestEvaluationWorkspaceConstructorBounds(t *testing.T) {
	m := testModel(t)
	for _, shape := range [][2]int{{0, 1}, {-1, 1}, {4097, 1}, {1, 0}, {1, -1}, {1, m.Config.Context + 1}} {
		if w, err := newEvaluationWorkspace(m, shape[0], shape[1]); err == nil || w != nil {
			t.Fatalf("accepted constructor shape %v", shape)
		}
	}
	if w, err := newEvaluationWorkspace(nil, 1, 1); err == nil || w != nil {
		t.Fatal("accepted nil model")
	}
	bad := m.Config
	bad.Heads = 0
	if w, err := newEvaluationWorkspace(&Model{Config: bad}, 1, 1); err == nil || w != nil {
		t.Fatal("accepted invalid model configuration")
	}
	// No parameter allocation is needed to reject this valid model geometry.
	c := Config{Vocab: baseVocab, Dim: 2, Hidden: 4, Layers: 1, Heads: 1, KVHeads: 1, Context: 32768, RopeTheta: 10000, NormEps: 1e-5}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if evaluationWorkspaceElements(c, 16, c.Context) <= maxActivationElements {
		t.Fatal("fixture must exceed the standalone workspace guard")
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	w, err := newEvaluationWorkspace(&Model{Config: c}, 16, c.Context)
	runtime.ReadMemStats(&after)
	if err == nil || w != nil {
		t.Fatal("accepted evaluation workspace beyond its allocation guard")
	}
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated >= 1<<20 {
		t.Fatalf("rejected constructor allocated %d bytes before returning", allocated)
	}
}

func TestEvaluationWorkspaceFitsRetainedTrainerBound(t *testing.T) {
	configs := []Config{testModel(t).Config,
		{Vocab: 65536, Dim: 2, Hidden: 1, Layers: 1, Heads: 1, KVHeads: 1, Context: 32768, RopeTheta: 10000, NormEps: 1e-5},
		{Vocab: baseVocab, Dim: 512, Hidden: 1, Layers: 1, Heads: 256, KVHeads: 256, Context: 128, RopeTheta: 10000, NormEps: 1e-5},
	}
	for _, name := range []string{"demo", "tiny", "small", "base"} {
		c, err := preset(name)
		if err != nil {
			t.Fatal(err)
		}
		configs = append(configs, c)
	}
	for _, c := range configs {
		if err := c.Validate(); err != nil {
			t.Fatal(err)
		}
		for _, seq := range []int{1, c.Context} {
			for _, batch := range []int{1, 2, 17} {
				// Training retains its backward workspace during validation. The
				// batch=1 validation payload must fit inside the allocation already
				// reserved for the training tape plus transient backward arrays.
				eval, active := evaluationWorkspaceElements(c, 1, seq), activationElements(c, batch, seq)
				retained := backwardWorkspaceElements(c, batch, seq)
				if eval > active {
					t.Fatalf("evaluation payload %d exceeds training activation bound %d", eval, active)
				}
				if retained+active <= maxActivationElements && retained+eval > maxActivationElements {
					t.Fatal("accepted trainer cannot retain its scratch during validation")
				}
			}
		}
	}
	w, err := newEvaluationWorkspace(testModel(t), 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	var actual int64
	for _, buffer := range evaluationBoundaryBuffers(w) {
		actual += int64(cap(buffer))
	}
	if actual != evaluationWorkspaceElements(w.config, w.maxBatch, w.maxSeq) {
		t.Fatalf("managed payload %d differs from its allocation guard", actual)
	}
}

// This reference uses the independent allocating Forward/Tape path and direct
// target windows, rather than the evaluation workspace or prefetch planner.
func evaluationBoundaryReference(t testing.TB, m *Model, targets []int, seq, limit int) (float64, int) {
	t.Helper()
	var sum float64
	n := 0
	for start, batch := 0, 0; start < len(targets) && (limit == 0 || batch < limit); start, batch = start+seq, batch+1 {
		end := min(start+seq, len(targets))
		x := append([]int{BOS}, targets[start:end-1]...)
		tape, err := m.Forward(x, 1, len(x))
		if err != nil {
			t.Fatal(err)
		}
		loss, _, err := CrossEntropy(tape.Logits, targets[start:end], m.Config.Vocab, 1, false)
		if err != nil {
			t.Fatal(err)
		}
		sum += loss * float64(end-start)
		n += end - start
	}
	return sum / float64(n), n
}

func TestEvaluationTextAndDatasetExactTailAndSubset(t *testing.T) {
	m := testModel(t)
	_, d := preparedDataset(t, Tokenizer{}, "train\nsecond\n", "abc\ndefgh\nijk", 10)
	for _, split := range []string{"train", "validation"} {
		var targets []int
		for i, shard := range d.manifest.Shards {
			if shard.Split == split {
				buf := make([]int, shard.Tokens-1)
				if err := d.ReadTokens(context.Background(), i, 1, buf); err != nil {
					t.Fatal(err)
				}
				targets = append(targets, buf...)
			}
		}
		for _, seq := range []int{1, 5, m.Config.Context} {
			for _, limit := range []int{0, 1, 2, 100} {
				t.Run(fmt.Sprintf("%s/seq%d/limit%d", split, seq, limit), func(t *testing.T) {
					wantLoss, wantCount := evaluationBoundaryReference(t, m, targets, seq, limit)
					for _, source := range []string{"text", "dataset"} {
						var gotLoss float64
						var gotCount int
						var err error
						if source == "text" {
							gotLoss, gotCount, err = evaluate(context.Background(), m, append([]int{BOS}, targets...), seq, limit)
						} else {
							gotLoss, gotCount, err = evaluateDatasetSplit(context.Background(), m, d, split, seq, limit)
						}
						if err != nil {
							t.Fatal(err)
						}
						if gotCount != wantCount || math.Float64bits(gotLoss) != math.Float64bits(wantLoss) {
							t.Fatalf("%s loss/count (%016x, %d) != reference (%016x, %d)", source, math.Float64bits(gotLoss), gotCount, math.Float64bits(wantLoss), wantCount)
						}
					}
				})
			}
		}
	}
}

func TestEvaluationClampsWorkspaceToAvailableTargets(t *testing.T) {
	// A full-context workspace would exceed the guard despite this tiny split
	// needing only a few token rows. Both entry points must size to actual work.
	c := Config{Vocab: 65536, Dim: 2, Hidden: 4, Layers: 1, Heads: 1, KVHeads: 1, Context: 32768, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	if evaluationWorkspaceElements(c, 1, c.Context) <= maxActivationElements {
		t.Fatal("fixture full-context workspace must exceed the allocation guard")
	}
	_, d := preparedDataset(t, Tokenizer{}, "train", "x", 10)
	data := (Tokenizer{}).Encode("x", true, true)
	wantLoss, wantCount := evaluationBoundaryReference(t, m, data[1:], c.Context, 0)
	textLoss, textCount, err := evaluate(context.Background(), m, data, c.Context, 0)
	if err != nil {
		t.Fatal(err)
	}
	datasetLoss, datasetCount, err := evaluateDataset(context.Background(), m, d, c.Context, 0)
	if err != nil {
		t.Fatal(err)
	}
	if textCount != wantCount || datasetCount != wantCount || math.Float64bits(textLoss) != math.Float64bits(wantLoss) || math.Float64bits(datasetLoss) != math.Float64bits(wantLoss) {
		t.Fatal("short split evaluation differs from the independent reference")
	}
}

// Run with GOMAXPROCS=4 for the documented comparison. Unlike the warmed
// forward microbenchmark, each operation includes construction, input framing,
// every validation window, and loss reduction over the authored heldout split.
func BenchmarkEvaluationText(b *testing.B) {
	text, err := os.ReadFile("examples/heldout/validation.txt")
	if err != nil {
		b.Fatal(err)
	}
	data := (Tokenizer{}).Encode(string(text), true, true)
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
			wantLoss, wantCount := evaluationBoundaryReference(b, m, data[1:], 64, 0)
			for _, path := range []string{"allocating", "workspace"} {
				b.Run(path, func(b *testing.B) {
					var loss float64
					var count int
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if path == "allocating" {
							loss, count = evaluationBoundaryReference(b, m, data[1:], 64, 0)
						} else {
							loss, count, err = evaluate(context.Background(), m, data, 64, 0)
							if err != nil {
								b.Fatal(err)
							}
						}
					}
					b.StopTimer()
					b.ReportMetric(float64(count), "targets/op")
					if count != wantCount || math.Float64bits(loss) != math.Float64bits(wantLoss) {
						b.Fatal("benchmark evaluation differs from the independent reference")
					}
				})
			}
		})
	}
}
