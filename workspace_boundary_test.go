// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"math"
	"runtime"
	"testing"
)

func workspaceBoundaryGradients(m *Model) [][]float32 {
	values := make([][]float32, len(m.Params))
	for i, p := range m.Params {
		values[i] = append([]float32(nil), p.Grad...)
	}
	return values
}

func TestBackwardWorkspaceRejectsBoundaryBeforeMutation(t *testing.T) {
	type mutation func(*backwardWorkspace, *Model)
	cases := map[string]mutation{
		"nil_owner":           func(w *backwardWorkspace, _ *Model) { w.model = nil },
		"foreign_owner":       func(w *backwardWorkspace, _ *Model) { w.model = &Model{Config: w.config} },
		"workspace_config":    func(w *backwardWorkspace, _ *Model) { w.config.NormEps *= 2 },
		"model_config":        func(_ *backwardWorkspace, m *Model) { m.Config.RopeTheta *= 2 },
		"batch":               func(w *backwardWorkspace, _ *Model) { w.batch++ },
		"sequence":            func(w *backwardWorkspace, _ *Model) { w.seq++ },
		"same_elements_shape": func(w *backwardWorkspace, _ *Model) { w.batch, w.seq = 3, 2 },
	}
	for name, selectBuffer := range map[string]func(*backwardWorkspace) *[]float32{
		"loss":       func(w *backwardWorkspace) *[]float32 { return &w.loss },
		"residual":   func(w *backwardWorkspace) *[]float32 { return &w.residual },
		"projection": func(w *backwardWorkspace) *[]float32 { return &w.projection },
		"temp":       func(w *backwardWorkspace) *[]float32 { return &w.temp },
		"dq":         func(w *backwardWorkspace) *[]float32 { return &w.dq },
		"dh":         func(w *backwardWorkspace) *[]float32 { return &w.dh },
		"dg":         func(w *backwardWorkspace) *[]float32 { return &w.dg },
		"du":         func(w *backwardWorkspace) *[]float32 { return &w.du },
		"dk":         func(w *backwardWorkspace) *[]float32 { return &w.dk },
		"dv":         func(w *backwardWorkspace) *[]float32 { return &w.dv },
	} {
		selectBuffer := selectBuffer
		cases[name+"/nil"] = func(w *backwardWorkspace, _ *Model) { *selectBuffer(w) = nil }
		cases[name+"/short_length"] = func(w *backwardWorkspace, _ *Model) {
			p := selectBuffer(w)
			*p = (*p)[:len(*p)-1]
		}
		cases[name+"/long_length"] = func(w *backwardWorkspace, _ *Model) {
			p := selectBuffer(w)
			*p = make([]float32, len(*p)+1)
		}
		cases[name+"/extra_capacity"] = func(w *backwardWorkspace, _ *Model) {
			p := selectBuffer(w)
			*p = make([]float32, len(*p), len(*p)+1)
		}
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			m := testModel(t)
			m.ZeroGrad()
			for _, p := range m.Params {
				for i := range p.Grad {
					p.Grad[i] = float32(i%11-5) / 7
				}
			}
			before := workspaceBoundaryGradients(m)
			tape, err := m.Forward([]int{BOS, 2, 3, BOS, 4, 5}, 2, 3)
			if err != nil {
				t.Fatal(err)
			}
			w, err := newBackwardWorkspace(m, 2, 3)
			if err != nil {
				t.Fatal(err)
			}
			mutate(w, m)
			if _, err = m.backward(tape, []int{2, 3, 4, 4, 5, 6}, 1, w); err == nil {
				t.Fatal("accepted an incompatible or malformed workspace")
			}
			if tape.Used {
				t.Fatal("rejected workspace consumed the tape")
			}
			for i, p := range m.Params {
				exactLinearBits(t, p.Grad, before[i])
			}
		})
	}
}

type workspaceBoundaryTapeSnapshot struct {
	ids    []int
	floats [][]float32
	lse    [][]float64
}

func workspaceBoundaryTapeFloats(tape *Tape) [][]float32 {
	values := [][]float32{tape.X, tape.Norm, tape.Inv, tape.Logits}
	for _, l := range tape.Layers {
		values = append(values, l.X, l.N1, l.R1, l.Q, l.K, l.V, l.A, l.U, l.N2, l.R2, l.Gate, l.Up, l.H)
	}
	return values
}

func snapshotWorkspaceBoundaryTape(tape *Tape) workspaceBoundaryTapeSnapshot {
	s := workspaceBoundaryTapeSnapshot{ids: append([]int(nil), tape.IDs...)}
	for _, values := range workspaceBoundaryTapeFloats(tape) {
		s.floats = append(s.floats, append([]float32(nil), values...))
	}
	for _, l := range tape.Layers {
		s.lse = append(s.lse, append([]float64(nil), l.LSE...))
	}
	return s
}

func (s workspaceBoundaryTapeSnapshot) check(t *testing.T, tape *Tape) {
	t.Helper()
	if len(tape.IDs) != len(s.ids) {
		t.Fatal("retained tape IDs changed shape")
	}
	for i, id := range tape.IDs {
		if id != s.ids[i] {
			t.Fatal("retained tape IDs were modified")
		}
	}
	values := workspaceBoundaryTapeFloats(tape)
	if len(values) != len(s.floats) || len(tape.Layers) != len(s.lse) {
		t.Fatal("retained tape changed shape")
	}
	for i, v := range values {
		exactLinearBits(t, v, s.floats[i])
	}
	for i, l := range tape.Layers {
		if len(l.LSE) != len(s.lse[i]) {
			t.Fatal("retained log-normalizers changed shape")
		}
		for j, v := range l.LSE {
			if math.Float64bits(v) != math.Float64bits(s.lse[i][j]) {
				t.Fatal("retained log-normalizer was modified")
			}
		}
	}
}

func TestBackwardWorkspacePreservesIndependentForwardTapes(t *testing.T) {
	m, reference := testModel(t), testModel(t)
	m.ZeroGrad()
	reference.ZeroGrad()
	w, err := newBackwardWorkspace(m, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	xs := [][]int{{BOS, 17, 31}, {BOS, 73, 59}}
	ys := [][]int{{17, 31, EOS}, {73, 59, EOS}}
	var tapes, references []*Tape
	var snapshots []workspaceBoundaryTapeSnapshot
	for i, x := range xs {
		tape, err := m.Forward(x, 1, 3)
		if err != nil {
			t.Fatal(err)
		}
		r, err := reference.Forward(x, 1, 3)
		if err != nil {
			t.Fatal(err)
		}
		tapes, references = append(tapes, tape), append(references, r)
		snapshots = append(snapshots, snapshotWorkspaceBoundaryTape(tape))
		for j := 0; j <= i; j++ {
			snapshots[j].check(t, tapes[j])
		}
	}
	// Consume in reverse order and reuse the same workspace. Forward owns its
	// tapes even when another tape is consumed first, and backward must not
	// borrow any of those retained arrays as its scratch destinations.
	for i := len(tapes) - 1; i >= 0; i-- {
		got, err := m.backward(tapes[i], ys[i], .5, w)
		if err != nil {
			t.Fatal(err)
		}
		want, err := reference.Backward(references[i], ys[i], .5)
		if err != nil {
			t.Fatal(err)
		}
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatal("workspace changed the loss")
		}
		for j, p := range m.Params {
			exactLinearBits(t, p.Grad, reference.Params[j].Grad)
		}
		for j, tape := range tapes {
			snapshots[j].check(t, tape)
			if tape.Used != (j >= i) {
				t.Fatal("consuming one tape changed another tape's lifecycle")
			}
		}
	}
}

func TestBackwardWorkspaceRejectsCombinedBudgetBeforeAllocation(t *testing.T) {
	// The original per-forward bound permits this shape (107,151,360 float32
	// equivalents). Adding retained scratch raises it to 134,873,088, beyond
	// the 134,217,728 limit. Parameters and RoPE tables themselves remain small.
	c := Config{Vocab: baseVocab, Dim: 2, Hidden: 4, Layers: 1, Heads: 1, KVHeads: 1, Context: 32768, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	w, err := newBackwardWorkspace(m, 3, 32768)
	runtime.ReadMemStats(&after)
	if err == nil || w != nil {
		t.Fatal("accepted shape that exceeds the combined activation budget")
	}
	// A rejected constructor may allocate its error, but must not allocate the
	// roughly 106 MiB workspace before it discovers the budget violation.
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated >= 1<<20 {
		t.Fatalf("budget rejection allocated %d bytes before returning", allocated)
	}
}

func TestBackwardWorkspaceLossFailureCanRetry(t *testing.T) {
	for _, invalid := range []string{"target", "logit", "scale"} {
		t.Run(invalid, func(t *testing.T) {
			m, ref := testModel(t), testModel(t)
			m.ZeroGrad()
			ref.ZeroGrad()
			x, targets := []int{BOS, 7, 7}, []int{7, 7, EOS}
			tape, err := m.Forward(x, 1, 3)
			if err != nil {
				t.Fatal(err)
			}
			control, err := ref.Forward(x, 1, 3)
			if err != nil {
				t.Fatal(err)
			}
			w, err := newBackwardWorkspace(m, 1, 3)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range m.Params {
				for i := range p.Grad {
					p.Grad[i] = .25
				}
			}
			for _, p := range ref.Params {
				for i := range p.Grad {
					p.Grad[i] = .25
				}
			}
			before := workspaceBoundaryGradients(m)
			badTargets := append([]int(nil), targets...)
			last := len(tape.Logits) - 1
			savedLogit, scale := tape.Logits[last], 1.0
			switch invalid {
			case "target":
				badTargets[len(badTargets)-1] = m.Config.Vocab
			case "logit":
				tape.Logits[last] = float32(math.NaN())
			case "scale":
				scale = math.NaN()
			}
			if _, err = m.backward(tape, badTargets, scale, w); err == nil || tape.Used {
				t.Fatal("invalid loss input was accepted or consumed the tape")
			}
			for i, p := range m.Params {
				exactLinearBits(t, p.Grad, before[i])
			}
			// Early rows may have overwritten loss scratch before a later error.
			// Retrying must overwrite it completely without clearing Param.Grad.
			tape.Logits[last] = savedLogit
			got, err := m.backward(tape, targets, 1, w)
			if err != nil {
				t.Fatal(err)
			}
			want, err := ref.Backward(control, targets, 1)
			if err != nil {
				t.Fatal(err)
			}
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Fatal("loss changed after retry")
			}
			for i, p := range m.Params {
				exactLinearBits(t, p.Grad, ref.Params[i].Grad)
			}
		})
	}
}
