// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

func testModel(t testing.TB) *Model {
	t.Helper()
	c := Config{Vocab: baseVocab, Dim: 8, Hidden: 12, Layers: 2, Heads: 2, KVHeads: 1, Context: 128, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func closeSlices(t testing.TB, a, b []float32, tol float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length %d != %d", len(a), len(b))
	}
	for i := range a {
		if !finite(float64(a[i])) || !finite(float64(b[i])) || math.Abs(float64(a[i]-b[i])) > tol {
			t.Fatalf("index %d: %.9g != %.9g (tol %g)", i, a[i], b[i], tol)
		}
	}
}
func lossFor(t testing.TB, m *Model, x, y []int, batch, seq int) float64 {
	t.Helper()
	tape, err := m.Forward(x, batch, seq)
	if err != nil {
		t.Fatal(err)
	}
	loss, _, err := CrossEntropy(tape.Logits, y, m.Config.Vocab, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	return loss
}
func TestConfig(t *testing.T) {
	for _, name := range []string{"demo", "tiny", "small", "base"} {
		c, err := preset(name)
		if err != nil {
			t.Fatal(err)
		}
		if name == "demo" {
			m, _ := NewModel(c, 1)
			var n int64
			for _, p := range m.Params {
				n += int64(len(p.Data))
			}
			if n != c.ParameterCount() {
				t.Fatalf("parameter count %d != %d", n, c.ParameterCount())
			}
		}
	}
	c := testModel(t).Config
	for name, mutate := range map[string]func(*Config){"odd_head": func(c *Config) { c.Dim = 6 }, "nondivisible_gqa": func(c *Config) { c.Heads = 4; c.KVHeads = 3 }, "nan_eps": func(c *Config) { c.NormEps = math.NaN() }, "zero_context": func(c *Config) { c.Context = 0 }, "overflow": func(c *Config) { c.Dim = 8192; c.Hidden = 32768; c.Layers = 128 }} {
		t.Run(name, func(t *testing.T) {
			bad := c
			mutate(&bad)
			if bad.Validate() == nil {
				t.Fatal("accepted invalid config")
			}
		})
	}
}
func TestTokenizerRoundTripAndDeterminism(t *testing.T) {
	text := strings.Repeat("SpiralReality 猫と雨 🐈‍⬛\x00\xff\n", 20)
	a, err := TrainTokenizer(text, 300)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TrainTokenizer(text, 300)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("BPE tie breaking is not deterministic")
	}
	if a.Vocab() <= baseVocab || len(a.Encode(text, false, false)) >= len(text) {
		t.Fatal("BPE did not compress")
	}
	var every []byte
	for i := 0; i < 256; i++ {
		every = append(every, byte(i))
	}
	for _, tok := range []Tokenizer{{}, a} {
		for _, s := range []string{"", text, string(every), "a<bos>b<eos>c", "日本語の猫"} {
			if got := tok.Decode(tok.Encode(s, true, true)); got != s {
				t.Fatalf("byte roundtrip failed: %q != %q", got, s)
			}
		}
	}
	for _, tok := range []Tokenizer{{[]Pair{{258, 0}}}, {[]Pair{{BOS, 0}}}, {[]Pair{{0, 1}, {0, 1}}}} {
		if tok.Validate() == nil {
			t.Fatal("invalid tokenizer accepted")
		}
	}
}
func TestCrossEntropyExtremeAndDerivative(t *testing.T) {
	logits := []float32{10000, -10000, 9999, -10000, 10000, 0}
	targets := []int{1, 1}
	loss, g, err := CrossEntropy(logits, targets, 3, .5, true)
	if err != nil || !finite(loss) || math.Abs(loss-(20000+math.Log1p(math.Exp(-1)))/2) > 1e-8 {
		t.Fatalf("unstable loss %.12g, %v", loss, err)
	}
	for r := 0; r < 2; r++ {
		var sum float32
		for _, v := range g[r*3 : (r+1)*3] {
			sum += v
		}
		if math.Abs(float64(sum)) > 1e-7 {
			t.Fatal("softmax gradient does not sum to zero")
		}
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		if _, _, err := CrossEntropy([]float32{bad, 0}, []int{1}, 2, 1, true); err == nil {
			t.Fatal("nonfinite logit accepted")
		}
	}
}
func TestFullModelFiniteDifferences(t *testing.T) {
	m := testModel(t)
	x := []int{BOS, 97, 98, 99, 98, 100}
	y := []int{97, 98, 99, 98, 100, EOS}
	m.ZeroGrad()
	tape, err := m.Forward(x, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.Backward(tape, y, 1); err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	checked := 0
	for _, p := range m.Params {
		indices := make([]int, len(p.Data))
		for i := range indices {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool {
			return math.Abs(float64(p.Grad[indices[i]])) > math.Abs(float64(p.Grad[indices[j]]))
		})
		indices = indices[:min(6, len(indices))]
		indices = append(indices, len(p.Data)/2, len(p.Data)-1)
		for _, idx := range indices {
			original := p.Data[idx]
			eps := float32(.0002)
			p.Data[idx] = original + eps
			plus := lossFor(t, m, x, y, 2, 3)
			p.Data[idx] = original - eps
			minus := lossFor(t, m, x, y, 2, 3)
			p.Data[idx] = original
			numeric := (plus - minus) / float64((original+eps)-(original-eps))
			analytic := float64(p.Grad[idx])
			diff := math.Abs(numeric - analytic)
			worst = math.Max(worst, diff)
			checked++
			if diff > 2e-4+.003*math.Max(math.Abs(numeric), math.Abs(analytic)) {
				t.Fatalf("%s[%d] analytic=%g numeric=%g diff=%g", p.Name, idx, analytic, numeric, diff)
			}
		}
	}
	t.Logf("%d finite differences across every parameter tensor; worst absolute error %.3g", checked, worst)
}

// This intentionally uses a separate dense float64 reference, not attend().
func TestAttentionAgainstDenseReference(t *testing.T) {
	m := testModel(t)
	c := m.Config
	b, seq := 2, 5
	hd, kd := c.Dim/c.Heads, c.KVDim()
	rng := &RNG{4}
	q, k, v := make([]float32, b*seq*c.Dim), make([]float32, b*seq*kd), make([]float32, b*seq*kd)
	for _, x := range [][]float32{q, k, v} {
		for i := range x {
			x[i] = float32(rng.Normal())
		}
	}
	got, lse := m.attention(q, k, v, b, seq)
	if len(lse) != b*c.Heads*seq {
		t.Fatal("attention retained a quadratic normalizer")
	}
	want := make([]float32, len(got))
	for batch := 0; batch < b; batch++ {
		for h := 0; h < c.Heads; h++ {
			for pos := 0; pos < seq; pos++ {
				qi := (batch*seq+pos)*c.Dim + h*hd
				scores := make([]float64, pos+1)
				maxS := math.Inf(-1)
				for s := 0; s <= pos; s++ {
					ki := (batch*seq+s)*kd + (h/(c.Heads/c.KVHeads))*hd
					for j := 0; j < hd; j++ {
						scores[s] += float64(q[qi+j]) * float64(k[ki+j]) / math.Sqrt(float64(hd))
					}
					maxS = math.Max(maxS, scores[s])
				}
				var sum float64
				for s := range scores {
					scores[s] = math.Exp(scores[s] - maxS)
					sum += scores[s]
				}
				for j := 0; j < hd; j++ {
					var total float64
					for s, p := range scores {
						ki := (batch*seq+s)*kd + (h/(c.Heads/c.KVHeads))*hd
						total += p / sum * float64(v[ki+j])
					}
					want[qi+j] = float32(total)
				}
			}
		}
	}
	closeSlices(t, got, want, 5e-7)
}
func TestCausalityAndBatchIsolation(t *testing.T) {
	m := testModel(t)
	a := []int{BOS, 11, 12, 13, BOS, 21, 22, 23}
	b := append([]int(nil), a...)
	b[3] = 88
	b[4] = 99
	b[5] = 100
	b[6] = 101
	b[7] = 102
	ta, _ := m.Forward(a, 2, 4)
	tb, _ := m.Forward(b, 2, 4)
	closeSlices(t, ta.Logits[:3*m.Config.Vocab], tb.Logits[:3*m.Config.Vocab], 0)
	single, _ := m.Forward(a[:4], 1, 4)
	closeSlices(t, ta.Logits[:4*m.Config.Vocab], single.Logits, 0)
}
func TestPagedCacheAndMixedPositionBatchParity(t *testing.T) {
	m := testModel(t)
	ids := make([]int, 70)
	ids[0] = BOS
	for i := 1; i < len(ids); i++ {
		ids[i] = i%31 + 32
	}
	full, err := m.Forward(ids, 1, len(ids))
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	defer cache.Close()
	if len(cache.Pages) != 0 {
		t.Fatal("eager KV allocation")
	}
	for i, id := range ids {
		logits, err := m.Step(cache, id)
		if err != nil {
			t.Fatal(err)
		}
		closeSlices(t, logits, full.Logits[i*m.Config.Vocab:(i+1)*m.Config.Vocab], 2e-6)
	}
	if len(cache.Pages) != (len(ids)+pageTokens-1)/pageTokens {
		t.Fatal("incorrect page count")
	}
	other := m.NewCache()
	defer other.Close()
	single := m.NewCache()
	defer single.Close()
	_, _ = m.Step(other, BOS)
	_, _ = m.Step(single, BOS)
	batched, err := m.DecodeBatch(context.Background(), []*KVCache{cache, other}, []int{120, 121})
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Step(single, 121)
	if err != nil {
		t.Fatal(err)
	}
	closeSlices(t, batched[m.Config.Vocab:], want, 0)
	prefix := append(append([]int(nil), ids...), 120)
	wantFull, _ := m.Forward(prefix, 1, len(prefix))
	closeSlices(t, batched[:m.Config.Vocab], wantFull.Logits[(len(prefix)-1)*m.Config.Vocab:], 2e-6)
	if _, err = m.DecodeBatch(context.Background(), []*KVCache{other, other}, []int{1, 1}); err == nil {
		t.Fatal("duplicate cache accepted")
	}
	other.Close()
	if _, err = m.Step(other, 1); err == nil {
		t.Fatal("released cache accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pos := cache.Pos
	if _, err = m.DecodeBatch(ctx, []*KVCache{cache}, []int{1}); !errors.Is(err, context.Canceled) || cache.Pos != pos {
		t.Fatal("canceled decode advanced state")
	}
	m.Revision++
	if _, err = m.Step(cache, 1); err == nil {
		t.Fatal("stale cache accepted")
	}
}
func TestContextAndTapeBoundaries(t *testing.T) {
	m := testModel(t)
	m.Config.Context = 2
	cache := m.NewCache()
	defer cache.Close()
	_, _ = m.Step(cache, 1)
	_, _ = m.Step(cache, 2)
	if _, err := m.Step(cache, 3); err == nil {
		t.Fatal("cache silently wrapped")
	}
	if _, err := m.Forward([]int{1, 2, 3}, 1, 3); err == nil {
		t.Fatal("oversize forward accepted")
	}
	if _, err := m.Forward([]int{-1}, 1, 1); err == nil {
		t.Fatal("invalid token accepted")
	}
	tape, _ := m.Forward([]int{1}, 1, 1)
	if _, err := m.Backward(tape, []int{2}, 1); err == nil {
		t.Fatal("missing gradients accepted")
	}
	m.ZeroGrad()
	if _, err := m.Backward(tape, []int{2}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Backward(tape, []int{2}, 1); err == nil {
		t.Fatal("double backward accepted")
	}
}
func TestGradientAccumulationMatchesBatch(t *testing.T) {
	a := testModel(t)
	b := testModel(t)
	x := []int{1, 2, 3, 7, 8, 9}
	y := []int{2, 3, 4, 8, 9, 10}
	a.ZeroGrad()
	b.ZeroGrad()
	tape, _ := a.Forward(x, 2, 3)
	_, err := a.Backward(tape, y, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		tape, _ = b.Forward(x[i*3:(i+1)*3], 1, 3)
		if _, err = b.Backward(tape, y[i*3:(i+1)*3], .5); err != nil {
			t.Fatal(err)
		}
	}
	for i, p := range a.Params {
		closeSlices(t, p.Grad, b.Params[i].Grad, 2e-6)
	}
}
func TestOptimizerPreflightAndSchedule(t *testing.T) {
	m := testModel(t)
	s := defaultTrainSpec()
	s.Seq = 4
	s.Steps = 20
	s.Warmup = 3
	m.ZeroGrad()
	before := append([]float32(nil), m.Emb.Data...)
	m.Params[len(m.Params)-1].Grad[0] = float32(math.NaN())
	if _, err := m.AdamW(s, 1); err == nil {
		t.Fatal("NaN gradient accepted")
	}
	closeSlices(t, m.Emb.Data, before, 0)
	if m.Revision != 0 || len(m.Emb.M) != 0 {
		t.Fatal("partial optimizer mutation")
	}
	if math.Abs(s.LearningRate(1)-s.LR/3) > 1e-14 || s.LearningRate(3) != s.LR || s.LearningRate(20) != s.MinLR {
		t.Fatal("incorrect learning-rate endpoints")
	}
	m.ZeroGrad()
	for _, p := range m.Params {
		for i := range p.Grad {
			p.Grad[i] = 100
		}
	}
	norm, err := m.AdamW(s, 1)
	if err != nil || norm <= s.Clip {
		t.Fatal("clipping test invalid", err)
	}
	for _, p := range m.Params {
		for _, v := range p.Data {
			if !finite(float64(v)) {
				t.Fatal("clipped update became nonfinite")
			}
		}
	}
}
func TestCheckpointExactResume(t *testing.T) {
	a := testModel(t)
	spec := defaultTrainSpec()
	spec.Seq = 4
	spec.Batch = 2
	spec.Accum = 2
	spec.Steps = 8
	spec.Warmup = 2
	text := strings.Repeat(demoCorpus, 2)
	data := Tokenizer{}.Encode(text, true, true)
	s := &TrainState{Spec: spec, RNG: RNG{123}, CorpusSHA256: corpusHash(text)}
	for i := 0; i < 3; i++ {
		if _, _, err := trainUpdate(a, s, data); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "resume.mglm")
	if err := SaveCheckpoint(path, a, Tokenizer{}, s); err != nil {
		t.Fatal(err)
	}
	b, tok, resumed, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if tok.Vocab() != baseVocab || !reflect.DeepEqual(s, resumed) {
		t.Fatal("training metadata changed")
	}
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	for s.Step < spec.Steps {
		la, _, e := trainUpdate(a, s, data)
		if e != nil {
			t.Fatal(e)
		}
		runtime.GOMAXPROCS(4)
		lb, _, e := trainUpdate(b, resumed, data)
		if e != nil {
			t.Fatal(e)
		}
		runtime.GOMAXPROCS(1)
		if la != lb {
			t.Fatalf("resumed loss differs: %g != %g", la, lb)
		}
	}
	if !reflect.DeepEqual(s, resumed) {
		t.Fatal("resumed cursor/RNG diverged")
	}
	for i, p := range a.Params {
		closeSlices(t, p.Data, b.Params[i].Data, 0)
		closeSlices(t, p.M, b.Params[i].M, 0)
		closeSlices(t, p.V, b.Params[i].V, 0)
	}
}
func TestCheckpointCorruptionAndAtomicFailure(t *testing.T) {
	m := testModel(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "model.mglm")
	if err := SaveCheckpoint(path, m, Tokenizer{}, nil); err != nil {
		t.Fatal(err)
	}
	good, _ := os.ReadFile(path)
	cases := map[string][]byte{"truncated": good[:len(good)-1], "extra": append(append([]byte(nil), good...), 0)}
	bad := append([]byte(nil), good...)
	bad[len(bad)-33] ^= 1
	cases["checksum"] = bad
	bad = append([]byte(nil), good...)
	binary.LittleEndian.PutUint32(bad[8:12], 0xffffffff)
	cases["huge_header"] = bad
	// A valid checksum must not bypass finite tensor validation.
	bad = append([]byte(nil), good...)
	header := int(binary.LittleEndian.Uint32(bad[8:12]))
	binary.LittleEndian.PutUint32(bad[12+header:], math.Float32bits(float32(math.NaN())))
	sum := sha256.Sum256(bad[:len(bad)-32])
	copy(bad[len(bad)-32:], sum[:])
	cases["nan_tensor"] = bad
	for name, b := range cases {
		t.Run(name, func(t *testing.T) {
			p := filepath.Join(dir, name)
			if err := os.WriteFile(p, b, 0600); err != nil {
				t.Fatal(err)
			}
			if _, _, _, err := LoadCheckpoint(p); err == nil {
				t.Fatal("accepted malformed checkpoint")
			}
		})
	}
	m.Emb.Data[0] = float32(math.NaN())
	if err := SaveCheckpoint(path, m, Tokenizer{}, nil); err == nil {
		t.Fatal("saved nonfinite model")
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(good, after) {
		t.Fatal("failed save destroyed original checkpoint")
	}
}
func TestTinyLearning(t *testing.T) {
	m := testModel(t)
	spec := defaultTrainSpec()
	spec.Steps = 300
	spec.Warmup = 3
	spec.Seq = 8
	spec.Batch = 2
	spec.LR = .01
	spec.MinLR = .001
	spec.WeightDecay = 0
	text := strings.Repeat("abcabcabcabc", 20)
	data := Tokenizer{}.Encode(text, true, true)
	s := &TrainState{Spec: spec, RNG: RNG{33}, CorpusSHA256: corpusHash(text)}
	before, _, err := evaluate(context.Background(), m, data, 8, 4)
	if err != nil {
		t.Fatal(err)
	}
	for s.Step < spec.Steps {
		if _, _, err = trainUpdate(m, s, data); err != nil {
			t.Fatal(err)
		}
	}
	after, _, err := evaluate(context.Background(), m, data, 8, 4)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("synthetic repetition loss %.6f -> %.6f", before, after)
	if after > .5 {
		t.Fatalf("failed to learn: %.6f -> %.6f", before, after)
	}
}

func await(t testing.TB, what string, predicate func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !predicate() {
		if time.Now().After(deadline) {
			t.Fatalf("timeout waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}
func TestContinuousBatchMatchesIndependentGeneration(t *testing.T) {
	m := testModel(t)
	tok := Tokenizer{}
	e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 4, Queue: 32, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	prompts := []string{"a", "the spiral", strings.Repeat("x", 37), "猫"}
	opts := defaultSampling()
	opts.MaxTokens = 16
	opts.RepeatPenalty = 1
	opts.TopK = 12
	wants := make([]Completion, len(prompts))
	for i, p := range prompts {
		opts.Seed = uint64(i + 1)
		wants[i], err = Generate(context.Background(), m, tok, p, opts)
		if err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			j := i % len(prompts)
			s := opts
			s.Seed = uint64(j + 1)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			got, err := e.Generate(ctx, prompts[j], s)
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			if !reflect.DeepEqual(got, wants[j]) {
				t.Errorf("request %d differs with batching: %v vs %v", i, got.TokenIDs, wants[j].TokenIDs)
			}
		}(i)
	}
	close(start)
	wg.Wait()
	await(t, "sessions released", func() bool { return e.Stats().Active == 0 })
	s := e.Stats()
	if s.Completed != 16 || s.KVReservedBytes != 0 || s.KVAllocatedBytes > e.Config.CacheBytes {
		t.Fatalf("accounting invariant: %+v", s)
	}
}
func TestEngineCancellationShutdownAndBudget(t *testing.T) {
	c, _ := preset("demo")
	c.Context = 2048
	m, _ := NewModel(c, 19)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{MaxBatch: 1, Queue: 4, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	opts := defaultSampling()
	opts.MaxTokens = 20
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() { _, err := e.Generate(ctx, strings.Repeat("x", 1500), opts); result <- err }()
	await(t, "active session", func() bool { return e.Stats().Active > 0 })
	cancel()
	if err = <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
	await(t, "canceled KV reservation", func() bool { return e.Stats().KVReservedBytes == 0 && e.Stats().Active == 0 })
	e.Close()
	if _, err = e.Generate(context.Background(), "", opts); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed engine accepted request: %v", err)
	}
	pageBytes := int64(2 * pageTokens * m.Config.KVDim() * m.Config.Layers * 4)
	limited, err := NewEngine(m, Tokenizer{}, EngineConfig{1, 1, pageBytes})
	if err != nil {
		t.Fatal(err)
	}
	defer limited.Close()
	opts.MaxTokens = 64
	if _, err = limited.Generate(context.Background(), "", opts); err == nil {
		t.Fatal("oversized reservation accepted")
	}
}
func TestBoundedQueueRejectsInsteadOfBlocking(t *testing.T) {
	// Pause the actor by constructing only its mailbox; exercise admission itself.
	m := testModel(t)
	e := &Engine{Model: m, Tokenizer: Tokenizer{}, queue: make(chan *engineRequest, 1), admission: make(chan struct{}, 1), done: make(chan struct{}), pool: pagePool{Limit: 100}}
	e.queue <- &engineRequest{}
	s := defaultSampling()
	s.MaxTokens = 8
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := e.Generate(ctx, "", s); !errors.Is(err, ErrBusy) {
		t.Fatalf("full mailbox: %v", err)
	}
	if e.Stats().Rejected != 1 || len(e.queue) != 1 {
		t.Fatal("mailbox capacity invariant")
	}
}
func TestHTTPAPI(t *testing.T) {
	m := testModel(t)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{2, 8, 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	h := e.Handler()
	cases := []struct {
		method, path, body string
		status             int
	}{
		{"GET", "/healthz", "", 200}, {"GET", "/metrics", "", 200}, {"GET", "/v1/completions", "", 405},
		{"POST", "/v1/completions", `{"prompt":"cat","max_tokens":8,"temperature":0}`, 200},
		{"POST", "/v1/completions", `{"prompt":"cat","unknown":1}`, 400},
		{"POST", "/v1/completions", `{"max_tokens":99999}`, 400},
		{"POST", "/v1/completions", `{"top_p":0}`, 400},
		{"POST", "/v1/completions", `{} {}`, 400},
		{"POST", "/v1/completions", strings.Repeat("x", (1<<20)+1), 413},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s-%s-%d-%d", tc.method, tc.path, tc.status, len(tc.body)), func(t *testing.T) {
			r := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != tc.status {
				t.Fatalf("status %d != %d: %s", w.Code, tc.status, w.Body.String())
			}
			if tc.path == "/v1/completions" && tc.status == http.StatusOK {
				var got Completion
				if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil || got.CompletionTokens != len(got.TokenIDs) || got.CompletionTokens < 1 {
					t.Fatal("invalid completion", err)
				}
			}
		})
	}
}
func TestSamplingBoundsAndDeterminism(t *testing.T) {
	logits := make([]float32, baseVocab)
	logits[BOS] = 100
	logits[17] = 10
	logits[18] = 9
	s := defaultSampling()
	s.Temperature = 0
	id, err := sample(logits, nil, s, &RNG{1})
	if err != nil || id != 17 {
		t.Fatal("greedy or BOS masking failed", id, err)
	}
	s.RepeatPenalty = 2
	id, err = sample(logits, []int{17}, s, &RNG{1})
	if err != nil || id != 18 {
		t.Fatal("repetition penalty failed", id, err)
	}
	s.Temperature = 1
	s.TopK = 1
	for i := 0; i < 50; i++ {
		id, err = sample(logits, nil, s, &RNG{uint64(i)})
		if err != nil || id != 17 {
			t.Fatal("top-k failed")
		}
	}
	s.TopK = 0
	s.TopP = .01
	id, err = sample(logits, nil, s, &RNG{42})
	if err != nil || id != 17 {
		t.Fatal("top-p failed")
	}
	m := testModel(t)
	s.MaxTokens = m.Config.Context
	if _, err = Generate(context.Background(), m, Tokenizer{}, "", s); err == nil {
		t.Fatal("implicit context truncation")
	}
}
func TestEvaluationWeightsPartialWindow(t *testing.T) {
	m := testModel(t)
	ids := []int{BOS, 1, 2, 3, 4, 5}
	a := lossFor(t, m, ids[:4], ids[1:5], 1, 4)
	b := lossFor(t, m, []int{BOS}, ids[5:6], 1, 1)
	loss, count, err := evaluate(context.Background(), m, ids, 4, 0)
	if err != nil || count != 5 || math.Abs(loss-(4*a+b)/5) > 1e-12 {
		t.Fatal("evaluation lost or misweighted final targets", loss, count, err)
	}
}
func TestParallelDeterminism(t *testing.T) {
	c, _ := preset("demo")
	a, _ := NewModel(c, 1)
	b, _ := NewModel(c, 1)
	ids := make([]int, 64)
	targets := make([]int, 64)
	for i := range ids {
		ids[i] = i + 30
		targets[i] = i + 31
	}
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	a.ZeroGrad()
	ta, _ := a.Forward(ids, 2, 32)
	_, err := a.Backward(ta, targets, 1)
	if err != nil {
		t.Fatal(err)
	}
	runtime.GOMAXPROCS(4)
	b.ZeroGrad()
	tb, _ := b.Forward(ids, 2, 32)
	_, err = b.Backward(tb, targets, 1)
	if err != nil {
		t.Fatal(err)
	}
	closeSlices(t, ta.Logits, tb.Logits, 0)
	for i, p := range a.Params {
		closeSlices(t, p.Grad, b.Params[i].Grad, 0)
	}
}
func BenchmarkForward(b *testing.B) {
	c, _ := preset("demo")
	m, _ := NewModel(c, 1)
	ids := make([]int, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Forward(ids, 1, 64); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkDecodeBatch(b *testing.B) {
	for _, n := range []int{1, 4, 8} {
		b.Run(fmt.Sprintf("sessions_%d", n), func(b *testing.B) {
			c, _ := preset("demo")
			m, _ := NewModel(c, 1)
			caches := make([]*KVCache, n)
			tokens := make([]int, n)
			for i := range caches {
				caches[i] = m.NewCache()
				for j := 0; j < 32; j++ {
					_, _ = m.Step(caches[i], 1)
				}
				defer caches[i].Close()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, c := range caches {
					c.Pos = 32
				}
				if _, err := m.DecodeBatch(context.Background(), caches, tokens); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(n), "tokens/op")
		})
	}
}

func TestTiledLinearIndependentReference(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 8, 17} {
		for _, out := range []int{1, 31, 32, 33, 65} {
			in := 37
			rng := &RNG{123}
			x := make([]float32, n*in)
			p := &Param{Rows: out, Cols: in, Data: make([]float32, out*in)}
			for i := range x {
				x[i] = float32(rng.Normal())
			}
			for i := range p.Data {
				p.Data[i] = float32(rng.Normal())
			}
			got := linear(x, p, n)
			want := make([]float32, n*out)
			for row := 0; row < n; row++ {
				for o := 0; o < out; o++ {
					var sum float32
					for j := 0; j < in; j++ {
						sum += x[row*in+j] * p.Data[o*in+j]
					}
					want[row*out+o] = sum
				}
			}
			closeSlices(t, got, want, 0)
		}
	}
}
func BenchmarkAttentionLinearMemory(b *testing.B) {
	for _, seq := range []int{128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("tokens_%d", seq), func(b *testing.B) {
			c, _ := preset("demo")
			c.Context = 2048
			m, _ := NewModel(c, 1)
			q, k, v := make([]float32, seq*c.Dim), make([]float32, seq*c.KVDim()), make([]float32, seq*c.KVDim())
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.attention(q, k, v, 1, seq)
			}
		})
	}
}

func TestTrainingWindowBOSAndNextTokenContract(t *testing.T) {
	data := []int{BOS, 10, 11, 12, 13, 14, 15, EOS}
	x, y := sampleBatch(data, 20, 4, &RNG{123})
	for row := 0; row < 20; row++ {
		if x[row*4] != BOS {
			t.Fatal("training/inference BOS mismatch")
		}
		for j := 1; j < 4; j++ {
			if x[row*4+j] != y[row*4+j-1] {
				t.Fatal("shifted target mismatch")
			}
		}
		for _, id := range y[row*4 : (row+1)*4] {
			if id == BOS {
				t.Fatal("BOS supervised as a continuation")
			}
		}
	}
}

type readProbe struct{ Reads int }

func (r *readProbe) Read(p []byte) (int, error) { r.Reads++; return 0, errors.New("unexpected read") }
func TestHTTPOverloadRejectsBeforeReadingBody(t *testing.T) {
	e, err := NewEngine(testModel(t), Tokenizer{}, EngineConfig{1, 1, 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	for i := 0; i < cap(e.httpSlots); i++ {
		e.httpSlots <- struct{}{}
	}
	reader := &readProbe{}
	w := httptest.NewRecorder()
	e.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/completions", reader))
	if w.Code != 429 || reader.Reads != 0 {
		t.Fatalf("overload read request body: status=%d reads=%d", w.Code, reader.Reads)
	}
}

func TestSmallPresetForwardDecodeParity(t *testing.T) {
	c, _ := preset("small")
	m, err := NewModel(c, 55)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{BOS, 97, 98}
	tape, err := m.Forward(ids, 1, len(ids))
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	defer cache.Close()
	for i, id := range ids {
		logits, err := m.Step(cache, id)
		if err != nil {
			t.Fatal(err)
		}
		closeSlices(t, logits, tape.Logits[i*c.Vocab:(i+1)*c.Vocab], 2e-5)
	}
	t.Logf("%d parameters, %d layers, %d query heads, %d KV heads", c.ParameterCount(), c.Layers, c.Heads, c.KVHeads)
}
func TestHeadConfigurations(t *testing.T) {
	for _, kv := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("kv_%d", kv), func(t *testing.T) {
			c := testModel(t).Config
			c.Dim = 16
			c.Heads = 4
			c.KVHeads = kv
			m, err := NewModel(c, 91)
			if err != nil {
				t.Fatal(err)
			}
			ids := []int{BOS, 1, 2, 3, 4}
			tape, _ := m.Forward(ids, 1, len(ids))
			cache := m.NewCache()
			defer cache.Close()
			for i, id := range ids {
				got, err := m.Step(cache, id)
				if err != nil {
					t.Fatal(err)
				}
				closeSlices(t, got, tape.Logits[i*c.Vocab:(i+1)*c.Vocab], 2e-6)
			}
		})
	}
}
func TestShutdownWakesActiveAndQueuedCallers(t *testing.T) {
	c, _ := preset("demo")
	c.Context = 2048
	m, _ := NewModel(c, 7)
	e, err := NewEngine(m, Tokenizer{}, EngineConfig{2, 8, 2 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	s := defaultSampling()
	s.MaxTokens = 20
	result := make(chan error, 8)
	for i := 0; i < 8; i++ {
		go func() { _, err := e.Generate(context.Background(), strings.Repeat("x", 1500), s); result <- err }()
	}
	await(t, "shutdown workload admitted", func() bool { return e.Stats().Active > 0 })
	e.Close()
	for i := 0; i < 8; i++ {
		select {
		case err := <-result:
			if !errors.Is(err, ErrClosed) {
				t.Fatalf("shutdown waiter returned %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("shutdown left a blocked caller")
		}
	}
	if e.Stats().Active != 0 || e.Stats().KVReservedBytes != 0 {
		t.Fatalf("shutdown leaked reservation: %+v", e.Stats())
	}
}
func TestAdamWFirstStepGolden(t *testing.T) {
	m := testModel(t)
	s := defaultTrainSpec()
	s.Seq = 8
	s.Steps = 20
	s.Warmup = 0
	s.Clip = 1e6
	s.WeightDecay = .1
	m.ZeroGrad()
	old := make([][]float32, len(m.Params))
	for i, p := range m.Params {
		old[i] = append([]float32(nil), p.Data...)
		for j := range p.Grad {
			p.Grad[j] = .25
		}
	}
	if _, err := m.AdamW(s, 1); err != nil {
		t.Fatal(err)
	}
	for i, p := range m.Params {
		for j, prior := range old[i] {
			update := .25 / (.25 + s.Eps)
			if p.Decay {
				update += s.WeightDecay * float64(prior)
			}
			want := prior - float32(s.LR*update)
			if math.Abs(float64(p.Data[j]-want)) > 1e-7 {
				t.Fatalf("AdamW bias correction/decay mismatch %s[%d]", p.Name, j)
			}
		}
	}
}
