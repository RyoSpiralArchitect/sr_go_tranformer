package main

import (
	"bytes"
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"
)

func datasetTestState(d *TokenDataset) *TrainState {
	spec := defaultTrainSpec()
	spec.Steps = 6
	spec.Warmup = 1
	spec.Batch = 2
	spec.Seq = 5
	spec.Accum = 3
	return &TrainState{Spec: spec, RNG: RNG{79}, CorpusSHA256: d.id, Dataset: datasetTrainingState(d, 79)}
}
func checkpointBytes(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func TestDatasetCheckpointExactResume(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "first\nsecond\nthird\nfourth\n", "held out\nfinal", 14)
	dir := t.TempDir()
	full, partial := filepath.Join(dir, "full.mglm"), filepath.Join(dir, "partial.mglm")
	o := loopOptions{Out: full, SaveEvery: 1, EvalEvery: 2, EvalBatches: 0, LogEvery: 6}
	m, s := testModel(t), datasetTestState(d)
	if err := trainDatasetLoop(context.Background(), m, Tokenizer{}, s, d, 1, 1, o); err != nil {
		t.Fatal(err)
	}
	o.Out = partial
	o.StopAfter = 2
	if err := trainDatasetLoop(context.Background(), testModel(t), Tokenizer{}, datasetTestState(d), d, 3, 5, o); err != nil {
		t.Fatal(err)
	}
	r, tok, state, err := LoadCheckpoint(partial)
	if err != nil {
		t.Fatal(err)
	}
	if state.Step != 2 || state.TokensSeen != 60 {
		t.Fatal("incorrect committed progress", state)
	}
	if err = validateDatasetResume(d, tok, state); err != nil {
		t.Fatal(err)
	}
	o.StopAfter = 0
	if err = trainDatasetLoop(context.Background(), r, tok, state, d, 4, 7, o); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(checkpointBytes(t, full), checkpointBytes(t, partial)) {
		t.Fatal("resumed checkpoint differs from continuous training")
	}
}

func TestDatasetCanceledAccumulationSavesCommittedPrefix(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "abcdefghijklmnopqrstuvwxyz\n", "held\n", 40)
	m, s := testModel(t), datasetTestState(d)
	ensureMoments(m)
	// Begin from a nonzero committed prefix, including real Adam moments.
	warm, err := NewTokenPrefetch(context.Background(), d, "train", PrefetchConfig{Batch: s.Spec.Batch, Seq: s.Spec.Seq, Workers: 1, Depth: 1, Seed: s.Dataset.OrderSeed, Shuffle: true}, s.Dataset.Cursor)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = trainDatasetUpdate(context.Background(), m, s, warm); err != nil {
		warm.Close()
		t.Fatal(err)
	}
	warm.Close()
	dir := t.TempDir()
	before, after := filepath.Join(dir, "before"), filepath.Join(dir, "after")
	if err := SaveCheckpoint(before, m, Tokenizer{}, s); err != nil {
		t.Fatal(err)
	}
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
	defer p.Close()
	err = trainingLoop(ctx, m, Tokenizer{}, s, 1, 1, loopOptions{Out: after, LogEvery: 1},
		func() (float64, float64, error) { return trainDatasetUpdate(ctx, m, s, p) }, func() (float64, int, error) { return 1, 1, nil })
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Err() == nil || reads.Load() != 2 {
		t.Fatal("test did not interrupt accumulation")
	}
	if !bytes.Equal(checkpointBytes(t, before), checkpointBytes(t, after)) {
		t.Fatal("partial accumulation advanced saved weights/moments/cursor")
	}
	// The saved prefix can still continue with fresh prefetch buffers.
	r, tok, state, err := LoadCheckpoint(after)
	if err != nil {
		t.Fatal(err)
	}
	if err = trainDatasetLoop(context.Background(), r, tok, state, d, 2, 3, loopOptions{Out: after, LogEvery: 6}); err != nil {
		t.Fatal(err)
	}
}

func TestDatasetResumeRejectsChangedIdentityAndCursor(t *testing.T) {
	_, a := preparedDataset(t, Tokenizer{}, "aaa\nbbb\n", "val\n", 8)
	_, b := preparedDataset(t, Tokenizer{}, "ccc\nddd\n", "val\n", 8)
	s := datasetTestState(a)
	if err := validateDatasetResume(b, Tokenizer{}, s); err == nil {
		t.Fatal("changed dataset accepted")
	}
	for _, mutate := range []func(*TrainState){
		func(s *TrainState) { s.Dataset.Cursor.Offset++ },
		func(s *TrainState) { s.Dataset.Cursor.Epoch++ },
		func(s *TrainState) { s.Dataset.Cursor.Shard++ },
		func(s *TrainState) { s.Dataset.TokenizerSHA256 = tokenizerHash(Tokenizer{Merges: []Pair{{'a', 'b'}}}) },
	} {
		s = datasetTestState(a)
		mutate(s)
		if err := validateDatasetResume(a, Tokenizer{}, s); err == nil {
			t.Fatal("invalid resume state accepted")
		}
	}
	// Moving immutable files changes no content identity.
	old := a.root
	destination := filepath.Join(t.TempDir(), "moved")
	if err := os.Rename(old, destination); err != nil {
		t.Fatal(err)
	}
	relocated, err := OpenTokenDataset(context.Background(), filepath.Join(destination, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = validateDatasetResume(relocated, Tokenizer{}, datasetTestState(a)); err != nil {
		t.Fatal(err)
	}
	// Order is part of identity even when all bytes and checksums are unchanged.
	m := relocated.manifest
	m.Shards[0], m.Shards[1] = m.Shards[1], m.Shards[0]
	path := filepath.Join(destination, "manifest.json")
	if err = writeJSONFile(path, m); err != nil {
		t.Fatal(err)
	}
	reordered, err := OpenTokenDataset(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if err = validateDatasetResume(reordered, Tokenizer{}, datasetTestState(a)); err == nil {
		t.Fatal("reordered manifest accepted")
	}
}

func TestDatasetEvaluationCountsTailOnce(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "train\n", "abc\ndefgh\nijk", 10)
	m := testModel(t)
	var targets []int
	for i, s := range d.manifest.Shards {
		if s.Split == "validation" {
			buf := make([]int, s.Tokens-1)
			if err := d.ReadTokens(context.Background(), i, 1, buf); err != nil {
				t.Fatal(err)
			}
			targets = append(targets, buf...)
		}
	}
	for _, limit := range []int{0, 1, 2, 10} {
		loss, count, err := evaluateDataset(context.Background(), m, d, 5, limit)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		n := 0
		for start, batch := 0, 0; start < len(targets) && (limit == 0 || batch < limit); start, batch = start+5, batch+1 {
			end := min(start+5, len(targets))
			x := append([]int{BOS}, targets[start:end-1]...)
			tape, err := m.Forward(x, 1, len(x))
			if err != nil {
				t.Fatal(err)
			}
			l, _, err := CrossEntropy(tape.Logits, targets[start:end], m.Config.Vocab, 1, false)
			if err != nil {
				t.Fatal(err)
			}
			sum += l * float64(end-start)
			n += end - start
		}
		if count != n || math.Abs(loss-sum/float64(n)) > 1e-12 {
			t.Fatal("evaluation duplicated/dropped tail", limit, loss, count, n)
		}
	}
}

func TestDatasetTrainingCLIAndCheckpointGuards(t *testing.T) {
	tok := Tokenizer{Merges: []Pair{{'a', 'b'}}}
	manifest, d := preparedDataset(t, tok, "abab\nababab\n", "abval\n", 12)
	dir := t.TempDir()
	config, out := filepath.Join(dir, "config.json"), filepath.Join(dir, "model.mglm")
	c := testModel(t).Config
	if err := writeJSONFile(config, c); err != nil {
		t.Fatal(err)
	}
	args := []string{"-dataset", manifest, "-config", config, "-steps", "3", "-warmup", "1", "-seq", "3", "-batch", "1", "-accum", "2", "-stop-after", "1", "-out", out, "-save-every", "0", "-eval-every", "0", "-log-every", "3"}
	if err := runTrain(context.Background(), args); err != nil {
		t.Fatal(err)
	}
	_, got, state, err := LoadCheckpoint(out)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, tok) || state.Dataset == nil {
		t.Fatal("manifest tokenizer not used")
	}
	for _, bad := range [][]string{
		{"-dataset", manifest, "-data", "examples/tiny.txt"},
		{"-dataset", manifest, "-tokenizer", "ignored.json"},
		{"-dataset", manifest, "-val-fraction", "0.1"},
		{"-data", "examples/tiny.txt", "-resume", out},
		{"-dataset", manifest, "-resume", out, "-seed", "1"},
		{"-data", "examples/tiny.txt", "-prefetch", "4"},
	} {
		if err := runTrain(context.Background(), bad); err == nil {
			t.Fatal("invalid CLI accepted", bad)
		}
	}
	if err := runTrain(context.Background(), []string{"-dataset", manifest, "-resume", out, "-out", out, "-data-workers", "1", "-prefetch", "1"}); err != nil {
		t.Fatal(err)
	}
	// A tokenizer mismatch must be rejected at save, before any new file appears.
	m := testModel(t)
	s := datasetTestState(d)
	ensureMoments(m)
	if err := SaveCheckpoint(filepath.Join(dir, "bad"), m, Tokenizer{}, s); err == nil {
		t.Fatal("dataset tokenizer mismatch saved")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := runTrain(ctx, []string{"-dataset", manifest}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
