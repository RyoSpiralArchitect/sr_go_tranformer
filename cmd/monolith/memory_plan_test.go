// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func TestTrainingMemoryPlanPayloadAndBudgetBoundary(t *testing.T) {
	m := testModel(t)
	m.ZeroGrad()
	ensureMoments(m)
	s := defaultTrainSpec()
	s.Batch, s.Seq = 2, 7
	input := trainingMemoryInput{Kind: "text", TextBytes: 1024}
	p, err := newTrainingMemoryPlan(m.Config, s, input, 0)
	if err != nil {
		t.Fatal(err)
	}
	var weights, gradients, moments int64
	for _, param := range m.Params {
		weights += int64(cap(param.Data)) * 4
		gradients += int64(cap(param.Grad)) * 4
		moments += int64(cap(param.M)+cap(param.V)) * 4
	}
	a := p.Components
	if a.Weights != weights || a.Gradients != gradients || a.AdamMoments != moments || a.RoPE != int64(cap(m.Cos)+cap(m.Sin))*4 {
		t.Fatal("plan does not match allocated model payload", a)
	}
	w, err := newBackwardWorkspace(m, s.Batch, s.Seq)
	if err != nil {
		t.Fatal(err)
	}
	var scratch int64
	for _, b := range workspaceExactBuffers(w) {
		scratch += int64(cap(b)) * 4
	}
	ew, err := newEvaluationWorkspace(m, 1, s.Seq)
	if err != nil {
		t.Fatal(err)
	}
	var evaluation int64
	for _, b := range evaluationExactBuffers(ew) {
		evaluation += int64(cap(b)) * 4
	}
	if a.BackwardWorkspace != scratch || a.EvaluationWorkspace != evaluation {
		t.Fatal("plan does not match allocated workspace payload", a, scratch, evaluation)
	}
	for _, budget := range []int64{0, p.RequiredBytes - 1, p.RequiredBytes, p.RequiredBytes + 1} {
		bounded, err := newTrainingMemoryPlan(m.Config, s, input, budget)
		if err != nil {
			t.Fatal(err)
		}
		want := budget == 0 || budget >= p.RequiredBytes
		if bounded.RequiredBytes != p.RequiredBytes || bounded.FitsBudget != want || (bounded.checkBudget() == nil) != want {
			t.Fatal("budget boundary changed or rounded down", bounded)
		}
	}
	for _, mib := range []int64{0, 1, math.MaxInt64 / (1 << 20)} {
		got, err := memoryBudgetBytes(mib)
		if err != nil || got != mib*(1<<20) {
			t.Fatal("valid budget conversion", mib, got, err)
		}
	}
	for _, mib := range []int64{-1, math.MaxInt64/(1<<20) + 1, math.MaxInt64} {
		if _, err := memoryBudgetBytes(mib); err == nil {
			t.Fatal("accepted overflowing/negative budget", mib)
		}
	}
}

func TestTrainingMemoryPlanScalingAndValidationPeak(t *testing.T) {
	c := testModel(t).Config
	s := defaultTrainSpec()
	s.Batch, s.Seq = 2, 8
	input := trainingMemoryInput{Kind: "dataset", Shards: 4, Workers: 2, Depth: 3}
	plan := func(c Config, s TrainSpec, input trainingMemoryInput) trainingMemoryPlan {
		t.Helper()
		p, err := newTrainingMemoryPlan(c, s, input, 0)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	base := plan(c, s, input)
	s.Accum = 1024
	accum := plan(c, s, input)
	if accum.Components != base.Components || accum.RequiredBytes != base.RequiredBytes {
		t.Fatal("serial accumulation multiplied live memory")
	}
	s.Batch *= 2
	batch := plan(c, s, input)
	if batch.Components.Weights != base.Components.Weights || batch.Components.EvaluationWorkspace != base.Components.EvaluationWorkspace || batch.Components.BackwardWorkspace != 2*base.Components.BackwardWorkspace || batch.RequiredBytes <= base.RequiredBytes {
		t.Fatal("batch scaling changed model/evaluation ownership")
	}
	c.Layers++
	layers := plan(c, s, input)
	if layers.Components.Weights <= batch.Components.Weights || layers.Components.TrainingActivationBound <= batch.Components.TrainingActivationBound || layers.Components.BackwardWorkspace != batch.Components.BackwardWorkspace || layers.Components.EvaluationWorkspace != batch.Components.EvaluationWorkspace {
		t.Fatal("scratch incorrectly scales with layers")
	}
	input.Depth++
	deeper := plan(c, s, input)
	if deeper.TrainingBytes <= layers.TrainingBytes || deeper.EvaluationBytes <= layers.EvaluationBytes {
		t.Fatal("training prefetch not charged during both phases")
	}
	// Many tiny shards and a one-token window make validation's second planner
	// and reader dominate the activation payload. Max must include this phase.
	c = Config{Vocab: baseVocab, Dim: 2, Hidden: 1, Layers: 1, Heads: 1, KVHeads: 1, Context: 2, RopeTheta: 10000, NormEps: 1e-5}
	s.Batch, s.Seq = 1, 1
	input.Shards = maxDatasetShards
	validation := plan(c, s, input)
	if validation.EvaluationBytes <= validation.TrainingBytes || validation.RequiredBytes != validation.EvaluationBytes {
		t.Fatal("missed validation peak", validation)
	}
	if validation.EvaluationBytes < validation.PersistentBytes+validation.Components.EvaluationWorkspace+validation.Components.EvaluationInputAllowance {
		t.Fatal("dropped trainer-owned buffers during evaluation")
	}
}

func TestTrainingMemoryPlanCoversRetainedTextTokenCapacity(t *testing.T) {
	c := testModel(t).Config
	s := defaultTrainSpec()
	s.Seq = 1
	for _, tok := range []Tokenizer{{}, {Merges: []Pair{{'a', 'a'}, {baseVocab, baseVocab}}}} {
		c.Vocab = tok.Vocab()
		for _, size := range []int{16, 255, 256, 257, 1023, 1024, 1025, 65536} {
			text := strings.Repeat("a", size)
			train, val, err := splitCorpus(tok, text, s.ValFraction, s.Seq)
			if err != nil {
				t.Fatal(err)
			}
			p, err := newTrainingMemoryPlan(c, s, trainingMemoryInput{Kind: "text", TextBytes: int64(size)}, 0)
			if err != nil {
				t.Fatal(err)
			}
			retained := int64(len(text)) + 8*int64(cap(train)+cap(val))
			if p.Components.CorpusAllowance < retained {
				t.Fatal("text capacity exceeds plan", size, tok, retained, p.Components.CorpusAllowance)
			}
		}
	}
}

func TestTrainingMemoryPlanRejectsInvalidShapesBeforeMultiplication(t *testing.T) {
	c := testModel(t).Config
	s := defaultTrainSpec()
	input := trainingMemoryInput{Kind: "text", TextBytes: 100}
	for _, mutate := range []func(*Config, *TrainSpec, *trainingMemoryInput){
		func(c *Config, _ *TrainSpec, _ *trainingMemoryInput) { c.Dim = math.MaxInt },
		func(c *Config, _ *TrainSpec, _ *trainingMemoryInput) { c.Layers = math.MaxInt },
		func(c *Config, _ *TrainSpec, _ *trainingMemoryInput) { c.Heads = 0 },
		func(_ *Config, s *TrainSpec, _ *trainingMemoryInput) { s.Batch = math.MaxInt },
		func(_ *Config, s *TrainSpec, _ *trainingMemoryInput) { s.Seq = math.MaxInt },
		func(_ *Config, s *TrainSpec, _ *trainingMemoryInput) { s.Accum = 0 },
		func(c *Config, s *TrainSpec, _ *trainingMemoryInput) {
			*c, _ = preset("base")
			s.Batch, s.Seq = 4096, c.Context
		},
		func(_ *Config, _ *TrainSpec, i *trainingMemoryInput) { i.TextBytes = math.MaxInt64 },
		func(_ *Config, _ *TrainSpec, i *trainingMemoryInput) { i.Kind = "unknown" },
		func(_ *Config, _ *TrainSpec, i *trainingMemoryInput) {
			*i = trainingMemoryInput{Kind: "dataset", Shards: 2, Workers: 3, Depth: 2}
		},
		func(_ *Config, _ *TrainSpec, i *trainingMemoryInput) {
			*i = trainingMemoryInput{Kind: "dataset", Shards: math.MaxInt, Workers: 1, Depth: 1}
		},
	} {
		cc, ss, ii := c, s, input
		mutate(&cc, &ss, &ii)
		if _, err := newTrainingMemoryPlan(cc, ss, ii, math.MaxInt64); err == nil {
			t.Fatal("accepted invalid plan", cc, ss, ii)
		}
	}
	// Increasing the optional cap must not bypass the existing staging limit.
	c = Config{Vocab: baseVocab, Dim: 2, Hidden: 1, Layers: 1, Heads: 1, KVHeads: 1, Context: 32768, RopeTheta: 10000, NormEps: 1e-5}
	s.Batch, s.Seq = 1, 32768
	if _, err := newTrainingMemoryPlan(c, s, trainingMemoryInput{Kind: "dataset", Shards: 2, Workers: 1, Depth: 64}, math.MaxInt64); err != nil {
		t.Fatal("exact staging boundary rejected", err)
	}
	s.Batch = 2
	if _, err := newTrainingMemoryPlan(c, s, trainingMemoryInput{Kind: "dataset", Shards: 2, Workers: 1, Depth: 64}, math.MaxInt64); err == nil || !strings.Contains(err.Error(), "prefetch staging") {
		t.Fatal("bypassed staging limit", err)
	}
}

// Capture only the CLI plan. No test in this suite changes stdout in parallel.
func captureTrainingPlan(t *testing.T, args []string) (trainingMemoryPlan, error) {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "stdout")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	original := os.Stdout
	os.Stdout = f
	defer func() { os.Stdout = original }()
	err = runTrain(context.Background(), append(append([]string{}, args...), "-memory-plan"))
	if _, seekErr := f.Seek(0, io.SeekStart); seekErr != nil {
		t.Fatal(seekErr)
	}
	var p trainingMemoryPlan
	decodeErr := json.NewDecoder(f).Decode(&p)
	if decodeErr != nil && !(err != nil && errors.Is(decodeErr, io.EOF)) {
		t.Fatal("invalid memory-plan JSON", decodeErr, err)
	}
	return p, err
}

func TestMemoryPlanCLIRejectsBeforeTensorAllocationAndOutputs(t *testing.T) {
	old := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(old)
	dir := t.TempDir()
	data := filepath.Join(dir, "input.txt")
	text := strings.Repeat("one two three four\n", 40)
	if err := os.WriteFile(data, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	checkpoint := filepath.Join(dir, "resume.mglm")
	// This checkpoint has about 50 MiB of parameter/moment tensors. The CLI
	// must scan its checksum but allocate none of those tensors for a plan.
	func() {
		c, _ := preset("small")
		m, err := allocateModel(c, nil)
		if err != nil {
			t.Fatal(err)
		}
		ensureMoments(m)
		s := &TrainState{Spec: defaultTrainSpec(), CorpusSHA256: corpusHash(text)}
		if err := SaveCheckpoint(checkpoint, m, Tokenizer{}, s); err != nil {
			t.Fatal(err)
		}
	}()
	for _, resume := range []bool{false, true} {
		for _, dry := range []bool{false, true} {
			for _, mib := range []int{0, 1} {
				if !dry && mib == 0 {
					continue
				}
				t.Run(strconv.FormatBool(resume)+"/"+strconv.FormatBool(dry)+"/"+strconv.Itoa(mib), func(t *testing.T) {
					outDir := t.TempDir()
					paths := []string{filepath.Join(outDir, "model"), filepath.Join(outDir, "events"), filepath.Join(outDir, "cpu"), filepath.Join(outDir, "alloc")}
					args := []string{"-data", data, "-memory-mib", strconv.Itoa(mib), "-out", paths[0], "-metrics", paths[1], "-cpu-profile", paths[2], "-alloc-profile", paths[3]}
					if resume {
						args = append(args, "-resume", checkpoint)
					} else {
						args = append(args, "-preset", "small")
					}
					runtime.GC()
					var before, after runtime.MemStats
					runtime.ReadMemStats(&before)
					var p trainingMemoryPlan
					var err error
					if dry {
						p, err = captureTrainingPlan(t, args)
					} else {
						err = runTrain(context.Background(), args)
					}
					runtime.ReadMemStats(&after)
					if mib > 0 && (err == nil || !strings.Contains(err.Error(), "exceeds -memory-mib")) || mib == 0 && err != nil {
						t.Fatal("wrong budget result", err)
					}
					if dry && (p.Format != "monolith-training-memory-v1" || p.FitsBudget != (mib == 0) || p.RequiredBytes < 50<<20) {
						t.Fatal("incomplete dry-run plan", p)
					}
					allocated := after.TotalAlloc - before.TotalAlloc
					if allocated >= 4<<20 {
						t.Fatalf("plan/rejection allocated %d bytes; tensors were loaded", allocated)
					}
					t.Logf("allocated %d bytes; tensor allocation guard is %d bytes", allocated, 4<<20)
					for _, path := range paths {
						if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
							t.Fatal("plan/rejection created an output", path, err)
						}
					}
				})
			}
		}
	}
	// A plan is not a way around integrity checks, including a tensor checksum.
	f, err := os.OpenFile(checkpoint, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte{0xff}, 8192); err != nil {
		t.Fatal(err)
	}
	f.Close()
	if _, err := captureTrainingPlan(t, []string{"-data", data, "-resume", checkpoint}); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatal("dry run accepted corrupt checkpoint", err)
	}
}

func TestMemoryBudgetDoesNotChangeCheckpointsOrResume(t *testing.T) {
	old := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(old)
	for _, source := range []string{"text", "dataset"} {
		t.Run(source, func(t *testing.T) {
			dir := t.TempDir()
			config := filepath.Join(dir, "config.json")
			if err := writeJSONFile(config, testModel(t).Config); err != nil {
				t.Fatal(err)
			}
			tok := Tokenizer{Merges: []Pair{{'o', 'n'}, {baseVocab, 'e'}}}
			tokenizer := filepath.Join(dir, "tokenizer.json")
			if err := writeJSONFile(tokenizer, tok); err != nil {
				t.Fatal(err)
			}
			data := filepath.Join(dir, "input.txt")
			if err := os.WriteFile(data, []byte(strings.Repeat("one two three four\n", 20)), 0600); err != nil {
				t.Fatal(err)
			}
			common := []string{"-data", data}
			freshOnly := []string{"-tokenizer", tokenizer}
			if source == "dataset" {
				path, _ := preparedDataset(t, tok, "one\ntwo\nthree\n", "four\nfive\n", 12)
				common = []string{"-dataset", path, "-data-workers", "1", "-prefetch", "2"}
				freshOnly = nil
			}
			common = append(common, "-eval-every", "2", "-eval-batches", "2", "-save-every", "0")
			fresh := append(append(append([]string{}, common...), freshOnly...), "-config", config, "-steps", "4", "-warmup", "1", "-seq", "4", "-batch", "2", "-accum", "2", "-seed", "9")
			run := func(args []string) {
				t.Helper()
				if err := runTrain(context.Background(), args); err != nil {
					t.Fatal(err)
				}
			}
			plain, bounded, partial := filepath.Join(dir, "plain"), filepath.Join(dir, "bounded"), filepath.Join(dir, "partial")
			run(append(append([]string{}, fresh...), "-out", plain))
			metrics := filepath.Join(dir, "metrics")
			run(append(append([]string{}, fresh...), "-out", bounded, "-memory-mib", "4", "-metrics", metrics))
			planned, err := captureTrainingPlan(t, append(append([]string{}, fresh...), "-memory-mib", "4"))
			if err != nil {
				t.Fatal(err)
			}
			actual := readTrainingEvents(t, checkpointBytes(t, metrics))[0].Run.MemoryPlan
			if actual == nil || !reflect.DeepEqual(*actual, planned) {
				t.Fatal("telemetry plan differs from dry-run plan", actual, planned)
			}
			run(append(append([]string{}, fresh...), "-out", partial, "-stop-after", "2", "-memory-mib", "4"))
			resumeArgs := append(append([]string{}, common...), "-resume", partial, "-out", partial, "-memory-mib", "4")
			resumedPlan, err := captureTrainingPlan(t, resumeArgs)
			if err != nil || !reflect.DeepEqual(resumedPlan, planned) {
				t.Fatal("resume plan ignores saved model/spec", resumedPlan, planned, err)
			}
			resumeArgs = append(resumeArgs, "-memory-mib", "8")
			if source == "dataset" {
				resumeArgs = append(resumeArgs, "-data-workers", "2", "-prefetch", "3")
			}
			run(resumeArgs)
			if !bytes.Equal(checkpointBytes(t, plain), checkpointBytes(t, bounded)) || !bytes.Equal(checkpointBytes(t, plain), checkpointBytes(t, partial)) {
				t.Fatal("memory budget changed the training trajectory or checkpoint format")
			}
		})
	}
}

func TestCheckpointBudgetGapKeepsOriginalDescriptor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "checkpoint")
	m := testModel(t)
	ensureMoments(m)
	m.Params[0].M[0] = .125
	s := &TrainState{Spec: defaultTrainSpec(), CorpusSHA256: corpusHash("fixture")}
	if err := SaveCheckpoint(path, m, Tokenizer{}, s); err != nil {
		t.Fatal(err)
	}
	original := checkpointBytes(t, path)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	checked, err := readCheckpointMetadata(f)
	if err != nil {
		t.Fatal(err)
	}
	// Replace the pathname after metadata/budget inspection, before tensor load.
	m.Params[0].Data[0]++
	m.Params[0].M[0] = .5
	if err := SaveCheckpoint(path, m, Tokenizer{}, s); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadCheckpointTensors(f, checked)
	if err != nil {
		t.Fatal(err)
	}
	copyPath := filepath.Join(t.TempDir(), "loaded")
	if err := SaveCheckpoint(copyPath, loaded, checked.meta.Tokenizer, checked.meta.Training); err != nil {
		t.Fatal(err)
	}
	if checked.identity != corpusHash(string(original)) || !bytes.Equal(original, checkpointBytes(t, copyPath)) {
		t.Fatal("budget inspection and tensor load used different checkpoints")
	}
}
