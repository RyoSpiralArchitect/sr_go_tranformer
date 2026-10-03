// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"testing"
	"time"
)

func readTrainingEvents(t *testing.T, data []byte) []trainingEvent {
	t.Helper()
	d := json.NewDecoder(bytes.NewReader(data))
	var events []trainingEvent
	for {
		var event trainingEvent
		err := d.Decode(&event)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if event.Format != "monolith-train-v1" {
			t.Fatal("unknown event format", event)
		}
		events = append(events, event)
	}
	return events
}

func TestTelemetryCLIExactCheckpointsAndResume(t *testing.T) {
	old := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(old)
	for _, source := range []string{"text", "dataset"} {
		t.Run(source, func(t *testing.T) {
			dir := t.TempDir()
			config := filepath.Join(dir, "config.json")
			b, _ := json.Marshal(testModel(t).Config)
			if err := os.WriteFile(config, b, 0600); err != nil {
				t.Fatal(err)
			}
			data := filepath.Join(dir, "data.txt")
			if err := os.WriteFile(data, []byte(strings.Repeat("one two three four\n", 15)), 0600); err != nil {
				t.Fatal(err)
			}
			sourceArgs := []string{"-data", data}
			if source == "dataset" {
				path, _ := preparedDataset(t, Tokenizer{}, "one\ntwo\nthree\n", "four\nfive\n", 12)
				sourceArgs = []string{"-dataset", path, "-data-workers", "2", "-prefetch", "3"}
			}
			common := append(append([]string{}, sourceArgs...), "-eval-every", "2", "-save-every", "2", "-eval-batches", "2", "-log-every", "2")
			fresh := append(append([]string{}, common...), "-config", config, "-steps", "4", "-warmup", "1", "-seq", "4", "-batch", "2", "-accum", "2", "-seed", "9")
			plain, observed, resumed := filepath.Join(dir, "plain"), filepath.Join(dir, "observed"), filepath.Join(dir, "resumed")
			run := func(args ...string) {
				t.Helper()
				if err := runTrain(context.Background(), args); err != nil {
					t.Fatal(err)
				}
			}
			run(append(append([]string{}, fresh...), "-out", plain)...)
			metrics := filepath.Join(dir, "events.jsonl")
			extra := []string{"-out", observed, "-metrics", metrics}
			if source == "dataset" {
				extra = append(extra, "-cpu-profile", filepath.Join(dir, "cpu.pprof"), "-alloc-profile", filepath.Join(dir, "alloc.pprof"))
			}
			run(append(append([]string{}, fresh...), extra...)...)
			if !bytes.Equal(checkpointBytes(t, plain), checkpointBytes(t, observed)) {
				t.Fatal("telemetry changed checkpoint")
			}
			events := readTrainingEvents(t, checkpointBytes(t, metrics))
			first, last := events[0], events[len(events)-1]
			if first.Event != "start" || first.Step != 0 || first.Run == nil || first.Run.Source.Kind != source || first.Run.InitializationSeed == nil || *first.Run.InitializationSeed != 9 || first.Run.GoVersion != runtime.Version() {
				t.Fatal("incomplete start", first)
			}
			if last.Event != "end" || last.Status != "completed" || last.Step != 4 || last.TokensSeen != 64 || last.Phases.Input <= 0 || last.Phases.ZeroGrad <= 0 || last.Phases.Forward <= 0 || last.Phases.Backward <= 0 || last.Phases.Optimizer <= 0 || last.Phases.Evaluation <= 0 || last.Phases.Checkpoint <= 0 || last.AllocBytes == 0 {
				t.Fatal("incomplete end", last)
			}
			updates, prev := 0, int64(0)
			for _, e := range events {
				if e.ElapsedNS < prev {
					t.Fatal("time moved backwards")
				}
				prev = e.ElapsedNS
				if e.Event == "update" {
					updates++
					if e.Loss == nil || e.GradNorm == nil {
						t.Fatal("missing update metrics")
					}
				}
			}
			if updates != 2 {
				t.Fatal("unexpected record frequency", updates)
			}
			if source == "dataset" {
				if last.Cursor == nil || last.ReadTokens < last.TokensSeen || last.ReadNS <= 0 {
					t.Fatal("missing IO counters", last)
				}
				for _, name := range []string{"cpu.pprof", "alloc.pprof"} {
					profile := checkpointBytes(t, filepath.Join(dir, name))
					if len(profile) < 50 || !bytes.Equal(profile[:2], []byte{0x1f, 0x8b}) {
						t.Fatal("profile not flushed", name)
					}
				}
			}
			run(append(append([]string{}, fresh...), "-out", resumed, "-stop-after", "2")...)
			digest := corpusHash(string(checkpointBytes(t, resumed)))
			resumeMetrics := filepath.Join(dir, "resume.jsonl")
			run(append(append([]string{}, common...), "-resume", resumed, "-out", resumed, "-metrics", resumeMetrics)...)
			if !bytes.Equal(checkpointBytes(t, plain), checkpointBytes(t, resumed)) {
				t.Fatal("telemetry changed resume")
			}
			start := readTrainingEvents(t, checkpointBytes(t, resumeMetrics))[0]
			if start.Step != 2 || start.TokensSeen != 32 || start.Run.ResumeSHA256 != digest || start.Run.InitializationSeed != nil {
				t.Fatal("ambiguous resume boundary", start)
			}
		})
	}
}

func TestTelemetryOutputOwnership(t *testing.T) {
	dir := t.TempDir()
	existing := filepath.Join(dir, "input")
	if err := os.WriteFile(existing, []byte("preserve me"), 0600); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(t.TempDir(), "alias")
	if err := os.Symlink(dir, alias); err != nil {
		t.Fatal(err)
	}
	for _, o := range []telemetryOptions{
		{Metrics: existing, Checkpoint: filepath.Join(dir, "model")},
		{Metrics: filepath.Join(dir, "same"), Checkpoint: filepath.Join(dir, "same")},
		{Metrics: filepath.Join(alias, "same"), Checkpoint: filepath.Join(dir, "same")},
		{Metrics: filepath.Join(alias, "new", "same"), Checkpoint: filepath.Join(dir, "new", "same")},
		{Metrics: filepath.Join(dir, "same"), CPUProfile: filepath.Join(alias, "same"), Checkpoint: filepath.Join(dir, "model")},
		{Metrics: filepath.Join(dir, "clean-on-failure"), AllocProfile: existing, Checkpoint: filepath.Join(dir, "model")},
	} {
		if observer, err := openTrainingObserver(o); err == nil {
			_ = observer.close()
			t.Fatal("unsafe outputs accepted", o)
		}
	}
	if string(checkpointBytes(t, existing)) != "preserve me" {
		t.Fatal("input overwritten")
	}
	if _, err := os.Stat(filepath.Join(dir, "clean-on-failure")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("setup left its partial output", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "INPUT")); err == nil {
		// The temporary directory is on a case-insensitive filesystem.
		o := telemetryOptions{Metrics: filepath.Join(dir, "FUTURE"), Checkpoint: filepath.Join(dir, "future")}
		if observer, err := openTrainingObserver(o); err == nil {
			_ = observer.close()
			t.Fatal("case-folded checkpoint alias accepted")
		}
	}
}

func TestCPUProfileWriteFailureIsReported(t *testing.T) {
	w := &profileWriter{Writer: &failTrainingWriter{writes: 3}}
	if err := pprof.StartCPUProfile(w); err != nil {
		t.Fatal(err)
	}
	o := &trainingObserver{cpu: true, cpuWriter: w}
	if err := o.close(); err == nil || !strings.Contains(err.Error(), "injected metrics failure") {
		t.Fatal("lost profile write error", err)
	}
}

type failTrainingWriter struct{ writes int }

func (w *failTrainingWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes >= 3 {
		return 0, errors.New("injected metrics failure")
	}
	return len(p), nil
}
func TestTelemetryWriteFailureSavesCompletedUpdate(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "one\ntwo\n", "three\n", 12)
	m, state := testModel(t), datasetTestState(d)
	observer := &trainingObserver{encoder: json.NewEncoder(&failTrainingWriter{}), start: time.Now()}
	runtime.ReadMemStats(&observer.baseline)
	out := filepath.Join(t.TempDir(), "model")
	err := trainDatasetLoop(context.Background(), m, Tokenizer{}, state, d, 1, 1, loopOptions{Out: out, LogEvery: 1, Observer: observer})
	if err == nil || !strings.Contains(err.Error(), "injected metrics failure") {
		t.Fatal(err)
	}
	loaded, tok, saved, err := LoadCheckpoint(out)
	if err != nil || saved.Step != 1 {
		t.Fatal("lost committed boundary", saved, err)
	}
	if err = validateDatasetResume(d, tok, saved); err != nil {
		t.Fatal(err)
	}
	if err = trainDatasetLoop(context.Background(), loaded, tok, saved, d, 1, 1, loopOptions{Out: out, LogEvery: 6}); err != nil {
		t.Fatal(err)
	}
	full := filepath.Join(t.TempDir(), "full")
	if err = trainDatasetLoop(context.Background(), testModel(t), Tokenizer{}, datasetTestState(d), d, 1, 1, loopOptions{Out: full, LogEvery: 6}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(checkpointBytes(t, out), checkpointBytes(t, full)) {
		t.Fatal("metrics failure lost exact resume")
	}
}

func TestTelemetryCancellationEnd(t *testing.T) {
	m := testModel(t)
	spec := defaultTrainSpec()
	spec.Steps = 2
	spec.Warmup = 1
	spec.Batch = 1
	spec.Seq = 4
	s := &TrainState{Spec: spec, RNG: RNG{3}, CorpusSHA256: corpusHash("fixture")}
	var b bytes.Buffer
	observer := &trainingObserver{encoder: json.NewEncoder(&b), start: time.Now()}
	runtime.ReadMemStats(&observer.baseline)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := filepath.Join(t.TempDir(), "model")
	err := trainingLoop(ctx, m, Tokenizer{}, s, 1, 1, loopOptions{Out: out, LogEvery: 1, Observer: observer}, func() (float64, float64, error) {
		loss, norm, err := observedTrainUpdate(ctx, m, s, Tokenizer{}.Encode("some fixture text", true, true), observer)
		cancel()
		return loss, norm, err
	}, func() (float64, int, error) { return 1, 1, nil })
	if err != nil {
		t.Fatal(err)
	}
	events := readTrainingEvents(t, b.Bytes())
	end := events[len(events)-1]
	if end.Event != "end" || end.Status != "canceled" || end.Step != 1 {
		t.Fatal(end)
	}
	_, _, saved, err := LoadCheckpoint(out)
	if err != nil || saved.Step != 1 {
		t.Fatal("missing canceled prefix", err)
	}
}
