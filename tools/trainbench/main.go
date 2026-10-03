// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0
// Trainbench compares two local executables on fixed, authored data. It never
// downloads data or trains a tokenizer. Run from the repository root.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"time"
)

type run struct {
	Implementation   string  `json:"implementation"`
	Repetition       int     `json:"repetition"`
	WallSeconds      float64 `json:"wall_seconds"`
	UserSeconds      float64 `json:"user_seconds"`
	SystemSeconds    float64 `json:"system_seconds"`
	CheckpointSHA256 string  `json:"checkpoint_sha256"`
}
type scenario struct {
	Preset     string          `json:"preset"`
	Steps      int             `json:"steps"`
	Arguments  []string        `json:"arguments"`
	Runs       []run           `json:"runs"`
	Evaluation json.RawMessage `json:"evaluation"`
}
type report struct {
	Format           string     `json:"format"`
	BaselineSHA256   string     `json:"baseline_binary_sha256"`
	CandidateSHA256  string     `json:"candidate_binary_sha256"`
	TrainSHA256      string     `json:"train_text_sha256"`
	ValidationSHA256 string     `json:"validation_text_sha256"`
	GoVersion        string     `json:"harness_go_version"`
	GOOS             string     `json:"goos"`
	GOARCH           string     `json:"goarch"`
	Cases            []scenario `json:"cases"`
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func main() {
	if err := execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func execute() error {
	baseline := flag.String("baseline", "", "baseline executable")
	candidate := flag.String("candidate", "", "candidate executable")
	out := flag.String("out", "", "new result directory")
	repeats := flag.Int("repeats", 5, "paired repetitions (2..100), alternating order")
	flag.Parse()
	if *baseline == "" || *candidate == "" || *out == "" || *repeats < 2 || *repeats > 100 || flag.NArg() != 0 {
		return fmt.Errorf("provide -baseline, -candidate, -out and 2..100 repetitions")
	}
	var err error
	*baseline, err = filepath.Abs(*baseline)
	if err != nil {
		return err
	}
	*candidate, err = filepath.Abs(*candidate)
	if err != nil {
		return err
	}
	r := report{Format: "monolith-training-comparison-v1", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH}
	if r.BaselineSHA256, err = hashFile(*baseline); err != nil {
		return err
	}
	if r.CandidateSHA256, err = hashFile(*candidate); err != nil {
		return err
	}
	train, validation := "examples/heldout/train.txt", "examples/heldout/validation.txt"
	if r.TrainSHA256, err = hashFile(train); err != nil {
		return err
	}
	if r.ValidationSHA256, err = hashFile(validation); err != nil {
		return err
	}
	if err = os.Mkdir(*out, 0755); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	dataset := filepath.Join(*out, "dataset")
	cmd := exec.CommandContext(ctx, *baseline, "prepare", "-train", train, "-validation", validation, "-out", dataset, "-shard-tokens", "2048")
	if b, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("prepare: %w: %s", err, b)
	}
	for _, spec := range []struct {
		preset string
		steps  int
	}{{"tiny", 160}, {"small", 40}} {
		common := []string{"-preset", spec.preset, "-steps", strconv.Itoa(spec.steps), "-warmup", "5", "-seq", "64", "-batch", "2", "-accum", "1", "-threads", "4", "-seed", "73", "-data-workers", "2", "-prefetch", "4", "-eval-every", "0", "-eval-batches", "0", "-save-every", "0", "-log-every", strconv.Itoa(spec.steps)}
		c := scenario{Preset: spec.preset, Steps: spec.steps, Arguments: common}
		expected := ""
		for rep := 0; rep < *repeats; rep++ {
			order := []string{"baseline", "candidate"}
			if rep%2 == 1 {
				order[0], order[1] = order[1], order[0]
			}
			for _, kind := range order {
				binary := *baseline
				if kind == "candidate" {
					binary = *candidate
				}
				prefix := filepath.Join(*out, fmt.Sprintf("%s-%02d-%s", spec.preset, rep+1, kind))
				checkpoint := prefix + ".mglm"
				log, err := os.Create(prefix + ".log")
				if err != nil {
					return err
				}
				args := append([]string{"train", "-dataset", filepath.Join(dataset, "manifest.json"), "-out", checkpoint}, common...)
				cmd := exec.CommandContext(ctx, binary, args...)
				cmd.Stdout, cmd.Stderr = log, log
				start := time.Now()
				runErr := cmd.Run()
				elapsed := time.Since(start).Seconds()
				closeErr := log.Close()
				if runErr != nil {
					return fmt.Errorf("%s: %w (see log)", prefix, runErr)
				}
				if closeErr != nil {
					return closeErr
				}
				digest, err := hashFile(checkpoint)
				if err != nil {
					return err
				}
				if expected == "" {
					expected = digest
				}
				if digest != expected {
					return fmt.Errorf("%s: checkpoint differs from matched baseline", prefix)
				}
				c.Runs = append(c.Runs, run{kind, rep + 1, elapsed, cmd.ProcessState.UserTime().Seconds(), cmd.ProcessState.SystemTime().Seconds(), digest})
				fmt.Fprintf(os.Stderr, "%s %s repeat=%d wall=%.3fs\n", spec.preset, kind, rep+1, elapsed)
				// Evaluate both executables outside the timed training interval. The
				// receipt must agree byte for byte, not just after rounding its loss.
				if rep == 0 {
					eval, err := exec.CommandContext(ctx, binary, "eval", "-model", checkpoint, "-dataset", filepath.Join(dataset, "manifest.json"), "-seq", "64", "-threads", "4").Output()
					if err != nil {
						return fmt.Errorf("evaluation: %w", err)
					}
					if len(c.Evaluation) == 0 {
						c.Evaluation = bytes.TrimSpace(eval)
					} else if !bytes.Equal(c.Evaluation, bytes.TrimSpace(eval)) {
						return fmt.Errorf("evaluation receipts differ")
					}
				}
				// Keep one checkpoint per implementation/scenario for inspection. Other
				// copies are redundant: their complete hashes are already in the report.
				if rep > 0 {
					if err = os.Remove(checkpoint); err != nil {
						return err
					}
				}
			}
		}
		r.Cases = append(r.Cases, c)
	}
	f, err := os.OpenFile(filepath.Join(*out, "comparison.json"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	e := json.NewEncoder(f)
	e.SetIndent("", "  ")
	err = e.Encode(r)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
