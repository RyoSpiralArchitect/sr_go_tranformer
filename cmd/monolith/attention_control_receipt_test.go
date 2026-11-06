package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func TestAttentionControlReceipts(t *testing.T) {
	read := func(name string, target any) {
		t.Helper()
		if filepath.Base(name) != name {
			t.Fatal("invalid receipt name")
		}
		data, err := os.ReadFile(filepath.Join("validation", "performance", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, target); err != nil {
			t.Fatal(err)
		}
	}
	var index struct {
		Runtime  string               `json:"runtime_sha256"`
		Source   string               `json:"timing_test_sha256"`
		Binary   string               `json:"binary_sha256"`
		Receipts []string             `json:"receipts"`
		Medians  map[string][]float64 `json:"approximate_median_ratios"`
	}
	read("attention-control-mac-20261106.json", &index)
	if len(index.Receipts) != 3 || len(index.Medians) != 3 {
		t.Fatal("incomplete controlled experiment")
	}
	for run, name := range index.Receipts {
		var receipt struct {
			Protocol                     string
			CallsPerBlock, TrialsPerKind int
			Hashes                       map[string]string
			Samples                      []attentionControlSample
		}
		read(name, &receipt)
		if receipt.Protocol != "attention-small-controlled-v1" || receipt.CallsPerBlock != 4 || receipt.TrialsPerKind != 16 || len(receipt.Samples) != 48 {
			t.Fatalf("invalid protocol in %s", name)
		}
		if receipt.Hashes["monolith.go"] != index.Runtime || receipt.Hashes["attention_paired_test.go"] != index.Source || receipt.Hashes["attention-controlled-mac.test"] != index.Binary {
			t.Fatalf("inconsistent frozen fingerprint in %s", name)
		}
		for kind, medians := range index.Medians {
			if len(medians) != 3 {
				t.Fatal("incomplete median summaries")
			}
			seen := make(map[int]bool)
			var ratios []float64
			for _, sample := range receipt.Samples {
				if sample.Kind != kind {
					continue
				}
				if sample.Trial < 0 || sample.Trial >= 16 || seen[sample.Trial] || sample.FirstVariant != sample.Trial%2 {
					t.Fatalf("unbalanced trial in %s", name)
				}
				seen[sample.Trial] = true
				for _, ns := range sample.NSPerCall {
					if ns <= 0 || math.IsNaN(ns) || math.IsInf(ns, 0) {
						t.Fatal("invalid duration")
					}
				}
				ratio := sample.NSPerCall[1] / sample.NSPerCall[0]
				if math.IsNaN(sample.Ratio) || math.Abs(sample.Ratio-ratio) > 1e-12*math.Max(1, ratio) {
					t.Fatal("incorrect paired ratio")
				}
				ratios = append(ratios, ratio)
			}
			if len(ratios) != 16 {
				t.Fatalf("incomplete %s trials in %s", kind, name)
			}
			sort.Float64s(ratios)
			if math.Abs((ratios[7]+ratios[8])/2-medians[run]) > 0.00005 {
				t.Fatalf("incorrect median for %s in %s", kind, name)
			}
		}
	}
}
