package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type hfReference struct {
	IDs       []int                        `json:"ids"`
	Logits    [][]float32                  `json:"logits"`
	Cache     []struct{ K, V [][]float32 } `json:"cache"`
	WeightSHA string                       `json:"weight_sha256"`
}

func hfNear(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	for i, v := range got {
		if !finite(float64(v)) || math.Abs(float64(v-want[i])) > tolerance {
			t.Fatalf("element %d: got %.9g, want %.9g, abs tolerance %g", i, v, want[i], tolerance)
		}
	}
}
func TestHFLlamaTransformersReference(t *testing.T) {
	for _, dtype := range []string{"f32", "f16", "bf16"} {
		t.Run(dtype, func(t *testing.T) {
			dir := filepath.Join("testdata", "hf-llama", dtype)
			var ref hfReference
			b, err := os.ReadFile(filepath.Join(dir, "reference.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(b, &ref); err != nil {
				t.Fatal(err)
			}
			weights, err := os.ReadFile(filepath.Join(dir, "model.safetensors"))
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(weights)
			if hex.EncodeToString(sum[:]) != ref.WeightSHA {
				t.Fatal("reference/weight identity mismatch")
			}
			m, _, err := loadHFModel(context.Background(), dir, 64, 1)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range m.Params {
				if len(p.Grad)+len(p.M)+len(p.V) != 0 {
					t.Fatal("allocated optimizer storage")
				}
			}
			// Public scalar path and private fixed-worker path share the oracle.
			for _, workers := range []int{1, 2, 4} {
				exec := newKernelExecutor(workers)
				for _, chunk := range []int{1, 4, 16, 35} {
					cache := m.NewCache()
					for lo := 0; lo < 35; lo += chunk {
						hi := min(lo+chunk, 35)
						got, err := m.prefillBatch(context.Background(), []*KVCache{cache}, [][]int{ref.IDs[lo:hi]}, exec)
						if err != nil {
							t.Fatal(err)
						}
						hfNear(t, got, ref.Logits[hi-1], 2e-5)
					}
					for pos := 35; pos < len(ref.IDs); pos++ {
						got, err := m.Step(cache, ref.IDs[pos])
						if err != nil {
							t.Fatal(err)
						}
						hfNear(t, got, ref.Logits[pos], 2e-5)
					}
					for layer := range m.Blocks {
						for pos := range ref.IDs {
							k, v := cache.row(layer, pos)
							hfNear(t, k, ref.Cache[layer].K[pos], 2e-5)
							hfNear(t, v, ref.Cache[layer].V[pos], 2e-5)
						}
					}
					cache.Close()
				}
				exec.close()
			}
			// Ragged sessions keep independent positions during shared prefill.
			ca, cb := m.NewCache(), m.NewCache()
			defer ca.Close()
			defer cb.Close()
			got, err := m.PrefillBatch(context.Background(), []*KVCache{ca, cb}, [][]int{ref.IDs[:35], ref.IDs[:4]})
			if err != nil {
				t.Fatal(err)
			}
			hfNear(t, got[:m.Config.Vocab], ref.Logits[34], 2e-5)
			hfNear(t, got[m.Config.Vocab:], ref.Logits[3], 2e-5)
			got, err = m.DecodeBatch(context.Background(), []*KVCache{ca, cb}, []int{ref.IDs[35], ref.IDs[4]})
			if err != nil {
				t.Fatal(err)
			}
			hfNear(t, got[:m.Config.Vocab], ref.Logits[35], 2e-5)
			hfNear(t, got[m.Config.Vocab:], ref.Logits[4], 2e-5)
			if err := SaveCheckpoint(filepath.Join(t.TempDir(), "bad.mglm"), m, Tokenizer{}, nil); err == nil {
				t.Fatal("saved HF model as native checkpoint")
			}
		})
	}
}
func TestHFProbeAndLoadBoundaries(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := loadHFModel(ctx, "missing", 0, 1); err != context.Canceled {
		t.Fatal(err)
	}
	m, _, err := loadHFModel(context.Background(), "testdata/hf-llama/f32", 64, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, ids := range [][]int{nil, {-1}, {258}, make([]int, 64)} {
		if _, err := inferHFIDs(context.Background(), m, ids, 2, 16, false); err == nil {
			t.Fatal("accepted invalid probe", ids)
		}
	}
	out, err := inferHFIDs(context.Background(), m, []int{1, 256}, 3, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.TokenIDs) == 0 || len(out.Logits) != len(out.TokenIDs) {
		t.Fatal(out)
	}
	for i, logits := range out.Logits {
		best := 0
		for j, v := range logits {
			if v > logits[best] {
				best = j
			}
		}
		if out.TokenIDs[i] != best {
			t.Fatal("greedy parity")
		}
	}
}
