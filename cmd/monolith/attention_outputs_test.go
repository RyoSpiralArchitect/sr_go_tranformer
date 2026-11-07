package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"
)

// frozenScalarAttend is the pre-change score/reduce loop, kept for bit-exact
// and paired-timing comparisons.
func frozenScalarAttend(q, out, scores []float32, length int, kv func(int) ([]float32, []float32)) float64 {
	hd := len(q)
	scale := float32(1 / math.Sqrt(float64(hd)))
	maxScore := float32(-math.MaxFloat32)
	clear(out)
	for s := 0; s < length; s++ {
		k, _ := kv(s)
		var dot float32
		for j, v := range q {
			dot += v * k[j]
		}
		scores[s] = dot * scale
		if scores[s] > maxScore {
			maxScore = scores[s]
		}
	}
	var sum float64
	for s := 0; s < length; s++ {
		sum += math.Exp(float64(scores[s]) - float64(maxScore))
	}
	lse := float64(maxScore) + math.Log(sum)
	for s := 0; s < length; s++ {
		p := float32(math.Exp(float64(scores[s]) - lse))
		_, v := kv(s)
		for j := range out {
			out[j] += p * v[j]
		}
	}
	return lse
}

// pairedScoreAttend shares each query load across four positions in the score
// pass; every position keeps its ascending-index dot accumulation order.
func pairedScoreAttend(q, out, scores []float32, length int, kv func(int) ([]float32, []float32)) float64 {
	hd := len(q)
	scale := float32(1 / math.Sqrt(float64(hd)))
	maxScore := float32(-math.MaxFloat32)
	clear(out)
	s := 0
	for ; s+3 < length; s += 4 {
		k0, _ := kv(s)
		k1, _ := kv(s + 1)
		k2, _ := kv(s + 2)
		k3, _ := kv(s + 3)
		var d0, d1, d2, d3 float32
		for j, v := range q {
			d0 += v * k0[j]
			d1 += v * k1[j]
			d2 += v * k2[j]
			d3 += v * k3[j]
		}
		scores[s] = d0 * scale
		scores[s+1] = d1 * scale
		scores[s+2] = d2 * scale
		scores[s+3] = d3 * scale
		if scores[s] > maxScore {
			maxScore = scores[s]
		}
		if scores[s+1] > maxScore {
			maxScore = scores[s+1]
		}
		if scores[s+2] > maxScore {
			maxScore = scores[s+2]
		}
		if scores[s+3] > maxScore {
			maxScore = scores[s+3]
		}
	}
	for ; s < length; s++ {
		k, _ := kv(s)
		var dot float32
		for j, v := range q {
			dot += v * k[j]
		}
		scores[s] = dot * scale
		if scores[s] > maxScore {
			maxScore = scores[s]
		}
	}
	var sum float64
	for s := 0; s < length; s++ {
		sum += math.Exp(float64(scores[s]) - float64(maxScore))
	}
	lse := float64(maxScore) + math.Log(sum)
	for s := 0; s < length; s++ {
		p := float32(math.Exp(float64(scores[s]) - lse))
		_, v := kv(s)
		for j := range out {
			out[j] += p * v[j]
		}
	}
	return lse
}

func attendFixture(hd, length, kvheads int, seed uint64) ([]float32, [][]float32, [][]float32) {
	r := &RNG{seed}
	q := make([]float32, hd)
	for i := range q {
		q[i] = float32(r.Normal()) * 0.5
	}
	ks := make([][]float32, length)
	vs := make([][]float32, length)
	for s := 0; s < length; s++ {
		k := make([]float32, hd*kvheads)
		v := make([]float32, hd*kvheads)
		for i := range k {
			k[i] = float32(r.Normal()) * 0.4
			v[i] = float32(r.Normal()) * 0.4
		}
		ks[s], vs[s] = k, v
	}
	return q, ks, vs
}

func exactAttendBits(t *testing.T, fn func(q, out, scores []float32, length int, kv func(int) ([]float32, []float32)) float64, q []float32, ks, vs [][]float32, length, kvHead int, want []float32, wantLSE float64) {
	t.Helper()
	hd := len(q)
	out := make([]float32, hd)
	scores := make([]float32, length)
	kv := func(pos int) ([]float32, []float32) {
		return ks[pos][kvHead*hd : (kvHead+1)*hd], vs[pos][kvHead*hd : (kvHead+1)*hd]
	}
	lse := fn(q, out, scores, length, kv)
	if math.Float64bits(lse) != math.Float64bits(wantLSE) {
		t.Fatalf("lse %v != %v", lse, wantLSE)
	}
	for i := range out {
		if math.Float32bits(out[i]) != math.Float32bits(want[i]) {
			t.Fatalf("out[%d] %v != %v", i, out[i], want[i])
		}
	}
}

func TestPairedScoreAttendExact(t *testing.T) {
	for _, hd := range []int{1, 5, 64, 128} {
		for _, length := range []int{1, 2, 3, 4, 5, 7, 8, 9, 33, 64, 65, 129} {
			for _, kvHead := range []int{0, 1, 2} {
				q, ks, vs := attendFixture(hd, length, 3, uint64(hd*1000+length*10+kvHead))
				hdQ := len(q)
				wantOut := make([]float32, hdQ)
				wantScores := make([]float32, length)
				kv := func(pos int) ([]float32, []float32) {
					return ks[pos][kvHead*hdQ : (kvHead+1)*hdQ], vs[pos][kvHead*hdQ : (kvHead+1)*hdQ]
				}
				wantLSE := frozenScalarAttend(q, wantOut, wantScores, length, kv)
				exactAttendBits(t, pairedScoreAttend, q, ks, vs, length, kvHead, wantOut, wantLSE)
			}
		}
	}
	// Special values in q and k/v rows.
	bits := []uint32{0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000}
	q, ks, vs := attendFixture(64, 37, 2, 77)
	for i := range q {
		q[i] = math.Float32frombits(bits[i%len(bits)]) * 0.25
	}
	for s := range ks {
		for i := range ks[s] {
			if (i+s)%7 == 0 {
				ks[s][i] = math.Float32frombits(bits[(i*3+s)%len(bits)])
				vs[s][i] = math.Float32frombits(bits[(i*5+s+1)%len(bits)])
			}
		}
	}
	wantOut := make([]float32, 64)
	wantScores := make([]float32, 37)
	kv := func(pos int) ([]float32, []float32) { return ks[pos][64:128], vs[pos][64:128] }
	wantLSE := frozenScalarAttend(q, wantOut, wantScores, 37, kv)
	exactAttendBits(t, pairedScoreAttend, q, ks, vs, 37, 1, wantOut, wantLSE)
}

type attendTimingResult struct {
	Name         string    `json:"name"`
	BeforeNS     []float64 `json:"before_ns"`
	AfterNS      []float64 `json:"after_ns"`
	Ratios       []float64 `json:"paired_ratios"`
	MedianRatio  float64   `json:"median_ratio"`
	BeforeAllocs float64   `json:"before_allocs"`
	AfterAllocs  float64   `json:"after_allocs"`
}

func TestPairedScoreAttendTiming(t *testing.T) {
	output := os.Getenv("MONOLITH_ATTEND_SCORES_OUTPUT")
	if output == "" {
		t.Skip("opt-in attend score comparison")
	}
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	artifact := struct {
		Protocol, GoVersion, GOOS, GOARCH string
		Hashes                            map[string]string
		Results                           []attendTimingResult
	}{Protocol: "attend-paired-scores-v1", GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, Hashes: make(map[string]string)}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{executable, "cmd/monolith/monolith.go", "cmd/monolith/attention_outputs_test.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		artifact.Hashes[filepath.Base(path)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	for _, hd := range []int{64, 128} {
		for _, length := range []int{33, 128, 512, 2048} {
			q, ks, vs := attendFixture(hd, length, 1, uint64(hd*7+length))
			hdQ := len(q)
			kv := func(pos int) ([]float32, []float32) { return ks[pos][:hdQ], vs[pos][:hdQ] }
			outA, outB := make([]float32, hdQ), make([]float32, hdQ)
			scoresA, scoresB := make([]float32, length), make([]float32, length)
			functions := []func(){
				func() { frozenScalarAttend(q, outA, scoresA, length, kv) },
				func() { pairedScoreAttend(q, outB, scoresB, length, kv) },
			}
			for warm := 0; warm < 8; warm++ {
				functions[0]()
				functions[1]()
			}
			exactWorkerFloats(t, outA, outB)
			iterations := max(1, (1<<22)/max(1, length*hd))
			result := attendTimingResult{Name: fmt.Sprintf("hd%d/len%d", hd, length)}
			for trial := 0; trial < 32; trial++ {
				var elapsed [2]float64
				for leg := 0; leg < 2; leg++ {
					variant := (trial + leg) % 2
					start := time.Now()
					for i := 0; i < iterations; i++ {
						functions[variant]()
					}
					elapsed[variant] = float64(time.Since(start).Nanoseconds()) / float64(iterations)
				}
				result.BeforeNS = append(result.BeforeNS, elapsed[0])
				result.AfterNS = append(result.AfterNS, elapsed[1])
				result.Ratios = append(result.Ratios, elapsed[1]/elapsed[0])
			}
			ratios := append([]float64(nil), result.Ratios...)
			sort.Float64s(ratios)
			result.MedianRatio = (ratios[(len(ratios)-1)/2] + ratios[len(ratios)/2]) / 2
			result.BeforeAllocs = testing.AllocsPerRun(5, functions[0])
			result.AfterAllocs = testing.AllocsPerRun(5, functions[1])
			exactWorkerFloats(t, outA, outB)
			artifact.Results = append(artifact.Results, result)
			t.Logf("%s: ratio %.4f, allocations %.0f -> %.0f", result.Name, result.MedianRatio, result.BeforeAllocs, result.AfterAllocs)
		}
	}
	encoded, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := file.Write(append(encoded, '\n'))
	closeErr := file.Close()
	if writeErr != nil {
		t.Fatal(writeErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
}
