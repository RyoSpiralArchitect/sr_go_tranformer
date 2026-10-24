// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

// Manual, offline validation: one model/format per process. Ordinary CI skips
// the real-model probe; the metric and mixed-storage contracts still run.
import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

type qualityCorpus struct {
	Format    string   `json:"format"`
	Purpose   string   `json:"purpose"`
	Selection []string `json:"selection_prompts"`
	Documents []struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	} `json:"evaluation_documents"`
	Prompts []string `json:"generation_prompts"`
}

func qualityKL(a, b []float32) float64 {
	ma, mb := float64(a[0]), float64(b[0])
	for i := range a {
		ma = math.Max(ma, float64(a[i]))
		mb = math.Max(mb, float64(b[i]))
	}
	var za, zb float64
	for i := range a {
		za += math.Exp(float64(a[i]) - ma)
		zb += math.Exp(float64(b[i]) - mb)
	}
	la, lb := ma+math.Log(za), mb+math.Log(zb)
	var kl float64
	for i := range a {
		x, y := float64(a[i])-la, float64(b[i])-lb
		kl += math.Exp(x) * (x - y)
	}
	return kl
}

func qualityDistribution(row []float32, target int) (logZ, nll float64, best int, err error) {
	if len(row) == 0 || target < 0 || target >= len(row) {
		err = fmt.Errorf("invalid quality target/row")
		return
	}
	maximum := float64(row[0])
	for i, v := range row {
		if !finite(float64(v)) {
			err = fmt.Errorf("non-finite quality logits")
			return
		}
		if v > row[best] {
			best = i
		}
		maximum = math.Max(maximum, float64(v))
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) - maximum)
	}
	logZ = maximum + math.Log(sum)
	nll = logZ - float64(row[target])
	return
}
func TestQualityDistribution(t *testing.T) {
	a := []float32{float32(math.Log(.75)), float32(math.Log(.25))}
	b := []float32{0, 0}
	_, n, best, err := qualityDistribution(a, 0)
	if err != nil || best != 0 || math.Abs(n+math.Log(.75)) > 1e-7 {
		t.Fatal(n, best, err)
	}
	want := .75*math.Log(1.5) + .25*math.Log(.5)
	if math.Abs(qualityKL(a, b)-want) > 1e-7 || qualityKL(a, a) != 0 {
		t.Fatal("KL metric")
	}
	if _, _, _, err := qualityDistribution([]float32{float32(math.NaN())}, 0); err == nil {
		t.Fatal("nonfinite accepted")
	}
}

func TestQ4MixedStoragePolicy(t *testing.T) {
	for _, dir := range []string{"testdata/hf-llama/f32", "testdata/hf-llama/untied-f32", "testdata/hf-qwen2/untied-bf16", "testdata/hf-gpt2/f16"} {
		m, plan, err := loadHFModelFormat(context.Background(), dir, 64, 1, "q4-mixed")
		if err != nil {
			t.Fatal(err)
		}
		var q8, q4, payload int64
		for _, p := range m.Params {
			if p == m.Emb || p == m.output {
				if p.Q8 == nil || p.Q4 != nil || len(p.Data) != 0 {
					t.Fatal("embedding/head not protected", p.Name)
				}
			} else if p.quantized() && p.Q4 == nil {
				t.Fatal("projection unexpectedly Q8", p.Name)
			}
			if p.Q8 != nil {
				q8 += int64(p.Rows * p.Cols)
				payload += int64(len(p.Q8.Values) + len(p.Q8.Scales)*4)
			} else if p.Q4 != nil {
				q4 += int64(p.Rows * p.Cols)
				payload += int64(len(p.Q4.Values) + len(p.Q4.Scales)*4)
			} else {
				payload += int64(len(p.Data)) * 4
			}
		}
		if plan.Q8Parameters != q8 || plan.Q4Parameters != q4 || plan.WeightBytes != payload || q8+q4 != plan.QuantizedParameters {
			t.Fatal("mixed plan mismatch", plan, q8, q4, payload)
		}
	}
}

type qualityHeader struct {
	Version   string `json:"version"`
	Source    string `json:"source"`
	Corpus    string `json:"corpus_sha256"`
	Tokens    string `json:"tokens_sha256"`
	Reference string `json:"reference_format"`
	Vocab     int    `json:"vocab"`
	Rows      int    `json:"rows"`
}

// External IDs support numerical experiments before a Go text tokenizer exists.
// They never enable the production text/HTTP entry points for that profile.
type qualityTokenInputs struct {
	Format       string  `json:"format"`
	Source       string  `json:"source"`
	Corpus       string  `json:"corpus_sha256"`
	TokenizerSHA string  `json:"tokenizer_sha256"`
	Version      string  `json:"tokenizers_version"`
	Documents    [][]int `json:"document_ids"`
	Prompts      [][]int `json:"prompt_ids"`
}

func (q *qualityTokenInputs) validate(source, corpus string, documents, prompts, steps, vocab int) error {
	if q.Format != "monolith-quality-tokens-v1" || q.Source != source || q.Corpus != corpus || len(q.Documents) != documents || len(q.Prompts) != prompts || len(q.TokenizerSHA) != 64 || q.Version == "" {
		return fmt.Errorf("external token input identity/count mismatch")
	}
	for group, rows := range [][][]int{q.Documents, q.Prompts} {
		for _, row := range rows {
			lower, upper := 4+steps, 128
			if group == 1 {
				lower, upper = 1, 112
			}
			if len(row) < lower || len(row) > upper {
				return fmt.Errorf("external token input length")
			}
			for _, id := range row {
				if id < 0 || id >= vocab {
					return fmt.Errorf("external token ID out of range")
				}
			}
		}
	}
	return nil
}

func TestHFQualityProbe(t *testing.T) {
	dir := os.Getenv("MONOLITH_QUALITY_MODEL")
	if dir == "" {
		t.Skip("manual offline real-model probe")
	}
	number := func(key string, def, limit int) int {
		s := os.Getenv(key)
		if s == "" {
			return def
		}
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 || n > limit {
			t.Fatal("invalid bounded probe option", key)
		}
		return n
	}
	steps, docs := number("MONOLITH_QUALITY_STEPS", 16, 32), number("MONOLITH_QUALITY_DOCUMENTS", 6, 6)
	format, refFormat := os.Getenv("MONOLITH_QUALITY_FORMAT"), os.Getenv("MONOLITH_QUALITY_REFERENCE_FORMAT")
	if !validHFWeightFormat(format) || (refFormat != "f32" && refFormat != "q8") {
		t.Fatal("invalid probe format")
	}
	source := os.Getenv("MONOLITH_QUALITY_SOURCE")
	if source == "" {
		t.Fatal("missing pinned source identity")
	}
	corpusBytes, err := os.ReadFile("validation/quality/corpus.json")
	if err != nil || len(corpusBytes) > 64<<10 {
		t.Fatal("corpus bounds", err)
	}
	var corpus qualityCorpus
	if err = json.Unmarshal(corpusBytes, &corpus); err != nil || corpus.Format != "monolith-quantization-corpus-v1" || len(corpus.Documents) < docs || len(corpus.Prompts) > 4 {
		t.Fatal("corpus schema", err)
	}
	corpusHash := sha256.Sum256(corpusBytes)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	start := time.Now()
	m, plan, err := loadHFModelFormat(ctx, dir, 128, int64(number("MONOLITH_QUALITY_WEIGHT_MIB", 8192, 16384)), format)
	if err != nil {
		t.Fatal(err)
	}
	var tok *hfTokenizer
	var external *qualityTokenInputs
	if path := os.Getenv("MONOLITH_QUALITY_TOKEN_INPUTS"); path != "" {
		external = new(qualityTokenInputs)
		if err := readStrictBoundedJSON(path, 1<<20, external); err != nil {
			t.Fatal(err)
		}
		if err := external.validate(source, hex.EncodeToString(corpusHash[:]), len(corpus.Documents), len(corpus.Prompts), steps, m.Config.Vocab); err != nil {
			t.Fatal(err)
		}
	} else {
		tok, err = loadHFTokenizer(dir, *m.hf)
		if err != nil {
			t.Fatal(err)
		}
	}
	loadSeconds := time.Since(start).Seconds()
	const prefix = 4
	ids := make([][]int, docs)
	idsJSON := [][]int{}
	caches := make([]*KVCache, docs)
	chunks := make([][]int, docs)
	for i := 0; i < docs; i++ {
		if external != nil {
			ids[i] = external.Documents[i]
		} else {
			ids[i], err = tok.encodePrompt(ctx, corpus.Documents[i].Text, 128)
		}
		if err != nil || len(ids[i]) < prefix+steps {
			t.Fatal("document too short/long", i, err)
		}
		ids[i] = ids[i][:prefix+steps]
		idsJSON = append(idsJSON, ids[i])
		caches[i] = m.NewCache()
		defer caches[i].Close()
		chunks[i] = ids[i][:prefix]
	}
	tokenBytes, _ := json.Marshal(idsJSON)
	tokenHash := sha256.Sum256(tokenBytes)
	header := qualityHeader{"monolith-quality-logits-v1", source, hex.EncodeToString(corpusHash[:]), hex.EncodeToString(tokenHash[:]), refFormat, m.Config.Vocab, docs * steps}
	referencePath := os.Getenv("MONOLITH_QUALITY_REFERENCE")
	writing := os.Getenv("MONOLITH_QUALITY_WRITE_REFERENCE") == "1"
	var ref *os.File
	if writing {
		if format != refFormat {
			t.Fatal("reference format mismatch")
		}
		ref, err = os.OpenFile(referencePath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	} else {
		ref, err = os.Open(referencePath)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	headerBytes, _ := json.Marshal(header)
	if writing {
		if err = binary.Write(ref, binary.LittleEndian, uint32(len(headerBytes))); err != nil {
			t.Fatal(err)
		}
		if _, err = ref.Write(headerBytes); err != nil {
			t.Fatal(err)
		}
	} else {
		var length uint32
		if err = binary.Read(ref, binary.LittleEndian, &length); err != nil || length > 64<<10 {
			t.Fatal("reference header", err)
		}
		raw := make([]byte, length)
		if _, err = io.ReadFull(ref, raw); err != nil {
			t.Fatal(err)
		}
		var got qualityHeader
		if err = json.Unmarshal(raw, &got); err != nil || !reflect.DeepEqual(got, header) {
			t.Fatal("reference does not describe these inputs", err)
		}
	}
	type rowResult struct {
		Document      string  `json:"document"`
		Position      int     `json:"position"`
		Target        int     `json:"target_id"`
		NLL           float64 `json:"nll"`
		KL            float64 `json:"kl_reference_candidate"`
		Top1          int     `json:"top1_id"`
		ReferenceTop1 int     `json:"reference_top1_id"`
	}
	rows := []rowResult{}
	var nllSum, klSum float64
	matches := 0
	e := newKernelExecutor(4)
	defer e.close()
	var w decodeWorkspace
	vectors := make([]byte, m.Config.Vocab*4)
	refRow := make([]float32, m.Config.Vocab)
	vectorHash := sha256.New()
	start = time.Now()
	for step := 0; step < steps; step++ {
		if step > 0 {
			for i := range chunks {
				chunks[i] = ids[i][prefix+step-1 : prefix+step]
			}
		}
		logits, err := m.prefillBorrowed(ctx, caches, chunks, e, &w)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < docs; i++ {
			row := logits[i*m.Config.Vocab : (i+1)*m.Config.Vocab]
			target := ids[i][prefix+step]
			_, nll, best, err := qualityDistribution(row, target)
			if err != nil {
				t.Fatal(err)
			}
			if writing {
				for j, v := range row {
					binary.LittleEndian.PutUint32(vectors[j*4:], math.Float32bits(v))
				}
				if _, err = ref.Write(vectors); err != nil {
					t.Fatal(err)
				}
				copy(refRow, row)
			} else {
				if _, err = io.ReadFull(ref, vectors); err != nil {
					t.Fatal(err)
				}
				for j := range refRow {
					refRow[j] = math.Float32frombits(binary.LittleEndian.Uint32(vectors[j*4:]))
				}
			}
			vectorHash.Write(vectors)
			_, _, rb, err := qualityDistribution(refRow, target)
			if err != nil {
				t.Fatal(err)
			}
			kl := qualityKL(refRow, row)
			if !finite(kl) {
				t.Fatal("nonfinite KL")
			}
			if rb == best {
				matches++
			}
			nllSum += nll
			klSum += kl
			rows = append(rows, rowResult{corpus.Documents[i].ID, prefix + step, target, nll, kl, best, rb})
		}
	}
	forwardSeconds := time.Since(start).Seconds()
	if writing {
		if err = ref.Sync(); err != nil {
			t.Fatal(err)
		}
	} else {
		var extra [1]byte
		if _, err = ref.Read(extra[:]); err != io.EOF {
			t.Fatal("trailing reference data", err)
		}
	}
	var completions []Completion
	for i, prompt := range corpus.Prompts {
		var p []int
		var err error
		if external != nil {
			p = external.Prompts[i]
		} else {
			p, err = tok.encodePrompt(ctx, prompt, 112)
		}
		if err != nil {
			t.Fatal(err)
		}
		r, err := inferHFIDs(ctx, m, p, 8, 16, false)
		if err != nil {
			t.Fatal(err)
		}
		if tok != nil {
			completions = append(completions, completion(tok, r.TokenIDs, len(p), r.FinishReason))
		} else {
			completions = append(completions, Completion{TokenIDs: r.TokenIDs, PromptTokens: len(p), CompletionTokens: len(r.TokenIDs), FinishReason: r.FinishReason})
		}
	}
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	rssBytes := int64(0)
	if b, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output(); err == nil {
		n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		if err == nil {
			rssBytes = n * 1024
		}
	}
	sourceBytes, err := os.ReadFile("monolith.go")
	if err != nil {
		t.Fatal(err)
	}
	sourceHash := sha256.Sum256(sourceBytes)
	result := map[string]any{"format": "monolith-quant-quality-v1", "source": source, "source_sha256": hex.EncodeToString(sourceHash[:]), "reference": header, "reference_vectors_sha256": hex.EncodeToString(vectorHash.Sum(nil)), "plan": plan, "scored_tokens": len(rows), "mean_nll": nllSum / float64(len(rows)), "perplexity": math.Exp(nllSum / float64(len(rows))), "mean_kl": klSum / float64(len(rows)), "top1_matches": matches, "rows": rows, "generation_prompts": corpus.Prompts, "completions": completions, "load_seconds": loadSeconds, "score_seconds": forwardSeconds, "post_probe_rss_bytes": rssBytes, "heap_alloc_bytes": memory.HeapAlloc, "rss_method": "one ps observation after scoring/generation; not peak", "threads": runtime.GOMAXPROCS(0), "kernel_workers": 4}
	if external != nil {
		result["external_token_inputs"] = external
		result["text_decoding"] = "not performed by Go; token-ID inference only"
	}
	b, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(os.Getenv("MONOLITH_QUALITY_OUTPUT"), append(b, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	t.Log(format, "NLL", result["mean_nll"], "KL", result["mean_kl"], "top1", matches, "/", len(rows))
}
