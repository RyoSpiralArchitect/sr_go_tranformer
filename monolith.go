// Monolith is a self-contained, CPU-only decoder Transformer. No cgo or ML runtime.
// Layout: configuration/RNG -> tokenizer -> kernels -> model/backprop -> cache ->
// optimizer -> checkpoint -> training -> sampling/HTTP -> CLI.
package main

import (
	"bufio"
	"bytes"
	"container/heap"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const (
	BOS                   = 256
	EOS                   = 257
	baseVocab             = 258
	maxParameters         = 250_000_000
	maxHFParameters       = 8_000_000_000
	maxHFVocab            = 262144
	maxActivationElements = 128 * 1024 * 1024
	checkpointMagic       = "MGLM0001"
)

// Config describes an untied-QKV, bias-free, pre-RMSNorm decoder. The output
// projection shares the token embedding. RoPE rotates adjacent coordinate pairs.
type Config struct {
	Vocab     int     `json:"vocab"`
	Dim       int     `json:"dim"`
	Hidden    int     `json:"hidden"`
	Layers    int     `json:"layers"`
	Heads     int     `json:"heads"`
	KVHeads   int     `json:"kv_heads"`
	Context   int     `json:"context"`
	RopeTheta float64 `json:"rope_theta"`
	NormEps   float64 `json:"norm_eps"`
}

func preset(name string) (Config, error) {
	c := Config{Vocab: baseVocab, RopeTheta: 10000, NormEps: 1e-5}
	switch name {
	case "demo":
		c.Dim, c.Hidden, c.Layers, c.Heads, c.KVHeads, c.Context = 32, 96, 2, 4, 2, 128
	case "tiny":
		c.Dim, c.Hidden, c.Layers, c.Heads, c.KVHeads, c.Context = 64, 192, 4, 4, 2, 256
	case "small":
		c.Dim, c.Hidden, c.Layers, c.Heads, c.KVHeads, c.Context = 256, 704, 6, 8, 2, 512
	case "base":
		c.Dim, c.Hidden, c.Layers, c.Heads, c.KVHeads, c.Context = 768, 2048, 12, 12, 4, 2048
	default:
		return c, fmt.Errorf("unknown preset %q (demo, tiny, small, base)", name)
	}
	return c, c.Validate()
}
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
func (c Config) Validate() error {
	return c.validateLimits(65536, maxParameters)
}
func (c Config) validateLimits(vocab int, parameters int64) error {
	if c.Vocab < baseVocab || c.Vocab > vocab || c.Dim < 2 || c.Dim > 8192 || c.Hidden < 1 || c.Hidden > 32768 || c.Layers < 1 || c.Layers > 128 || c.Heads < 1 || c.Heads > c.Dim || c.KVHeads < 1 || c.KVHeads > c.Heads || c.Context < 1 || c.Context > 32768 {
		return errors.New("invalid model dimensions (see README limits)")
	}
	if c.Dim%c.Heads != 0 || c.Heads%c.KVHeads != 0 || (c.Dim/c.Heads)%2 != 0 {
		return errors.New("dim must be divisible by heads with an even head dimension; heads must be divisible by kv_heads")
	}
	if !finite(c.RopeTheta) || c.RopeTheta <= 1 || !finite(c.NormEps) || c.NormEps < 1e-12 || c.NormEps > 1 {
		return errors.New("invalid RoPE theta or RMS epsilon")
	}
	if c.ParameterCount() > parameters {
		return fmt.Errorf("model exceeds %d parameter allocation limit", parameters)
	}
	if int64(c.Context)*int64(c.KVDim())*int64(c.Layers)*2 > maxActivationElements {
		return errors.New("KV cache exceeds allocation limit")
	}
	return nil
}
func (c Config) KVDim() int { return c.Dim / c.Heads * c.KVHeads }
func (c Config) ParameterCount() int64 {
	d, h, k, l := int64(c.Dim), int64(c.Hidden), int64(c.KVDim()), int64(c.Layers)
	return int64(c.Vocab)*d + l*(2*d+2*d*d+2*d*k+3*d*h) + d
}

// HF imports deliberately accept explicit, bounded inference profiles. Metadata is
// inspected on open file descriptors before any parameter storage is allocated.
const maxHFHeader = 4 << 20

type hfConfig struct {
	Architectures []string `json:"architectures"`
	ModelType     string   `json:"model_type"`
	Vocab         int      `json:"vocab_size"`
	Dim           int      `json:"hidden_size"`
	Hidden        int      `json:"intermediate_size"`
	Layers        int      `json:"num_hidden_layers"`
	Heads         int      `json:"num_attention_heads"`
	KVHeads       int      `json:"num_key_value_heads"`
	Context       int      `json:"max_position_embeddings"`
	Theta         float64  `json:"rope_theta"`
	Eps           float64  `json:"rms_norm_eps"`
	Act           string   `json:"hidden_act"`
	Tied          bool     `json:"tie_word_embeddings"`
	AttentionBias bool     `json:"attention_bias"`
	MLPBias       bool     `json:"mlp_bias"`
	Interleaved   bool     `json:"rope_interleaved"`
	Dropout       float64  `json:"attention_dropout"`
	TP            int      `json:"pretraining_tp"`
	BOS           int      `json:"bos_token_id"`
	EOS           int      `json:"eos_token_id"`
	Sliding       bool     `json:"use_sliding_window,omitempty"`
	LayerTypes    []string `json:"layer_types,omitempty"`
}

// encoding/json otherwise accepts duplicate keys and invalid UTF-8. Reject
// ambiguous metadata, excessive nesting and trailing documents at the boundary.
func uniqueJSON(data []byte, dst any) error {
	if !utf8.Valid(data) {
		return errors.New("invalid JSON UTF-8")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var walk func(int) error
	nodes := 0
	walk = func(depth int) error {
		nodes++
		if depth > 32 || nodes > 2_000_000 {
			return errors.New("JSON complexity limit")
		}
		t, err := d.Token()
		if err != nil {
			return err
		}
		switch t {
		case json.Delim('{'):
			seen := map[string]bool{}
			for d.More() {
				k, err := d.Token()
				if err != nil {
					return err
				}
				key, ok := k.(string)
				if !ok || seen[key] {
					return fmt.Errorf("duplicate/invalid JSON key %v", k)
				}
				seen[key] = true
				if err := walk(depth + 1); err != nil {
					return err
				}
			}
			_, err = d.Token()
		case json.Delim('['):
			for d.More() {
				if err := walk(depth + 1); err != nil {
					return err
				}
			}
			_, err = d.Token()
		}
		return err
	}
	if err := walk(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON data")
	}
	return json.Unmarshal(data, dst)
}

func readBoundedJSON(path string, limit int64, dst any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	if !st.Mode().IsRegular() || st.Size() > limit {
		return fmt.Errorf("%s: metadata size/type limit", filepath.Base(path))
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return err
	}
	if int64(len(b)) > limit {
		return errors.New("metadata size limit")
	}
	return uniqueJSON(b, dst)
}

func readHFConfig(dir string, contextLimit int) (hfConfig, Config, error) {
	var raw map[string]json.RawMessage
	var h hfConfig
	if err := readBoundedJSON(filepath.Join(dir, "config.json"), 1<<20, &raw); err != nil {
		return h, Config{}, err
	}
	if string(raw["model_type"]) == `"gpt2"` {
		return readGPT2Config(raw, contextLimit)
	}
	// Unknown computational fields must not silently change model semantics.
	allowed := strings.Fields("architectures model_type vocab_size hidden_size intermediate_size num_hidden_layers num_attention_heads num_key_value_heads max_position_embeddings rope_theta rms_norm_eps hidden_act tie_word_embeddings attention_bias mlp_bias rope_interleaved attention_dropout pretraining_tp bos_token_id eos_token_id pad_token_id initializer_range is_llama_config torch_dtype dtype transformers_version transformers.js_config use_cache _name_or_path _attn_implementation")
	keys := map[string]bool{}
	for _, k := range allowed {
		keys[k] = true
	}
	if string(raw["model_type"]) == `"qwen2"` {
		for _, k := range strings.Fields("use_sliding_window sliding_window max_window_layers layer_types") {
			keys[k] = true
		}
	}
	for k, v := range raw {
		if k == "rope_scaling" && string(v) == "null" {
			continue
		}
		if !keys[k] {
			return h, Config{}, fmt.Errorf("unsupported HF config field %q", k)
		}
	}
	b, _ := json.Marshal(raw)
	if err := json.Unmarshal(b, &h); err != nil {
		return h, Config{}, err
	}
	arch := map[string]string{"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM"}[h.ModelType]
	if arch == "" || len(h.Architectures) != 1 || h.Architectures[0] != arch || h.Act != "silu" || h.AttentionBias || h.MLPBias || h.Interleaved || h.Dropout != 0 || (h.TP != 0 && h.TP != 1) || h.Sliding {
		return h, Config{}, errors.New("unsupported HF profile: require Llama or Qwen2, SiLU, unscaled half-split RoPE and full causal attention")
	}
	if len(h.LayerTypes) != 0 && len(h.LayerTypes) != h.Layers {
		return h, Config{}, errors.New("HF layer_types count mismatch")
	}
	for _, kind := range h.LayerTypes {
		if kind != "full_attention" {
			return h, Config{}, errors.New("unsupported HF attention layer type")
		}
	}
	if raw["bos_token_id"] == nil || raw["eos_token_id"] == nil || string(raw["bos_token_id"]) == "null" || string(raw["eos_token_id"]) == "null" || h.BOS < 0 || h.BOS >= h.Vocab || h.EOS < 0 || h.EOS >= h.Vocab {
		return h, Config{}, errors.New("HF profile requires scalar BOS/EOS IDs within vocabulary")
	}
	c := Config{Vocab: h.Vocab, Dim: h.Dim, Hidden: h.Hidden, Layers: h.Layers, Heads: h.Heads, KVHeads: h.KVHeads, Context: h.Context, RopeTheta: h.Theta, NormEps: h.Eps}
	if contextLimit < 0 || contextLimit > h.Context {
		return h, c, errors.New("HF context limit must be within model context")
	}
	if contextLimit > 0 {
		c.Context = contextLimit
	} else {
		c.Context = min(h.Context, 2048)
	}
	if err := c.validateLimits(maxHFVocab, maxHFParameters); err != nil {
		return h, c, err
	}
	if hfParameterCount(c, h) > maxHFParameters {
		return h, c, fmt.Errorf("HF model exceeds %d parameters including biases and output head", maxHFParameters)
	}
	return h, c, nil
}

func hfParameterCount(c Config, h hfConfig) int64 {
	if h.ModelType == "gpt2" {
		d, hidden, layers := int64(c.Dim), int64(c.Hidden), int64(c.Layers)
		return (int64(c.Vocab)+int64(h.Context))*d + layers*(4*d*d+2*d*hidden+9*d+hidden) + 2*d
	}
	n := c.ParameterCount()
	if h.ModelType == "qwen2" {
		n += int64(c.Layers) * int64(c.Dim+2*c.KVDim())
	}
	if !h.Tied {
		n += int64(c.Vocab) * int64(c.Dim)
	}
	return n
}

type hfTensorSpec struct {
	Name  string
	Shape []int
}

func hfTensorSpecs(c Config) []hfTensorSpec {
	out := []hfTensorSpec{{"model.embed_tokens.weight", []int{c.Vocab, c.Dim}}}
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		out = append(out,
			hfTensorSpec{p + "input_layernorm.weight", []int{c.Dim}},
			hfTensorSpec{p + "self_attn.q_proj.weight", []int{c.Dim, c.Dim}},
			hfTensorSpec{p + "self_attn.k_proj.weight", []int{c.KVDim(), c.Dim}},
			hfTensorSpec{p + "self_attn.v_proj.weight", []int{c.KVDim(), c.Dim}},
			hfTensorSpec{p + "self_attn.o_proj.weight", []int{c.Dim, c.Dim}},
			hfTensorSpec{p + "post_attention_layernorm.weight", []int{c.Dim}},
			hfTensorSpec{p + "mlp.gate_proj.weight", []int{c.Hidden, c.Dim}},
			hfTensorSpec{p + "mlp.up_proj.weight", []int{c.Hidden, c.Dim}},
			hfTensorSpec{p + "mlp.down_proj.weight", []int{c.Dim, c.Hidden}})
	}
	return append(out, hfTensorSpec{"model.norm.weight", []int{c.Dim}})
}

// Extra inference parameters follow the unchanged native parameter ordering.
func hfProfileSpecs(c Config, h hfConfig) []hfTensorSpec {
	if h.ModelType == "gpt2" {
		return gpt2Specs(c, h)
	}
	out := hfTensorSpecs(c)
	if h.ModelType == "qwen2" {
		for i := 0; i < c.Layers; i++ {
			p := fmt.Sprintf("model.layers.%d.self_attn.", i)
			out = append(out, hfTensorSpec{p + "q_proj.bias", []int{c.Dim}},
				hfTensorSpec{p + "k_proj.bias", []int{c.KVDim()}}, hfTensorSpec{p + "v_proj.bias", []int{c.KVDim()}})
		}
	}
	if !h.Tied {
		out = append(out, hfTensorSpec{"lm_head.weight", []int{c.Vocab, c.Dim}})
	}
	return out
}

// GPT-2 has absolute positions, affine LayerNorm and a two-projection GELU MLP.
// Defaults here match GPT2Config; unsupported computational switches fail closed.
func readGPT2Config(raw map[string]json.RawMessage, limit int) (hfConfig, Config, error) {
	var h hfConfig
	var g struct {
		Dim     int     `json:"n_embd"`
		Hidden  int     `json:"n_inner"`
		Layers  int     `json:"n_layer"`
		Heads   int     `json:"n_head"`
		Context int     `json:"n_positions"`
		NCtx    int     `json:"n_ctx"`
		Eps     float64 `json:"layer_norm_epsilon"`
		Act     string  `json:"activation_function"`
		Scale   bool    `json:"scale_attn_weights"`
		Inverse bool    `json:"scale_attn_by_inverse_layer_idx"`
		Upcast  bool    `json:"reorder_and_upcast_attn"`
		Cross   bool    `json:"add_cross_attention"`
	}
	g.Eps, g.Act, g.Scale = 1e-5, "gelu_new", true
	h.Tied = true
	allowed := strings.Fields("architectures model_type vocab_size n_embd n_inner n_layer n_head n_positions n_ctx layer_norm_epsilon activation_function scale_attn_weights scale_attn_by_inverse_layer_idx reorder_and_upcast_attn add_cross_attention tie_word_embeddings bos_token_id eos_token_id pad_token_id attn_pdrop resid_pdrop embd_pdrop initializer_range torch_dtype dtype transformers_version use_cache _name_or_path _attn_implementation summary_type summary_use_proj summary_activation summary_proj_to_labels summary_first_dropout task_specific_params")
	keys := map[string]bool{}
	for _, k := range allowed {
		keys[k] = true
	}
	for k, v := range raw {
		if !keys[k] {
			return h, Config{}, fmt.Errorf("unsupported GPT-2 config field %q", k)
		}
		if string(v) == "null" && k != "n_inner" && k != "pad_token_id" && k != "summary_activation" {
			return h, Config{}, fmt.Errorf("null GPT-2 config field %q", k)
		}
	}
	b, _ := json.Marshal(raw)
	if err := json.Unmarshal(b, &h); err != nil {
		return h, Config{}, err
	}
	if err := json.Unmarshal(b, &g); err != nil {
		return h, Config{}, err
	}
	if len(h.Architectures) != 1 || h.Architectures[0] != "GPT2LMHeadModel" || !h.Tied || g.Act != "gelu_new" || !g.Scale || g.Inverse || g.Upcast || g.Cross || (g.NCtx != 0 && g.NCtx != g.Context) {
		return h, Config{}, errors.New("unsupported GPT-2 profile: require tied GPT2LMHeadModel, gelu_new and standard scaled causal self-attention")
	}
	if g.Hidden == 0 {
		g.Hidden = 4 * g.Dim
	}
	h.Dim, h.Hidden, h.Layers, h.Heads, h.KVHeads = g.Dim, g.Hidden, g.Layers, g.Heads, g.Heads
	h.Context, h.Eps, h.Act = g.Context, g.Eps, g.Act
	if raw["bos_token_id"] == nil || raw["eos_token_id"] == nil || h.BOS < 0 || h.BOS >= h.Vocab || h.EOS < 0 || h.EOS >= h.Vocab {
		return h, Config{}, errors.New("GPT-2 requires scalar BOS/EOS IDs within vocabulary")
	}
	c := Config{Vocab: h.Vocab, Dim: h.Dim, Hidden: h.Hidden, Layers: h.Layers, Heads: h.Heads, KVHeads: h.Heads, Context: h.Context, RopeTheta: 10000, NormEps: h.Eps}
	if limit < 0 || limit > h.Context || h.Context > 32768 {
		return h, c, errors.New("invalid GPT-2 context limit")
	}
	if limit > 0 {
		c.Context = limit
	} else {
		c.Context = min(h.Context, 2048)
	}
	// Reuse dimension/KV guards; the exact GPT-2 storage check below replaces
	// the native three-projection MLP parameter estimate.
	if err := c.validateLimits(maxHFVocab, math.MaxInt64); err != nil {
		return h, c, err
	}
	if hfParameterCount(c, h) > maxHFParameters {
		return h, c, errors.New("GPT-2 exceeds HF parameter limit")
	}
	return h, c, nil
}

func gpt2Specs(c Config, h hfConfig) []hfTensorSpec {
	d := c.Dim
	out := []hfTensorSpec{{"wte.weight", []int{c.Vocab, d}}, {"wpe.weight", []int{h.Context, d}},
		{"ln_f.weight", []int{d}}, {"ln_f.bias", []int{d}}}
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("h.%d.", i)
		out = append(out,
			hfTensorSpec{p + "ln_1.weight", []int{d}}, hfTensorSpec{p + "ln_1.bias", []int{d}},
			hfTensorSpec{p + "attn.c_attn.weight", []int{d, 3 * d}}, hfTensorSpec{p + "attn.c_attn.bias", []int{3 * d}},
			hfTensorSpec{p + "attn.c_proj.weight", []int{d, d}}, hfTensorSpec{p + "attn.c_proj.bias", []int{d}},
			hfTensorSpec{p + "ln_2.weight", []int{d}}, hfTensorSpec{p + "ln_2.bias", []int{d}},
			hfTensorSpec{p + "mlp.c_fc.weight", []int{d, c.Hidden}}, hfTensorSpec{p + "mlp.c_fc.bias", []int{c.Hidden}},
			hfTensorSpec{p + "mlp.c_proj.weight", []int{c.Hidden, d}}, hfTensorSpec{p + "mlp.c_proj.bias", []int{d}})
	}
	return out
}

type hfTensor struct {
	DType   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets []int64 `json:"data_offsets"`
	file    *os.File
	start   int64
	count   int
}
type HFPlan struct {
	Format                  string         `json:"format"`
	ModelType               string         `json:"model_type"`
	TiedEmbeddings          bool           `json:"tied_embeddings"`
	Config                  Config         `json:"config"`
	Parameters              int64          `json:"parameters"`
	WeightBytes             int64          `json:"weight_bytes"`
	WeightFormat            string         `json:"weight_format"`
	FP32WeightBytes         int64          `json:"fp32_weight_bytes"`
	QuantizedParameters     int64          `json:"quantized_parameters,omitempty"`
	Q8Parameters            int64          `json:"q8_parameters,omitempty"`
	Q4Parameters            int64          `json:"q4_parameters,omitempty"`
	ScaleBytes              int64          `json:"scale_bytes,omitempty"`
	QuantizationBlock       int            `json:"quantization_block,omitempty"`
	QuantizationBufferBytes int            `json:"quantization_buffer_bytes,omitempty"`
	RopeBytes               int64          `json:"rope_bytes"`
	KVBytesPerToken         int64          `json:"kv_bytes_per_token"`
	FullContextKVBytes      int64          `json:"full_context_kv_bytes"`
	ConversionBufferBytes   int            `json:"conversion_buffer_bytes"`
	SourceBytes             int64          `json:"source_bytes"`
	Shards                  int            `json:"shards"`
	DTypes                  map[string]int `json:"tensor_dtypes"`
}

func validHFWeightFormat(format string) bool {
	return format == "f32" || format == "q8" || format == "q4" || format == "q4-mixed"
}

// Logical native names and HF source names share the same explicit policy.
// Tied embeddings are allocated only once; untied output heads get Q8 too.
func hfMatrixFormat(format, name string) string {
	if format != "q4-mixed" {
		return format
	}
	switch name {
	case "token_embedding", "output", "model.embed_tokens.weight", "lm_head.weight", "wte.weight":
		return "q8"
	default:
		return "q4"
	}
}

type hfWeightLayout struct{ bytes, q8, q4, scales int64 }

func hfWeightStorage(c Config, h hfConfig, format string) (payload, quantized, scales int64, err error) {
	l, err := hfWeightLayoutFor(c, h, format)
	return l.bytes, l.q8 + l.q4, l.scales, err
}

func hfWeightLayoutFor(c Config, h hfConfig, format string) (layout hfWeightLayout, err error) {
	if !validHFWeightFormat(format) {
		return layout, errors.New("weight-format must be f32, q8, q4 or q4-mixed")
	}
	for _, s := range hfProfileSpecs(c, h) {
		count := int64(1)
		for _, d := range s.Shape {
			count *= int64(d)
		}
		if format != "f32" && len(s.Shape) == 2 && s.Name != "wpe.weight" {
			rows, cols := s.Shape[0], s.Shape[1]
			if h.ModelType == "gpt2" && s.Name != "wte.weight" {
				rows, cols = cols, rows // Conv1D transpose, including fused QKV.
			}
			layout.scales += int64(rows) * int64((cols+q8Block-1)/q8Block) * 4
			if hfMatrixFormat(format, s.Name) == "q4" {
				layout.q4 += count
				layout.bytes += int64(rows) * int64((cols+1)/2)
			} else {
				layout.q8 += count
				layout.bytes += count
			}
		} else {
			layout.bytes += count * 4
		}
	}
	layout.bytes += layout.scales
	return layout, nil
}

type hfArchive struct {
	Config  hfConfig
	Plan    HFPlan
	tensors map[string]hfTensor
	files   []*os.File
}

func (a *hfArchive) Close() {
	for _, f := range a.files {
		_ = f.Close()
	}
	a.files = nil
}
func hfWidth(dtype string) int {
	switch dtype {
	case "F32":
		return 4
	case "F16", "BF16":
		return 2
	}
	return 0
}

func openHFArchive(dir string, contextLimit int, weightBudget int64) (_ *hfArchive, err error) {
	return openHFArchiveFormat(dir, contextLimit, weightBudget, "f32")
}

func openHFArchiveFormat(dir string, contextLimit int, weightBudget int64, format string) (_ *hfArchive, err error) {
	h, c, err := readHFConfig(dir, contextLimit)
	if err != nil {
		return nil, err
	}
	parameters := hfParameterCount(c, h)
	layout, err := hfWeightLayoutFor(c, h, format)
	storage, quantized, scales := layout.bytes, layout.q8+layout.q4, layout.scales
	if err != nil {
		return nil, err
	}
	if weightBudget < storage {
		return nil, fmt.Errorf("HF %s weights require %d MiB, exceed weight budget %d MiB; raise -weight-mib", format, (storage+(1<<20)-1)>>20, weightBudget>>20)
	}
	a := &hfArchive{Config: h, tensors: map[string]hfTensor{}}
	defer func() {
		if err != nil {
			a.Close()
		}
	}()
	a.Plan = HFPlan{Format: "monolith-hf-plan-v1", ModelType: h.ModelType, TiedEmbeddings: h.Tied, Config: c, Parameters: parameters, WeightBytes: parameters * 4, RopeBytes: int64(c.Context) * (int64(c.Dim) / int64(c.Heads)) * 4, KVBytesPerToken: int64(c.Layers*c.KVDim()) * 8, ConversionBufferBytes: 64 << 10, DTypes: map[string]int{}}
	a.Plan.WeightBytes, a.Plan.WeightFormat, a.Plan.FP32WeightBytes = storage, format, parameters*4
	a.Plan.QuantizedParameters, a.Plan.ScaleBytes = quantized, scales
	a.Plan.Q8Parameters, a.Plan.Q4Parameters = layout.q8, layout.q4
	if format != "f32" {
		a.Plan.QuantizationBlock = q8Block
		a.Plan.QuantizationBufferBytes = q8Block * 4
	}
	a.Plan.FullContextKVBytes = int64(c.Context) * a.Plan.KVBytesPerToken
	gpt2 := h.ModelType == "gpt2"
	if gpt2 {
		a.Plan.RopeBytes = 0
	}
	specs := hfProfileSpecs(c, h)
	expected := map[string][]int{}
	for _, s := range specs {
		expected[s.Name] = s.Shape
	}
	optional := map[string][]int{}
	canonicalName := func(name string) string {
		if gpt2 {
			return strings.TrimPrefix(name, "transformer.")
		}
		return name
	}
	// Older GPT-2 exports persist the causal mask. It is not a parameter and
	// never becomes a quadratic runtime allocation; validate it through chunks.
	auxBytes := int64(0)
	if gpt2 && int64(h.Context)*int64(h.Context)*int64(h.Layers) <= maxActivationElements {
		auxBytes = int64(h.Context) * int64(h.Context) * int64(h.Layers) * 4
		for i := 0; i < h.Layers; i++ {
			optional[fmt.Sprintf("h.%d.attn.bias", i)] = []int{1, 1, h.Context, h.Context}
		}
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	shards := map[string]bool{}
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, e := os.Stat(indexPath); e == nil {
		if err = readBoundedJSON(indexPath, maxHFHeader, &index); err != nil {
			return nil, err
		}
		if len(index.WeightMap) < len(expected) || len(index.WeightMap) > len(expected)+len(optional) {
			return nil, errors.New("HF shard index tensor count mismatch")
		}
		normalized := map[string]string{}
		for name, file := range index.WeightMap {
			key := canonicalName(name)
			if (expected[key] == nil && optional[key] == nil) || normalized[key] != "" || file == "" || filepath.Base(file) != file || strings.ContainsAny(file, `/\`) || !strings.HasSuffix(file, ".safetensors") {
				return nil, errors.New("unsupported HF shard index name/path")
			}
			normalized[key] = file
			shards[file] = true
		}
		index.WeightMap = normalized
	} else if errors.Is(e, os.ErrNotExist) {
		shards["model.safetensors"] = true
	} else {
		return nil, e
	}
	if len(shards) == 0 || len(shards) > 128 {
		return nil, errors.New("HF shard count limit")
	}
	names := make([]string, 0, len(shards))
	for name := range shards {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		f, e := os.Open(filepath.Join(dir, name))
		if e != nil {
			return nil, e
		}
		a.files = append(a.files, f)
		st, e := f.Stat()
		if e != nil {
			return nil, e
		}
		if !st.Mode().IsRegular() || st.Size() < 10 || st.Size() > parameters*4+auxBytes+maxHFHeader+8 {
			return nil, errors.New("Safetensors file size/type limit")
		}
		var prefix [8]byte
		if _, e = io.ReadFull(f, prefix[:]); e != nil {
			return nil, e
		}
		n := binary.LittleEndian.Uint64(prefix[:])
		if n < 2 || n > maxHFHeader || n > uint64(st.Size()-8) {
			return nil, errors.New("Safetensors header length limit")
		}
		header := make([]byte, int(n))
		if _, e = io.ReadFull(f, header); e != nil {
			return nil, e
		}
		if header[0] != '{' {
			return nil, errors.New("Safetensors header must start with object")
		}
		var raw map[string]json.RawMessage
		if e = uniqueJSON(header, &raw); e != nil {
			return nil, e
		}
		var ordered []hfTensor
		for key, value := range raw {
			if key == "__metadata__" {
				var metadata map[string]string
				if e = json.Unmarshal(value, &metadata); e != nil {
					return nil, e
				}
				continue
			}
			key = canonicalName(key)
			shape, ok := expected[key]
			if !ok {
				shape, ok = optional[key]
			}
			if !ok || a.tensors[key].file != nil || (index.WeightMap != nil && index.WeightMap[key] != name) {
				return nil, fmt.Errorf("unexpected or duplicate HF tensor %q", key)
			}
			var t hfTensor
			d := json.NewDecoder(bytes.NewReader(value))
			d.DisallowUnknownFields()
			if e = d.Decode(&t); e != nil {
				return nil, e
			}
			if len(t.Shape) != len(shape) {
				return nil, fmt.Errorf("HF tensor shape mismatch: %s", key)
			}
			t.count = 1
			for i, v := range shape {
				if t.Shape[i] != v {
					return nil, fmt.Errorf("HF tensor shape mismatch: %s", key)
				}
				t.count *= v
			}
			w := hfWidth(t.DType)
			if optional[key] != nil && t.DType != "F32" {
				return nil, errors.New("GPT-2 legacy causal masks must be F32")
			}
			if w == 0 || len(t.Offsets) != 2 || t.Offsets[0] < 0 || t.Offsets[1] < t.Offsets[0] || t.Offsets[1] > st.Size()-8-int64(n) || t.Offsets[1]-t.Offsets[0] != int64(t.count)*int64(w) {
				return nil, fmt.Errorf("invalid HF tensor dtype/extent: %s", key)
			}
			t.file, t.start = f, 8+int64(n)+t.Offsets[0]
			a.tensors[key] = t
			a.Plan.DTypes[t.DType]++
			ordered = append(ordered, t)
		}
		sort.Slice(ordered, func(i, j int) bool { return ordered[i].Offsets[0] < ordered[j].Offsets[0] })
		end := int64(0)
		for _, t := range ordered {
			if t.Offsets[0] != end {
				return nil, errors.New("Safetensors overlapping or unindexed bytes")
			}
			end = t.Offsets[1]
		}
		if end != st.Size()-8-int64(n) {
			return nil, errors.New("Safetensors trailing/unindexed bytes")
		}
		a.Plan.SourceBytes += st.Size()
	}
	for key := range expected {
		if a.tensors[key].file == nil {
			return nil, fmt.Errorf("missing HF tensor %q", key)
		}
	}
	a.Plan.Shards = len(a.files)
	return a, nil
}

func halfFloat(v uint16) float32 {
	sign := uint32(v&0x8000) << 16
	exp, frac := (v>>10)&31, uint32(v&1023)
	if exp == 0 {
		// Includes signed zero and exact half subnormals.
		x := float32(math.Ldexp(float64(frac), -24))
		return math.Float32frombits(math.Float32bits(x) | sign)
	}
	if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	}
	return math.Float32frombits(sign | uint32(exp+112)<<23 | frac<<13)
}
func (a *hfArchive) readTensor(ctx context.Context, name string, dst []float32) error {
	return a.readTensorValues(ctx, name, len(dst), func(i int, v float32) error { dst[i] = v; return nil })
}

// Row-major tensors need only one small block of converted values. The full
// FP32 tensor/model is never allocated when quantized storage is selected.
func (a *hfArchive) readParam(ctx context.Context, name string, p *Param) error {
	if !p.quantized() {
		return a.readTensor(ctx, name, p.Data)
	}
	var block [q8Block]float32
	groups := (p.Cols + q8Block - 1) / q8Block
	return a.readTensorValues(ctx, name, p.Rows*p.Cols, func(i int, v float32) error {
		row, col := i/p.Cols, i%p.Cols
		block[col%q8Block] = v
		if col%q8Block == q8Block-1 || col == p.Cols-1 {
			count := col%q8Block + 1
			var amax float32
			for _, x := range block[:count] {
				amax = max(amax, float32(math.Abs(float64(x))))
			}
			scale := p.quantScale(amax)
			p.quantScales()[row*groups+col/q8Block] = scale
			for j, x := range block[:count] {
				p.putQuantized(row, col-count+1+j, x, scale)
			}
		}
		return nil
	})
}

// Assignment is streamed so Conv1D transposition and fused-QKV splitting do
// not allocate a second tensor or a second model.
func (a *hfArchive) readTensorValues(ctx context.Context, name string, count int, put func(int, float32) error) error {
	t, ok := a.tensors[name]
	if !ok || count != t.count {
		return errors.New("HF tensor destination mismatch")
	}
	buf := make([]byte, a.Plan.ConversionBufferBytes)
	w := hfWidth(t.DType)
	for off := 0; off < count; {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := min(len(buf)/w, count-off)
		b := buf[:n*w]
		if _, err := t.file.ReadAt(b, t.start+int64(off)*int64(w)); err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			var v float32
			switch t.DType {
			case "F32":
				v = math.Float32frombits(binary.LittleEndian.Uint32(b[i*w:]))
			case "BF16":
				v = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[i*w:])) << 16)
			case "F16":
				v = halfFloat(binary.LittleEndian.Uint16(b[i*w:]))
			}
			if !finite(float64(v)) {
				return fmt.Errorf("non-finite HF weight in %s", name)
			}
			if err := put(off+i, v); err != nil {
				return err
			}
		}
		off += n
	}
	return ctx.Err()
}

func hfFlags(f *flag.FlagSet) (*string, *int, *int64, *string) {
	dir := f.String("hf-model", "", "local Hugging Face model directory (supported Llama/Qwen2/GPT-2 profiles)")
	limit := f.Int("context", 0, "HF context cap; 0 uses min(model maximum, 2048)")
	budget := f.Int64("weight-mib", 2048, "maximum resident HF weight payload in MiB (not RSS)")
	format := f.String("weight-format", "f32", "resident HF weights: f32, q8, q4 or q4-mixed (Q8 embeddings/head, Q4 projections)")
	return dir, limit, budget, format
}
func openHFFlags(dir string, limit int, mib int64) (*hfArchive, error) {
	return openHFFlagsFormat(dir, limit, mib, "f32")
}
func openHFFlagsFormat(dir string, limit int, mib int64, format string) (*hfArchive, error) {
	if dir == "" || mib < 1 || mib > 1<<20 {
		return nil, errors.New("provide -hf-model and a valid -weight-mib (1..1048576)")
	}
	return openHFArchiveFormat(dir, limit, mib<<20, format)
}
func runInspectHF(args []string, validateOnly bool) error {
	f := flags("inspect-hf")
	dir, limit, budget, format := hfFlags(f)
	if err := parseInferenceAction(f, args, validateOnly); err != nil {
		return err
	}
	if validateOnly {
		return nil
	}
	a, err := openHFFlagsFormat(*dir, *limit, *budget, *format)
	if err != nil {
		return err
	}
	defer a.Close()
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(a.Plan)
}

type hfCheck struct {
	Format              string `json:"format"`
	Weights             HFPlan `json:"weights"`
	TokenizerProfile    string `json:"tokenizer_profile"`
	ChatSupported       bool   `json:"chat_supported"`
	ModelContext        int    `json:"model_context"`
	WeightBudgetBytes   int64  `json:"weight_budget_bytes"`
	KVPageAlignedBytes  int64  `json:"kv_page_aligned_bytes_per_session"`
	WeightValuesChecked bool   `json:"weight_values_checked"`
}

func checkHF(dir string, limit int, budget int64, requireChat bool) (hfCheck, error) {
	return checkHFFormat(dir, limit, budget, requireChat, "f32")
}
func checkHFFormat(dir string, limit int, budget int64, requireChat bool, format string) (hfCheck, error) {
	a, err := openHFFlagsFormat(dir, limit, budget, format)
	if err != nil {
		return hfCheck{}, err
	}
	defer a.Close()
	tok, err := loadHFTokenizer(dir, a.Config)
	if err != nil {
		return hfCheck{}, err
	}
	if requireChat && !tok.chat {
		return hfCheck{}, errors.New("unsupported chat template; plain text inference remains available")
	}
	profile := "smollm-bytelevel"
	if tok.qwen {
		profile = "qwen2-bytelevel-nfc"
	}
	if tok.gpt2 {
		profile = "gpt2-bytelevel"
	}
	return hfCheck{Format: "monolith-hf-check-v1", Weights: a.Plan, TokenizerProfile: profile,
		ChatSupported: tok.chat, ModelContext: a.Config.Context, WeightBudgetBytes: budget << 20,
		KVPageAlignedBytes: int64((a.Plan.Config.Context+pageTokens-1)/pageTokens*pageTokens) * a.Plan.KVBytesPerToken}, nil
}

func runCheckHF(args []string, validateOnly bool) error {
	f := flags("check-hf")
	dir, limit, budget, format := hfFlags(f)
	asJSON := f.Bool("json", false, "emit machine-readable preflight report")
	chat := f.Bool("chat", false, "require a supported chat template")
	if err := parseInferenceAction(f, args, validateOnly); err != nil {
		return err
	}
	if validateOnly {
		return nil
	}
	report, err := checkHFFormat(*dir, *limit, *budget, *chat, *format)
	if err != nil {
		return err
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}
	fmt.Printf("Compatible metadata: %s / %s\n", report.Weights.ModelType, report.TokenizerProfile)
	fmt.Printf("Context: %d (model maximum %d); chat supported: %t\n", report.Weights.Config.Context, report.ModelContext, report.ChatSupported)
	fmt.Printf("%s weights: %.1f MiB / %d MiB budget; paged KV per full session: %.1f MiB\n", report.Weights.WeightFormat, float64(report.Weights.WeightBytes)/(1<<20), *budget, float64(report.KVPageAlignedBytes)/(1<<20))
	fmt.Println("Weights are not loaded; payload values are checked at load time. Scratch, tokenizer and Go runtime memory are additional.")
	return nil
}

// An inherited stdin descriptor may block even after Close. One input pump per
// CLI invocation lets cancellation return through normal engine cleanup. Its
// copy buffer is bounded; a blocked OS read can live only until process exit.
// Closing the reader releases any blocked pipe write without waiting on stdin.
func cliInput(ctx context.Context, in io.Reader) (io.Reader, func()) {
	r, w := io.Pipe()
	stop := context.AfterFunc(ctx, func() { _ = w.CloseWithError(ctx.Err()) })
	go func() {
		_, err := io.CopyBuffer(w, struct{ io.Reader }{in}, make([]byte, 4096))
		_ = w.CloseWithError(err)
	}()
	return r, func() {
		stop()
		_ = r.Close()
	}
}

type cliWriteResult struct {
	n   int
	err error
}
type cliWrite struct {
	data []byte
	done chan cliWriteResult
}
type cliWriter struct {
	ctx  context.Context
	jobs chan cliWrite
}

// A single output pump keeps cancellation independent of an inherited stdout
// or stderr descriptor. There is no queued output: each chunk is acknowledged
// before the next. An uninterruptible OS write can retain only one 64 KiB copy
// until the receiver resumes or the process exits; cleanup never waits on it.
func cliOutput(ctx context.Context, out io.Writer) (io.Writer, func()) {
	ctx, cancel := context.WithCancel(ctx)
	w := &cliWriter{ctx: ctx, jobs: make(chan cliWrite)}
	go func() {
		var writeErr error
		for {
			select {
			case <-ctx.Done():
				return
			case job := <-w.jobs:
				n := 0
				if err := ctx.Err(); err != nil {
					writeErr = err
				}
				if writeErr == nil {
					n, writeErr = out.Write(job.data)
					if n < 0 || n > len(job.data) {
						n, writeErr = 0, io.ErrShortWrite
					} else if n != len(job.data) && writeErr == nil {
						writeErr = io.ErrShortWrite
					}
				}
				job.done <- cliWriteResult{n, writeErr}
			}
		}
	}()
	return w, cancel
}
func (w *cliWriter) Write(p []byte) (int, error) {
	written := 0
	for len(p) > 0 {
		if err := w.ctx.Err(); err != nil {
			return written, err
		}
		n := min(len(p), 64<<10)
		// The caller may reuse p as soon as cancellation returns, even while
		// the underlying writer is still blocked. Give the pump its own copy.
		job := cliWrite{data: append([]byte(nil), p[:n]...), done: make(chan cliWriteResult, 1)}
		select {
		case <-w.ctx.Done():
			return written, w.ctx.Err()
		case w.jobs <- job:
		}
		select {
		case <-w.ctx.Done():
			return written, w.ctx.Err()
		case result := <-job.done:
			written += result.n
			if result.err != nil {
				return written, result.err
			}
		}
		p = p[n:]
	}
	return written, nil
}

// Prompt files and stdin share the same byte bound. Only explicitly selected
// stdin is consumed, and JSON parsing is performed by the messages caller.
func readCLIInput(ctx context.Context, path string, stdin io.Reader) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r := stdin
	if path == "-" {
		var closeInput func()
		r, closeInput = cliInput(ctx, stdin)
		defer closeInput()
	} else {
		st, err := os.Stat(path)
		if err != nil {
			return nil, err
		}
		if !st.Mode().IsRegular() || st.Size() > 1<<20 {
			return nil, errors.New("CLI input must be a regular file of at most 1 MiB, or - for stdin")
		}
		// On Unix a path can be replaced with a FIFO between Stat and Open.
		// Nonblocking open lets the descriptor check reject that race too.
		f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK|syscall.O_NOCTTY, 0)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		st, err = f.Stat()
		if err != nil {
			return nil, err
		}
		if !st.Mode().IsRegular() || st.Size() > 1<<20 {
			return nil, errors.New("CLI input must be a regular file of at most 1 MiB, or - for stdin")
		}
		r = f
	}
	b, err := io.ReadAll(io.LimitReader(r, (1<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > 1<<20 {
		return nil, errors.New("CLI input exceeds 1 MiB")
	}
	return b, nil
}

type HFIDResult struct {
	PromptTokens int         `json:"prompt_tokens"`
	TokenIDs     []int       `json:"token_ids"`
	FinishReason string      `json:"finish_reason"`
	Logits       [][]float32 `json:"logits,omitempty"`
}

// inferHFIDs is a bounded numerical probe. IDs are already framed by the caller;
// no token (including the native BOS ID 256) is implicitly inserted or masked.
func inferHFIDs(ctx context.Context, m *Model, ids []int, count, chunk int, logits bool) (HFIDResult, error) {
	out := HFIDResult{PromptTokens: len(ids), TokenIDs: []int{}, FinishReason: "length"}
	if m.hf == nil || len(ids) == 0 || count < 1 || count > 256 || chunk < 1 || chunk > 1024 || len(ids) > m.Config.Context-count {
		return out, errors.New("invalid HF probe: nonempty IDs, 1..256 outputs, 1..1024 chunk, total within context")
	}
	for _, id := range ids {
		if id < 0 || id >= m.Config.Vocab {
			return out, errors.New("HF token ID outside vocabulary")
		}
	}
	cache := m.NewCache()
	defer cache.Close()
	var next []float32
	var err error
	for lo := 0; lo < len(ids); lo += chunk {
		next, err = m.PrefillBatch(ctx, []*KVCache{cache}, [][]int{ids[lo:min(lo+chunk, len(ids))]})
		if err != nil {
			return out, err
		}
	}
	for i := 0; i < count; i++ {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		best := 0
		for j, v := range next {
			if !finite(float64(v)) {
				return out, errors.New("non-finite HF logits")
			}
			if v > next[best] {
				best = j
			}
		}
		if logits {
			out.Logits = append(out.Logits, next)
		}
		out.TokenIDs = append(out.TokenIDs, best)
		if best == m.hf.EOS {
			out.FinishReason = "eos"
			break
		}
		if i+1 < count {
			next, err = m.DecodeBatch(ctx, []*KVCache{cache}, []int{best})
			if err != nil {
				return out, err
			}
		}
	}
	return out, nil
}
func runInferHF(ctx context.Context, args []string, validateOnly bool) error {
	f := flags("infer-hf")
	dir, limit, budget, format := hfFlags(f)
	input := f.String("ids", "", "comma-separated token IDs, already framed")
	count := f.Int("max-tokens", 8, "greedy outputs (1..256)")
	chunk := f.Int("prefill-chunk", 16, "prompt tokens per call (1..1024)")
	logits := f.Bool("logits", false, "include full FP32 logits for numerical comparison")
	threads := threadFlag(f)
	if err := parseInferenceAction(f, args, validateOnly); err != nil {
		return err
	}
	if len(*input) > 1<<20 {
		return errors.New("ID input exceeds 1 MiB")
	}
	var ids []int
	for _, s := range strings.Split(*input, ",") {
		id, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return fmt.Errorf("invalid token ID: %w", err)
		}
		if id < 0 {
			return errors.New("token IDs must be nonnegative")
		}
		ids = append(ids, id)
	}
	if *count < 1 || *count > 256 || *chunk < 1 || *chunk > 1024 {
		return errors.New("HF probe requires 1..256 outputs and 1..1024 prefill chunk")
	}
	if validateOnly {
		return nil
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	m, _, err := loadHFModelFormat(ctx, *dir, *limit, *budget, *format)
	if err != nil {
		return err
	}
	out, err := inferHFIDs(ctx, m, ids, *count, *chunk, *logits)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(out)
}

func flagSet(f *flag.FlagSet, name string) bool {
	set := false
	f.Visit(func(v *flag.Flag) {
		if v.Name == name {
			set = true
		}
	})
	return set
}
func loadInference(ctx context.Context, f *flag.FlagSet, model, dir string, limit int, budget int64, format string) (*Model, inferenceTokenizer, error) {
	if dir != "" {
		if flagSet(f, "model") {
			return nil, nil, errors.New("-model and -hf-model are mutually exclusive")
		}
		h, _, err := readHFConfig(dir, limit)
		if err != nil {
			return nil, nil, err
		}
		tok, err := loadHFTokenizer(dir, h)
		if err != nil {
			return nil, nil, err
		}
		m, _, err := loadHFModelFormat(ctx, dir, limit, budget, format)
		if err != nil {
			return nil, nil, err
		}
		if err := validateInferenceTokenizer(m, tok); err != nil {
			return nil, nil, err
		}
		return m, tok, nil
	}
	if flagSet(f, "context") || flagSet(f, "weight-mib") || flagSet(f, "weight-format") {
		return nil, nil, errors.New("-context/-weight-mib/-weight-format require -hf-model")
	}
	m, tok, _, err := LoadCheckpoint(model)
	return m, tok, err
}
func chatFlags(f *flag.FlagSet) (*bool, *string) {
	return f.Bool("chat", false, "format -prompt as a user message with the supported HF chat template"), f.String("messages", "", "JSON array of chat messages; mutually exclusive with -prompt/-chat")
}
func inferencePrompt(ctx context.Context, f *flag.FlagSet, tok inferenceTokenizer, prompt string, chat bool, messagesPath string) (string, error) {
	if source := f.Lookup("prompt-file"); source != nil && source.Value.String() != "" {
		if flagSet(f, "prompt") || messagesPath != "" {
			return "", errors.New("-prompt-file is mutually exclusive with -prompt/-messages")
		}
		b, err := readCLIInput(ctx, source.Value.String(), os.Stdin)
		if err != nil {
			return "", err
		}
		prompt = string(b)
	}
	if !chat && messagesPath == "" {
		return prompt, nil
	}
	t, ok := tok.(*hfTokenizer)
	if !ok {
		return "", errors.New("chat formatting requires a compatible HF tokenizer")
	}
	messages := []ChatMessage{{Role: "user", Content: prompt}}
	if messagesPath != "" {
		if chat || flagSet(f, "prompt") {
			return "", errors.New("-messages is mutually exclusive with -prompt/-chat")
		}
		b, err := readCLIInput(ctx, messagesPath, os.Stdin)
		if err != nil {
			return "", err
		}
		if err := uniqueJSON(b, &messages); err != nil {
			return "", err
		}
		d := json.NewDecoder(bytes.NewReader(b))
		d.DisallowUnknownFields()
		if err := d.Decode(&messages); err != nil {
			return "", err
		}
	}
	return t.formatChat(messages)
}
func runTokenizeHF(ctx context.Context, args []string, validateOnly bool) error {
	f := flags("tokenize-hf")
	dir := f.String("hf-model", "", "local HF model/tokenizer directory")
	prompt := f.String("prompt", "", "text to encode")
	f.String("prompt-file", "", "UTF-8 prompt file; - reads stdin (at most 1 MiB)")
	limit := f.Int("context", 0, "token limit; 0 uses min(model maximum, 2048)")
	chat, messages := chatFlags(f)
	if err := parseInferenceAction(f, args, validateOnly); err != nil {
		return err
	}
	if validateOnly {
		return nil
	}
	if *dir == "" {
		return errors.New("provide -hf-model")
	}
	h, c, err := readHFConfig(*dir, *limit)
	if err != nil {
		return err
	}
	t, err := loadHFTokenizer(*dir, h)
	if err != nil {
		return err
	}
	text, err := inferencePrompt(ctx, f, t, *prompt, *chat, *messages)
	if err != nil {
		return err
	}
	ids, err := t.encodePrompt(ctx, text, c.Context)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"prompt": text, "token_ids": ids, "text": t.Decode(ids)})
}

// SplitMix64 has one serializable state. No hidden normal-distribution cache.
type RNG struct {
	State uint64 `json:"state"`
}

func (r *RNG) Uint64() uint64 {
	r.State += 0x9e3779b97f4a7c15
	z := r.State
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}
func (r *RNG) Float64() float64 { return float64(r.Uint64()>>11) * (1.0 / (1 << 53)) }
func (r *RNG) Intn(n int) int {
	if n <= 0 {
		panic("Intn: nonpositive bound")
	}
	bound := uint64(n)
	threshold := -bound % bound
	for {
		x := r.Uint64()
		if x >= threshold {
			return int(x % bound)
		}
	}
}
func (r *RNG) Normal() float64 {
	return math.Sqrt(-2*math.Log(1-r.Float64())) * math.Cos(2*math.Pi*r.Float64())
}

// Tokenizer starts with all 256 bytes plus BOS/EOS, then ordered BPE merges.
// It is reversible for every byte sequence, with no Unicode normalization.
type Pair struct {
	A int `json:"a"`
	B int `json:"b"`
}
type Tokenizer struct {
	Merges []Pair `json:"merges"`
}

// Training/checkpoint tokenizers remain concrete. Only inference depends on
// this read-only interface, so native serialization and framing stay unchanged.
type inferenceTokenizer interface {
	Vocab() int
	Validate() error
	Pieces() [][]byte
	Decode([]int) string
	encodePrompt(context.Context, string, int) ([]int, error)
	eosID() int
	blockedID() int
}

func (t Tokenizer) eosID() int     { return EOS }
func (t Tokenizer) blockedID() int { return BOS }
func (t Tokenizer) encodePrompt(ctx context.Context, s string, limit int) ([]int, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ids := t.Encode(s, true, false)
	if len(ids) > limit {
		return nil, errors.New("prompt plus generation exceeds context")
	}
	return ids, ctx.Err()
}

type hfAddedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
}
type hfMerge struct{ rank, id int }
type hfTokenizer struct {
	pieces   [][]byte
	bytes    [256]int
	merges   map[Pair]hfMerge
	specials []hfAddedToken
	bos, eos int
	chat     bool
	qwen     bool
	gpt2     bool
}

func (t *hfTokenizer) Vocab() int { return len(t.pieces) }
func (t *hfTokenizer) Validate() error {
	if t == nil || t.Vocab() < 258 || t.eos < 0 || t.eos >= t.Vocab() || t.bos < 0 || t.bos >= t.Vocab() {
		return errors.New("invalid HF tokenizer")
	}
	return nil // All tables are validated once, before this immutable object escapes.
}
func (t *hfTokenizer) Pieces() [][]byte { return t.pieces }
func (t *hfTokenizer) eosID() int       { return t.eos }
func (t *hfTokenizer) blockedID() int   { return -1 }
func (t *hfTokenizer) Decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		if id >= 0 && id < len(t.pieces) {
			b = append(b, t.pieces[id]...)
		}
	}
	d := incrementalText{hf: true}
	return d.Push(b, true) // Special pieces are empty: generation skips special tokens.
}

const smolChatTemplate = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Exact Qwen2.5 template identity. Only its system/user/assistant, no-tools
// path is implemented; arbitrary Jinja and tool-call input are never executed.
const qwenChatTemplateSHA = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
const qwenSplitPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

func (t *hfTokenizer) formatChat(messages []ChatMessage) (string, error) {
	if !t.chat {
		return "", errors.New("unsupported chat template")
	}
	if len(messages) == 0 || len(messages) > 128 {
		return "", errors.New("chat requires 1..128 messages")
	}
	var b strings.Builder
	if messages[0].Role != "system" {
		system := "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
		if t.qwen {
			system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
		}
		b.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
	}
	for _, m := range messages {
		if m.Role != "system" && m.Role != "user" && m.Role != "assistant" {
			return "", errors.New("chat role must be system, user or assistant")
		}
		if !utf8.ValidString(m.Content) || b.Len()+len(m.Content)+len(m.Role)+64 > 1<<20 {
			return "", errors.New("chat exceeds UTF-8/1 MiB limit")
		}
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteByte('\n')
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String(), nil
}
func hfByteAlphabet() ([256]rune, map[rune]byte) {
	var forward [256]rune
	reverse := map[rune]byte{}
	extra := rune(256)
	for i := 0; i < 256; i++ {
		r := rune(i)
		if !(i >= 33 && i <= 126 || i >= 161 && i <= 172 || i >= 174) {
			r = extra
			extra++
		}
		forward[i] = r
		reverse[r] = byte(i)
	}
	return forward, reverse
}
func readStrictBoundedJSON(path string, limit int64, v any) error {
	var raw json.RawMessage
	if err := readBoundedJSON(path, limit, &raw); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	return d.Decode(v)
}
func loadHFTokenizer(dir string, h hfConfig) (*hfTokenizer, error) {
	var raw struct {
		Version       string          `json:"version"`
		Truncation    json.RawMessage `json:"truncation"`
		Padding       json.RawMessage `json:"padding"`
		Added         []hfAddedToken  `json:"added_tokens"`
		Normalizer    json.RawMessage `json:"normalizer"`
		PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		Decoder       json.RawMessage `json:"decoder"`
		Model         struct {
			Type     string            `json:"type"`
			Dropout  *float64          `json:"dropout"`
			Unk      *string           `json:"unk_token"`
			Prefix   *string           `json:"continuing_subword_prefix"`
			Suffix   *string           `json:"end_of_word_suffix"`
			Fuse     bool              `json:"fuse_unk"`
			Fallback bool              `json:"byte_fallback"`
			Ignore   bool              `json:"ignore_merges"`
			Vocab    map[string]int    `json:"vocab"`
			Merges   []json.RawMessage `json:"merges"`
		} `json:"model"`
	}
	if err := readStrictBoundedJSON(filepath.Join(dir, "tokenizer.json"), 32<<20, &raw); err != nil {
		return nil, err
	}
	null := func(b json.RawMessage) bool { return len(b) == 0 || string(b) == "null" }
	empty := func(s *string) bool { return s == nil || *s == "" }
	// The original GPT-2 export predates the serialized BPE type discriminator.
	if h.ModelType == "gpt2" && raw.Model.Type == "" {
		raw.Model.Type = "BPE"
	}
	if raw.Version != "1.0" || !null(raw.Truncation) || !null(raw.Padding) || raw.Model.Type != "BPE" || raw.Model.Dropout != nil || raw.Model.Unk != nil || !empty(raw.Model.Prefix) || !empty(raw.Model.Suffix) || raw.Model.Fuse || raw.Model.Fallback || raw.Model.Ignore {
		return nil, errors.New("unsupported HF tokenizer normalization/BPE options")
	}
	// Exact supported pipelines; object key order is immaterial.
	canonical := func(b []byte) string {
		var v any
		if json.Unmarshal(b, &v) != nil {
			return ""
		}
		x, _ := json.Marshal(v)
		return string(x)
	}
	pre := `{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}`
	dec := `{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}`
	qwenByte := []byte("{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}")
	qwenPre, _ := json.Marshal(map[string]any{"type": "Sequence", "pretokenizers": []any{
		map[string]any{"type": "Split", "pattern": map[string]string{"Regex": qwenSplitPattern}, "behavior": "Isolated", "invert": false},
		json.RawMessage(qwenByte),
	}})
	qwen := canonical(raw.PreTokenizer) == canonical(qwenPre) &&
		canonical(raw.Normalizer) == "{\"type\":\"NFC\"}" &&
		canonical(raw.PostProcessor) == canonical(qwenByte) && canonical(raw.Decoder) == canonical(qwenByte)
	smol := null(raw.Normalizer) && null(raw.PostProcessor) &&
		canonical(raw.PreTokenizer) == canonical([]byte(pre)) && canonical(raw.Decoder) == canonical([]byte(dec))
	byteLevel := func(b []byte, prefix, trim bool) bool {
		var spec struct {
			Type   string `json:"type"`
			Prefix bool   `json:"add_prefix_space"`
			Trim   bool   `json:"trim_offsets"`
			Regex  bool   `json:"use_regex"`
		}
		spec.Regex = true // Tokenizers' default before use_regex was serialized.
		var fields map[string]json.RawMessage
		if json.Unmarshal(b, &fields) != nil {
			return false
		}
		for _, value := range fields {
			if string(value) == "null" {
				return false
			}
		}
		d := json.NewDecoder(bytes.NewReader(b))
		d.DisallowUnknownFields()
		return d.Decode(&spec) == nil && spec.Type == "ByteLevel" && spec.Prefix == prefix && spec.Trim == trim && spec.Regex
	}
	gpt2 := h.ModelType == "gpt2" && null(raw.Normalizer) &&
		byteLevel(raw.PreTokenizer, false, true) && byteLevel(raw.PostProcessor, true, false) && byteLevel(raw.Decoder, true, true)
	if !qwen && !smol && !gpt2 {
		return nil, errors.New("unsupported HF tokenizer pipeline")
	}
	var cfg struct {
		AddPrefix  bool                    `json:"add_prefix_space"`
		Added      map[string]hfAddedToken `json:"added_tokens_decoder"`
		Additional []string                `json:"additional_special_tokens"`
		BOS        string                  `json:"bos_token"`
		EOS        string                  `json:"eos_token"`
		Pad        string                  `json:"pad_token"`
		Unk        string                  `json:"unk_token"`
		Chat       string                  `json:"chat_template"`
		Cleanup    bool                    `json:"clean_up_tokenization_spaces"`
		MaxLength  int                     `json:"model_max_length"`
		Class      string                  `json:"tokenizer_class"`
		Vocab      int                     `json:"vocab_size"`
		AddBOS     bool                    `json:"add_bos_token"`
		Split      bool                    `json:"split_special_tokens"`
		Errors     string                  `json:"errors"`
	}
	if err := readStrictBoundedJSON(filepath.Join(dir, "tokenizer_config.json"), 1<<20, &cfg); err != nil {
		return nil, err
	}
	if gpt2 {
		if cfg.Class == "" {
			cfg.Class = "GPT2Tokenizer"
		}
		if cfg.BOS == "" {
			cfg.BOS = "<|endoftext|>"
		}
		if cfg.EOS == "" {
			cfg.EOS = "<|endoftext|>"
		}
		if cfg.Unk == "" {
			cfg.Unk = "<|endoftext|>"
		}
		// The original GPT-2 tokenizer config only sets model_max_length.
		// Its single special token is still validated against the vocabulary.
		if cfg.Added == nil && len(raw.Added) == 1 && raw.Added[0].Content == "<|endoftext|>" && raw.Added[0].Special {
			cfg.Added = map[string]hfAddedToken{strconv.Itoa(raw.Added[0].ID): raw.Added[0]}
		}
	}
	if cfg.AddPrefix || cfg.Cleanup || cfg.AddBOS || cfg.Split || (cfg.Errors != "" && cfg.Errors != "replace") ||
		(smol && (cfg.Class != "GPT2Tokenizer" || cfg.Vocab != h.Vocab || cfg.MaxLength != h.Context)) ||
		(qwen && (cfg.Class != "Qwen2Tokenizer" || cfg.Vocab != 0 && cfg.Vocab != h.Vocab || cfg.MaxLength < h.Context)) ||
		(gpt2 && (cfg.Class != "GPT2Tokenizer" || cfg.Vocab != 0 && cfg.Vocab != h.Vocab || cfg.MaxLength < h.Context || cfg.Chat != "")) ||
		h.Vocab > maxHFVocab || h.Vocab < 258 || h.BOS < 0 || h.BOS >= h.Vocab || h.EOS < 0 || h.EOS >= h.Vocab ||
		len(raw.Model.Vocab) > h.Vocab || len(raw.Added) > 256 || len(raw.Model.Merges) > maxHFVocab {
		return nil, errors.New("HF tokenizer config/model mismatch or unsupported options")
	}
	t := &hfTokenizer{pieces: make([][]byte, h.Vocab), merges: map[Pair]hfMerge{}, bos: h.BOS, eos: h.EOS, qwen: qwen, gpt2: gpt2}
	seen := make([]bool, h.Vocab)
	special := map[int]bool{}
	specialNames := map[string]bool{}
	added := map[int]hfAddedToken{}
	if len(cfg.Added) != len(raw.Added) {
		return nil, errors.New("HF added token table mismatch")
	}
	for _, s := range raw.Added {
		id, inVocab := raw.Model.Vocab[s.Content]
		configured, found := cfg.Added[strconv.Itoa(s.ID)]
		configured.ID = s.ID
		_, repeated := added[s.ID]
		if (inVocab && id != s.ID) || s.ID < 0 || s.ID >= h.Vocab || repeated || len(s.Content) == 0 || len(s.Content) > 256 || s.SingleWord || s.LStrip || s.RStrip || (s.Normalized && !gpt2) || !found || configured != s {
			return nil, errors.New("unsupported or inconsistent HF added token")
		}
		raw.Model.Vocab[s.Content] = s.ID
		added[s.ID] = s
		special[s.ID] = s.Special
		specialNames[s.Content] = s.Special
		t.specials = append(t.specials, s)
	}
	for _, s := range append(append([]string{}, cfg.Additional...), cfg.BOS, cfg.EOS, cfg.Pad, cfg.Unk) {
		if (qwen || gpt2) && s == "" {
			continue
		}
		if !specialNames[s] {
			return nil, errors.New("HF special token not registered")
		}
	}
	if (cfg.BOS != "" && raw.Model.Vocab[cfg.BOS] != h.BOS) || raw.Model.Vocab[cfg.EOS] != h.EOS {
		return nil, errors.New("HF BOS/EOS mismatch")
	}
	sort.Slice(t.specials, func(i, j int) bool {
		if len(t.specials[i].Content) == len(t.specials[j].Content) {
			return t.specials[i].ID < t.specials[j].ID
		}
		return len(t.specials[i].Content) > len(t.specials[j].Content)
	})
	alphabet, reverse := hfByteAlphabet()
	total := 0
	for word, id := range raw.Model.Vocab {
		if id < 0 || id >= h.Vocab || seen[id] || word == "" || len(word) > 65536 {
			return nil, errors.New("invalid HF vocabulary IDs/pieces")
		}
		seen[id] = true
		if s, ok := added[id]; ok {
			if !s.Special {
				t.pieces[id] = []byte(s.Content)
			}
			continue
		}
		for _, r := range word {
			b, ok := reverse[r]
			if !ok {
				return nil, errors.New("HF vocabulary outside byte alphabet")
			}
			t.pieces[id] = append(t.pieces[id], b)
		}
		total += len(t.pieces[id])
		if total > 16<<20 {
			return nil, errors.New("HF vocabulary expansion limit")
		}
	}
	// A tokenizer may omit trailing model padding IDs (e.g. Qwen2.5).
	// They decode to empty text, as HF does, but are not fabricated tokens.
	if len(raw.Model.Vocab) > h.Vocab || !seen[h.BOS] || !seen[h.EOS] {
		return nil, errors.New("HF tokenizer vocabulary exceeds model or lacks BOS/EOS")
	}
	for id := 0; id < len(raw.Model.Vocab); id++ {
		if !seen[id] {
			return nil, errors.New("HF tokenizer has a non-padding vocabulary hole")
		}
	}
	for b, r := range alphabet {
		id, ok := raw.Model.Vocab[string(r)]
		t.bytes[b] = -1
		if ok {
			if special[id] {
				return nil, errors.New("HF byte token is marked special")
			}
			t.bytes[b] = id
		}
	}
	for rank, v := range raw.Model.Merges {
		var pair []string
		if len(v) > 0 && v[0] == '"' {
			var s string
			if err := json.Unmarshal(v, &s); err != nil {
				return nil, err
			}
			pair = strings.Split(s, " ")
		} else {
			if err := json.Unmarshal(v, &pair); err != nil {
				return nil, err
			}
		}
		if len(pair) != 2 {
			return nil, errors.New("invalid HF BPE merge")
		}
		a, okA := raw.Model.Vocab[pair[0]]
		b, okB := raw.Model.Vocab[pair[1]]
		id, okC := raw.Model.Vocab[pair[0]+pair[1]]
		key := Pair{a, b}
		_, exists := t.merges[key]
		_, addedA := added[a]
		_, addedB := added[b]
		_, addedC := added[id]
		if !okA || !okB || !okC || addedA || addedB || addedC || exists {
			return nil, errors.New("invalid or duplicate HF merge")
		}
		t.merges[key] = hfMerge{rank, id}
	}
	t.chat = cfg.Chat == smolChatTemplate && cfg.BOS == "<|im_start|>" && cfg.EOS == "<|im_end|>"
	if qwen {
		sum := sha256.Sum256([]byte(cfg.Chat))
		t.chat = hex.EncodeToString(sum[:]) == qwenChatTemplateSHA &&
			specialNames["<|im_start|>"] && cfg.EOS == "<|im_end|>"
	}
	return t, t.Validate()
}

type hfBPENode struct {
	id, prev, next int
	live           bool
}
type hfBPEJob struct{ left, right, a, b, rank, id int }
type hfBPEHeap []hfBPEJob

func (h hfBPEHeap) Len() int { return len(h) }
func (h hfBPEHeap) Less(i, j int) bool {
	if h[i].rank == h[j].rank {
		return h[i].left < h[j].left
	}
	return h[i].rank < h[j].rank
}
func (h hfBPEHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *hfBPEHeap) Push(x any)   { *h = append(*h, x.(hfBPEJob)) }
func (h *hfBPEHeap) Pop() any     { x := *h; n := len(x) - 1; v := x[n]; *h = x[:n]; return v }
func (t *hfTokenizer) bpe(ctx context.Context, s string) ([]int, error) {
	if len(s) > 65536 {
		return nil, errors.New("HF pre-token exceeds 64 KiB")
	}
	if len(s) == 0 {
		return nil, nil
	}
	nodes := make([]hfBPENode, 0, len(s))
	for i := 0; i < len(s); i++ {
		// With unk_token=null and byte_fallback=false, HF BPE drops
		// alphabet bytes absent from the vocabulary (the target has some).
		if id := t.bytes[s[i]]; id >= 0 {
			n := len(nodes)
			nodes = append(nodes, hfBPENode{id, n - 1, n + 1, true})
		}
	}
	if len(nodes) == 0 {
		return nil, ctx.Err()
	}
	nodes[len(nodes)-1].next = -1
	h := &hfBPEHeap{}
	add := func(left int) {
		if left < 0 || !nodes[left].live {
			return
		}
		right := nodes[left].next
		if right < 0 {
			return
		}
		a, b := nodes[left].id, nodes[right].id
		if m, ok := t.merges[Pair{a, b}]; ok {
			heap.Push(h, hfBPEJob{left, right, a, b, m.rank, m.id})
		}
	}
	for i := range nodes {
		add(i)
	}
	for step := 0; h.Len() > 0; step++ {
		if step%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		x := heap.Pop(h).(hfBPEJob)
		a, b := &nodes[x.left], &nodes[x.right]
		if !a.live || !b.live || a.next != x.right || a.id != x.a || b.id != x.b {
			continue
		}
		a.id = x.id
		a.next = b.next
		b.live = false
		if b.next >= 0 {
			nodes[b.next].prev = x.left
		}
		add(a.prev)
		add(x.left)
	}
	out := make([]int, 0, len(s))
	for i := 0; i >= 0; i = nodes[i].next {
		out = append(out, nodes[i].id)
	}
	return out, ctx.Err()
}

// The GPT-2 regex has a whitespace lookahead unavailable in Go's RE2. Scan the
// same alternatives directly, after Digits has isolated each Unicode number.
func hfWordEnd(s string, start int) int {
	for _, suffix := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
		if strings.HasPrefix(s[start:], suffix) {
			return start + len(suffix)
		}
	}
	class := func(r rune) int {
		if unicode.IsLetter(r) {
			return 1
		}
		if unicode.IsNumber(r) {
			return 2
		}
		if unicode.IsSpace(r) {
			return 3
		}
		return 4
	}
	at := start
	if s[at] == ' ' && at+1 < len(s) {
		r, _ := utf8.DecodeRuneInString(s[at+1:])
		if class(r) != 3 {
			at++
		}
	}
	r, n := utf8.DecodeRuneInString(s[at:])
	kind := class(r)
	end := at + n
	previous := at
	for end < len(s) {
		r, n = utf8.DecodeRuneInString(s[end:])
		if class(r) != kind {
			break
		}
		previous = end
		end += n
	}
	if kind == 3 && end < len(s) && previous > at {
		return previous
	}
	return end
}

// Qwen's Split regex includes lookahead and case folding outside Go's RE2.
// Preserve its ordered alternatives, including punctuation prefixes on words
// and whitespace ending at the final newline.
func qwenWordEnd(s string, start int) int {
	if s[start] == '\'' {
		for _, suffix := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
			end := start
			for range suffix {
				if end < len(s) {
					_, n := utf8.DecodeRuneInString(s[end:])
					end += n
				}
			}
			if strings.EqualFold(s[start:end], suffix) {
				return end
			}
		}
	}
	r, n := utf8.DecodeRuneInString(s[start:])
	at := start
	if !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '\r' && r != '\n' && start+n < len(s) {
		next, _ := utf8.DecodeRuneInString(s[start+n:])
		if unicode.IsLetter(next) {
			at += n
		}
	}
	end := at
	for end < len(s) {
		r, n := utf8.DecodeRuneInString(s[end:])
		if !unicode.IsLetter(r) {
			break
		}
		end += n
	}
	if end > at {
		return end
	}
	r, n = utf8.DecodeRuneInString(s[start:])
	if unicode.IsNumber(r) {
		return start + n
	}
	punct := func(r rune) bool { return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) }
	at = start
	if r == ' ' && start+n < len(s) {
		next, _ := utf8.DecodeRuneInString(s[start+n:])
		if punct(next) {
			at += n
		}
	}
	end = at
	for end < len(s) {
		r, n := utf8.DecodeRuneInString(s[end:])
		if !punct(r) {
			break
		}
		end += n
	}
	if end > at {
		for end < len(s) && (s[end] == '\r' || s[end] == '\n') {
			end++
		}
		return end
	}
	end, previous, newline := start, start, start
	for end < len(s) {
		r, n := utf8.DecodeRuneInString(s[end:])
		if !unicode.IsSpace(r) {
			break
		}
		previous = end
		end += n
		if r == '\r' || r == '\n' {
			newline = end
		}
	}
	if newline > start {
		return newline
	}
	if end < len(s) && previous > start {
		return previous
	}
	return end
}

func qwenNormalize(s string) (string, error) {
	out := norm.NFC.String(s)
	// x/text inserts CGJ for stream safety after 30 non-starters; HF does
	// not. Reject that bounded corner instead of silently changing token IDs.
	if strings.Count(out, "\u034f") != strings.Count(s, "\u034f") {
		return "", errors.New("Qwen NFC combining sequence exceeds supported normalization bound")
	}
	if len(out) > 1<<20 {
		return "", errors.New("normalized HF prompt exceeds 1 MiB")
	}
	return out, nil
}

func (t *hfTokenizer) encodePrompt(ctx context.Context, s string, limit int) ([]int, error) {
	if !utf8.ValidString(s) || len(s) > 1<<20 {
		return nil, errors.New("HF prompt must be valid UTF-8 and at most 1 MiB")
	}
	out := make([]int, 0, min(len(s), max(limit, 0)))
	appendIDs := func(ids []int) error {
		if len(ids) > limit-len(out) {
			return errors.New("prompt plus generation exceeds context")
		}
		out = append(out, ids...)
		return nil
	}
	// Longest matching special token wins at the earliest byte position.
	for pos := 0; pos < len(s); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end, matched := len(s), -1
		var first [256]bool
		for _, sp := range t.specials {
			first[sp.Content[0]] = true
		}
	searchSpecial:
		for at := pos; at < len(s); at++ {
			if at%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if !first[s[at]] {
				continue
			}
			for i, sp := range t.specials {
				if strings.HasPrefix(s[at:], sp.Content) {
					end, matched = at, i
					break searchSpecial
				}
			}
		}
		plain := s[pos:end]
		if t.qwen {
			var err error
			plain, err = qwenNormalize(plain)
			if err != nil {
				return nil, err
			}
		}
		// Digits is applied before the byte-level regex, including whitespace.
		for lo := 0; lo < len(plain); {
			r, n := utf8.DecodeRuneInString(plain[lo:])
			hi := lo + n
			if t.qwen || t.gpt2 {
				hi = len(plain)
			} else if !unicode.IsNumber(r) {
				for hi < len(plain) {
					r, n = utf8.DecodeRuneInString(plain[hi:])
					if unicode.IsNumber(r) {
						break
					}
					hi += n
				}
			}
			segment := plain[lo:hi]
			for a := 0; a < len(segment); {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				b := hfWordEnd(segment, a)
				if t.qwen {
					b = qwenWordEnd(segment, a)
				}
				ids, err := t.bpe(ctx, segment[a:b])
				if err != nil {
					return nil, err
				}
				if err = appendIDs(ids); err != nil {
					return nil, err
				}
				a = b
			}
			lo = hi
		}
		pos = end
		if matched >= 0 {
			sp := t.specials[matched]
			if err := appendIDs([]int{sp.ID}); err != nil {
				return nil, err
			}
			pos += len(sp.Content)
		}
	}
	return out, ctx.Err()
}

func (t Tokenizer) Vocab() int { return baseVocab + len(t.Merges) }
func (t Tokenizer) Validate() error {
	if t.Vocab() > 65536 {
		return errors.New("tokenizer exceeds 65536 tokens")
	}
	lengths := make([]int, t.Vocab())
	for i := 0; i < 256; i++ {
		lengths[i] = 1
	}
	seen := make(map[Pair]bool)
	totalPieceBytes := 256
	for i, p := range t.Merges {
		id := baseVocab + i
		if p.A < 0 || p.B < 0 || p.A >= id || p.B >= id || p.A == BOS || p.A == EOS || p.B == BOS || p.B == EOS || seen[p] {
			return fmt.Errorf("invalid BPE merge %d", i)
		}
		lengths[id] = lengths[p.A] + lengths[p.B]
		totalPieceBytes += lengths[id]
		if lengths[id] > 65536 || totalPieceBytes > 16*1024*1024 {
			return fmt.Errorf("BPE merge %d exceeds token/table expansion limit", i)
		}
		seen[p] = true
	}
	return nil
}
func mergePair(ids []int, p Pair, id int) []int {
	out := make([]int, 0, len(ids))
	for i := 0; i < len(ids); i++ {
		if i+1 < len(ids) && ids[i] == p.A && ids[i+1] == p.B {
			out = append(out, id)
			i++
		} else {
			out = append(out, ids[i])
		}
	}
	return out
}
func (t Tokenizer) Encode(s string, bos, eos bool) []int {
	ids := make([]int, len(s))
	for i := range []byte(s) {
		ids[i] = int(s[i])
	}
	ranks := make(map[Pair]int, len(t.Merges))
	for i, p := range t.Merges {
		ranks[p] = i
	}
	for {
		best := len(t.Merges)
		for i := 0; i+1 < len(ids); i++ {
			if rank, ok := ranks[Pair{ids[i], ids[i+1]}]; ok && rank < best {
				best = rank
			}
		}
		if best == len(t.Merges) {
			break
		}
		ids = mergePair(ids, t.Merges[best], baseVocab+best)
	}
	if bos {
		ids = append([]int{BOS}, ids...)
	}
	if eos {
		ids = append(ids, EOS)
	}
	return ids
}
func (t Tokenizer) Pieces() [][]byte {
	pieces := make([][]byte, t.Vocab())
	for i := 0; i < 256; i++ {
		pieces[i] = []byte{byte(i)}
	}
	for i, p := range t.Merges {
		pieces[baseVocab+i] = append(append([]byte(nil), pieces[p.A]...), pieces[p.B]...)
	}
	return pieces
}
func (t Tokenizer) Decode(ids []int) string {
	pieces := t.Pieces()
	var out []byte
	for _, id := range ids {
		if id >= 0 && id < len(pieces) {
			out = append(out, pieces[id]...)
		}
	}
	return string(out)
}
func TrainTokenizer(text string, vocab int) (Tokenizer, error) {
	t := Tokenizer{}
	if vocab < baseVocab || vocab > 8192 || len(text) > 4*1024*1024 {
		return t, errors.New("BPE training supports vocab 258..8192 and at most 4 MiB of text")
	}
	ids := t.Encode(text, false, false)
	for t.Vocab() < vocab {
		counts := make(map[Pair]int)
		for i := 0; i+1 < len(ids); i++ {
			counts[Pair{ids[i], ids[i+1]}]++
		}
		best := Pair{}
		count := 1
		for p, n := range counts {
			if n > count || n == count && n > 1 && (p.A < best.A || p.A == best.A && p.B < best.B) {
				best = p
				count = n
			}
		}
		if count < 2 {
			break
		}
		ids = mergePair(ids, best, t.Vocab())
		t.Merges = append(t.Merges, best)
	}
	return t, t.Validate()
}

// Token shards contain complete, independently tokenized documents. A document
// is one input line, retaining its newline bytes. No BPE merge crosses a document.
const (
	shardMagic       = "MGTS0001"
	shardHeaderBytes = 64
	maxDatasetShards = 16384
	maxDocumentBytes = 1 << 20
	maxReadTokens    = 1 << 20
)

type TokenShard struct {
	File      string `json:"file"`
	Split     string `json:"split"`
	Tokens    int64  `json:"tokens"`
	Documents int64  `json:"documents"`
	Bytes     int64  `json:"bytes"`
	SHA256    string `json:"sha256"`
}
type DatasetManifest struct {
	Format          string       `json:"format"`
	Tokenizer       Tokenizer    `json:"tokenizer"`
	TokenizerSHA256 string       `json:"tokenizer_sha256"`
	Vocab           int          `json:"vocab"`
	Shards          []TokenShard `json:"shards"`
}
type TokenDataset struct {
	root     string
	manifest DatasetManifest
	id       string
}

func tokenizerHash(tok Tokenizer) string {
	// Normalize nil and empty merge tables to the same identity.
	merges := append(make([]Pair, 0, len(tok.Merges)), tok.Merges...)
	b, _ := json.Marshal(Tokenizer{Merges: merges})
	return corpusHash(string(b))
}
func validSHA256(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size && s == strings.ToLower(s)
}
func shardHeader(m DatasetManifest, s TokenShard) []byte {
	b := make([]byte, shardHeaderBytes)
	copy(b, shardMagic)
	binary.LittleEndian.PutUint32(b[8:], uint32(m.Vocab))
	binary.LittleEndian.PutUint64(b[16:], uint64(s.Tokens))
	binary.LittleEndian.PutUint64(b[24:], uint64(s.Documents))
	digest, _ := hex.DecodeString(m.TokenizerSHA256)
	copy(b[32:], digest)
	return b
}
func (m DatasetManifest) Validate() error {
	if m.Format != "monolith-tokens-v1" || len(m.Shards) == 0 || len(m.Shards) > maxDatasetShards {
		return errors.New("invalid dataset format or shard count")
	}
	if err := m.Tokenizer.Validate(); err != nil {
		return err
	}
	if m.Vocab != m.Tokenizer.Vocab() || m.TokenizerSHA256 != tokenizerHash(m.Tokenizer) {
		return errors.New("dataset tokenizer identity/vocabulary mismatch")
	}
	seen := make(map[string]bool)
	train, val := false, false
	for _, s := range m.Shards {
		if s.File == "" || s.File == "." || filepath.Base(s.File) != s.File || strings.ContainsAny(s.File, "/\\") || seen[s.File] {
			return errors.New("shard filenames must be unique local basenames")
		}
		seen[s.File] = true
		if s.Split != "train" && s.Split != "validation" {
			return errors.New("unknown dataset split")
		}
		train = train || s.Split == "train"
		val = val || s.Split == "validation"
		if s.Tokens < 3 || s.Tokens > 1<<50 || s.Documents < 1 || s.Documents > s.Tokens/3 || s.Bytes != shardHeaderBytes+4*s.Tokens || !validSHA256(s.SHA256) {
			return errors.New("invalid shard size/count/checksum metadata")
		}
	}
	if !train || !val {
		return errors.New("dataset requires nonempty train and validation splits")
	}
	return nil
}

// Opening verifies the entire dataset using a fixed buffer before training.
// Files must remain immutable until all readers/prefetchers have closed.
func OpenTokenDataset(ctx context.Context, path string) (*TokenDataset, error) {
	var m DatasetManifest
	if err := readJSONFile(path, &m); err != nil {
		return nil, err
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	d := &TokenDataset{root: filepath.Dir(path), manifest: m}
	canonical, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	d.id = corpusHash(string(canonical))
	for i := range m.Shards {
		if err := d.verifyShard(ctx, i); err != nil {
			return nil, fmt.Errorf("shard %s: %w", m.Shards[i].File, err)
		}
	}
	return d, nil
}
func (d *TokenDataset) openShard(index int) (*os.File, error) {
	if index < 0 || index >= len(d.manifest.Shards) {
		return nil, errors.New("shard index outside dataset")
	}
	s := d.manifest.Shards[index]
	path := filepath.Join(d.root, s.File)
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("shard must be a regular file (no symlinks)")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*os.File, error) { f.Close(); return nil, err }
	info, err = f.Stat()
	if err != nil {
		return fail(err)
	}
	if !info.Mode().IsRegular() || info.Size() != s.Bytes {
		return fail(errors.New("shard size mismatch"))
	}
	var header [shardHeaderBytes]byte
	if _, err = io.ReadFull(f, header[:]); err != nil {
		return fail(err)
	}
	if !bytes.Equal(header[:], shardHeader(d.manifest, s)) {
		return fail(errors.New("shard header/tokenizer mismatch"))
	}
	return f, nil
}
func (d *TokenDataset) verifyShard(ctx context.Context, index int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	f, err := d.openShard(index)
	if err != nil {
		return err
	}
	defer f.Close()
	s := d.manifest.Shards[index]
	h := sha256.New()
	h.Write(shardHeader(d.manifest, s))
	buf := make([]byte, 16*1024)
	inside, payload := false, false
	var documents int64
	for left := s.Tokens; left > 0; {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := int(min(left, int64(len(buf)/4)))
		if _, err = io.ReadFull(f, buf[:4*n]); err != nil {
			return err
		}
		h.Write(buf[:4*n])
		for j := 0; j < n; j++ {
			id := int(binary.LittleEndian.Uint32(buf[4*j:]))
			if id >= d.manifest.Vocab {
				return errors.New("token outside vocabulary")
			}
			switch {
			case !inside:
				if id != BOS {
					return errors.New("document must start with BOS")
				}
				inside, payload = true, false
			case id == BOS:
				return errors.New("unexpected BOS inside document")
			case id == EOS:
				if !payload {
					return errors.New("empty document")
				}
				inside = false
				documents++
			default:
				payload = true
			}
		}
		left -= int64(n)
	}
	if inside || documents != s.Documents {
		return errors.New("document boundary/count mismatch")
	}
	if hex.EncodeToString(h.Sum(nil)) != s.SHA256 {
		return errors.New("shard checksum mismatch")
	}
	return nil
}

// ReadTokens fills caller-owned memory, opening at most one file. No corpus-wide
// token array, per-document index, mmap, or retained file descriptor is needed.
func (d *TokenDataset) ReadTokens(ctx context.Context, shard int, offset int64, dst []int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if shard < 0 || shard >= len(d.manifest.Shards) || len(dst) > maxReadTokens {
		return errors.New("invalid shard or read allocation")
	}
	s := d.manifest.Shards[shard]
	if offset < 0 || offset > s.Tokens-int64(len(dst)) {
		return errors.New("token read outside shard")
	}
	f, err := d.openShard(shard)
	if err != nil {
		return err
	}
	defer f.Close()
	r := io.NewSectionReader(f, shardHeaderBytes+4*offset, int64(len(dst))*4)
	buf := make([]byte, min(16*1024, 4*len(dst)))
	for start := 0; start < len(dst); {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := min(len(dst)-start, len(buf)/4)
		if _, err = io.ReadFull(r, buf[:4*n]); err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			id := int(binary.LittleEndian.Uint32(buf[4*i:]))
			if id >= d.manifest.Vocab {
				return errors.New("token outside vocabulary; dataset changed")
			}
			dst[start+i] = id
		}
		start += n
	}
	return nil
}

type prepareOptions struct {
	Train, Validation, Out     string
	Tokenizer                  Tokenizer
	ShardTokens, DocumentBytes int
}

// Bounded line reads never split a document to satisfy a buffer size. A long
// line is rejected, so UTF-8 bytes and BPE merges remain identical to Encode.
func readDocument(r *bufio.Reader, limit int) ([]byte, error) {
	var doc []byte
	for {
		part, err := r.ReadSlice('\n')
		if len(doc)+len(part) > limit {
			return nil, errors.New("document exceeds -document-bytes; split documents explicitly before preparation")
		}
		doc = append(doc, part...)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err == io.EOF && len(doc) > 0 {
			return doc, nil
		}
		return doc, err
	}
}

func PrepareTokenDataset(ctx context.Context, o prepareOptions) error {
	if o.Train == "" || o.Validation == "" || o.Out == "" || o.DocumentBytes < 1 || o.DocumentBytes > maxDocumentBytes || o.ShardTokens < 3 || o.ShardTokens > 16<<20 {
		return errors.New("prepare requires train, validation, out; document-bytes 1..1048576; shard-tokens 3..16777216")
	}
	if err := o.Tokenizer.Validate(); err != nil {
		return err
	}
	if _, err := os.Lstat(o.Out); !errors.Is(err, os.ErrNotExist) {
		if err == nil {
			return errors.New("dataset output already exists")
		}
		return err
	}
	parent := filepath.Dir(filepath.Clean(o.Out))
	if err := os.MkdirAll(parent, 0755); err != nil {
		return err
	}
	dir, err := os.MkdirTemp(parent, ".monolith-dataset-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	m := DatasetManifest{Format: "monolith-tokens-v1", Tokenizer: o.Tokenizer, TokenizerSHA256: tokenizerHash(o.Tokenizer), Vocab: o.Tokenizer.Vocab()}
	for _, input := range []struct{ path, split string }{{o.Train, "train"}, {o.Validation, "validation"}} {
		if err = prepareSplit(ctx, dir, input.path, input.split, o, &m); err != nil {
			return err
		}
	}
	if err = m.Validate(); err != nil {
		return err
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if len(b)+1 > 4<<20 {
		return errors.New("dataset manifest exceeds 4 MiB")
	}
	f, err := os.OpenFile(filepath.Join(dir, "manifest.json"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	_, err = f.Write(append(b, '\n'))
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	// A prepared dataset becomes visible only after every shard is complete.
	if f, e := os.Open(dir); e == nil {
		_ = f.Sync()
		_ = f.Close()
	}
	if err = os.Rename(dir, o.Out); err != nil {
		return err
	}
	if f, e := os.Open(parent); e == nil {
		_ = f.Sync()
		_ = f.Close()
	}
	return nil
}

func prepareSplit(ctx context.Context, dir, path, split string, o prepareOptions, m *DatasetManifest) error {
	in, err := os.Open(path)
	if err != nil {
		return err
	}
	defer in.Close()
	r := bufio.NewReaderSize(in, 16*1024)
	var f *os.File
	var s TokenShard
	defer func() {
		if f != nil {
			_ = f.Close()
		}
	}()
	finish := func() error {
		if f == nil {
			return nil
		}
		s.Bytes = shardHeaderBytes + 4*s.Tokens
		if _, err := f.WriteAt(shardHeader(*m, s), 0); err != nil {
			return err
		}
		h := sha256.New()
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return err
		}
		buf := make([]byte, 16*1024)
		for {
			if err := ctx.Err(); err != nil {
				return err
			}
			n, err := f.Read(buf)
			h.Write(buf[:n])
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
		}
		s.SHA256 = hex.EncodeToString(h.Sum(nil))
		if err := f.Sync(); err != nil {
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		f = nil
		m.Shards = append(m.Shards, s)
		return nil
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		doc, err := readDocument(r, o.DocumentBytes)
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("%s: %w", split, err)
		}
		ids := o.Tokenizer.Encode(string(doc), true, true)
		if len(ids) > o.ShardTokens {
			return errors.New("one document exceeds -shard-tokens")
		}
		if f != nil && s.Tokens+int64(len(ids)) > int64(o.ShardTokens) {
			if err = finish(); err != nil {
				return err
			}
		}
		if f == nil {
			if len(m.Shards) >= maxDatasetShards {
				return errors.New("too many dataset shards")
			}
			s = TokenShard{File: fmt.Sprintf("%s-%06d.mgts", split, len(m.Shards)), Split: split}
			f, err = os.OpenFile(filepath.Join(dir, s.File), os.O_CREATE|os.O_EXCL|os.O_RDWR, 0644)
			if err != nil {
				return err
			}
			if _, err = f.Write(make([]byte, shardHeaderBytes)); err != nil {
				return err
			}
		}
		buf := make([]byte, 4*len(ids))
		for i, id := range ids {
			binary.LittleEndian.PutUint32(buf[4*i:], uint32(id))
		}
		if _, err = f.Write(buf); err != nil {
			return err
		}
		s.Tokens += int64(len(ids))
		s.Documents++
	}
	return finish()
}

func runPrepare(ctx context.Context, args []string) error {
	f := flags("prepare")
	o := prepareOptions{}
	f.StringVar(&o.Train, "train", "", "training text: one document per line, newline retained")
	f.StringVar(&o.Validation, "validation", "", "separate validation text in the same format")
	f.StringVar(&o.Out, "out", "", "new output directory (must not exist)")
	f.IntVar(&o.ShardTokens, "shard-tokens", 1<<20, "maximum tokens per shard, including BOS/EOS")
	f.IntVar(&o.DocumentBytes, "document-bytes", 64<<10, "maximum bytes per document, including newline")
	tokPath := f.String("tokenizer", "", "existing tokenizer JSON; default is byte vocabulary")
	if err := parse(f, args); err != nil {
		return err
	}
	if *tokPath != "" {
		if err := readJSONFile(*tokPath, &o.Tokenizer); err != nil {
			return err
		}
	}
	return PrepareTokenDataset(ctx, o)
}

// DatasetCursor points to the next target, not the end of speculative reads.
// Shard is a position in the epoch's deterministic shard order, not a file index.
type DatasetCursor struct {
	Epoch  uint64 `json:"epoch"`
	Shard  int    `json:"shard"`
	Offset int64  `json:"offset"`
}
type PrefetchConfig struct {
	Batch, Seq, Workers, Depth int
	Seed                       uint64
	Shuffle                    bool
}

type tokenSpan struct {
	shard      int
	offset     int64
	begin, end int
}
type tokenBuffer struct{ x, y []int }
type tokenJob struct {
	id     uint64
	buffer *tokenBuffer
	spans  []tokenSpan
	after  DatasetCursor
	err    error
}

// A batch lease owns its arrays until Release. Do not access them afterward.
// Credits follow ownership through dispatch, IO, reordering and the consumer.
type TokenBatch struct {
	X, Y    []int
	After   DatasetCursor
	err     error
	release func()
	once    sync.Once
}

func (b *TokenBatch) Release() { b.once.Do(b.release) }

type TokenPrefetch struct {
	ctx    context.Context
	cancel context.CancelFunc
	out    chan *TokenBatch
	wg     sync.WaitGroup
}

func (p *TokenPrefetch) Close() { p.cancel(); p.wg.Wait() }
func (p *TokenPrefetch) Next(ctx context.Context) (*TokenBatch, error) {
	if err := ctx.Err(); err != nil {
		p.cancel()
		return nil, err
	}
	select {
	case b, ok := <-p.out:
		if !ok {
			return nil, p.ctx.Err()
		}
		if b.err != nil {
			b.Release()
			p.cancel()
			return nil, b.err
		}
		return b, nil
	case <-ctx.Done():
		p.cancel()
		return nil, ctx.Err()
	case <-p.ctx.Done():
		return nil, p.ctx.Err()
	}
}

type tokenPlanner struct {
	d              *TokenDataset
	indices, order []int
	config         PrefetchConfig
	cursor         DatasetCursor
	perEpoch       int64
}

func (c PrefetchConfig) Validate() error {
	if c.Batch < 1 || c.Batch > 4096 || c.Seq < 1 || c.Seq > 32768 || c.Workers < 1 || c.Workers > 64 || c.Depth < 1 || c.Depth > 64 || c.Workers > c.Depth {
		return errors.New("invalid prefetch settings; workers 1..depth, depth 1..64")
	}
	// Conservative allowance for two int arrays plus range descriptors and
	// slice growth; metadata and at most 16 KiB per IO worker are additional.
	if int64(c.Batch)*int64(c.Seq) > maxReadTokens || int64(c.Batch)*int64(c.Seq)*int64(c.Depth)*64 > 128<<20 {
		return errors.New("prefetch staging exceeds 128 MiB allowance")
	}
	return nil
}
func newTokenPlanner(d *TokenDataset, split string, c PrefetchConfig, cursor DatasetCursor) (*tokenPlanner, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	p := &tokenPlanner{d: d, config: c, cursor: cursor}
	for i, s := range d.manifest.Shards {
		if s.Split != split {
			continue
		}
		if p.perEpoch > math.MaxInt64-(s.Tokens-1) {
			return nil, errors.New("dataset token count overflows int64")
		}
		p.perEpoch += s.Tokens - 1
		p.indices = append(p.indices, i)
	}
	if len(p.indices) == 0 {
		return nil, errors.New("empty or unknown dataset split")
	}
	p.setOrder()
	if cursor.Shard < 0 || cursor.Shard >= len(p.order) || cursor.Offset < 1 || cursor.Offset >= d.manifest.Shards[p.order[cursor.Shard]].Tokens {
		return nil, errors.New("dataset cursor outside shard")
	}
	return p, nil
}
func (p *tokenPlanner) setOrder() {
	p.order = append(p.order[:0], p.indices...)
	if p.config.Shuffle {
		rng := RNG{p.config.Seed ^ (p.cursor.Epoch * 0x9e3779b97f4a7c15)}
		for i := len(p.order) - 1; i > 0; i-- {
			j := rng.Intn(i + 1)
			p.order[i], p.order[j] = p.order[j], p.order[i]
		}
	}
}
func (p *tokenPlanner) plan(ctx context.Context) ([]tokenSpan, DatasetCursor, error) {
	var spans []tokenSpan
	for written, n := 0, p.config.Batch*p.config.Seq; written < n; {
		if err := ctx.Err(); err != nil {
			return nil, p.cursor, err
		}
		index := p.order[p.cursor.Shard]
		s := p.d.manifest.Shards[index]
		count := int(min(int64(n-written), s.Tokens-p.cursor.Offset))
		spans = append(spans, tokenSpan{index, p.cursor.Offset, written, written + count})
		written += count
		p.cursor.Offset += int64(count)
		if p.cursor.Offset == s.Tokens {
			p.cursor.Offset = 1 // Skip only the shard's leading BOS.
			p.cursor.Shard++
			if p.cursor.Shard == len(p.order) {
				if p.cursor.Epoch == math.MaxUint64 {
					return nil, p.cursor, errors.New("dataset epoch overflow")
				}
				p.cursor.Epoch++
				p.cursor.Shard = 0
				p.setOrder()
			}
		}
	}
	return spans, p.cursor, nil
}

type tokenReadFunc func(context.Context, int, int64, []int) error

// Repeated epochs in one microbatch can revisit the same ranges many times.
// Fill in physical range order, copying covered ranges from this job's existing
// target buffer. Logical order is encoded in begin/end, so no extra token cache
// or file descriptors are needed and output order remains unchanged.
func fillTokenJob(ctx context.Context, job *tokenJob, read tokenReadFunc) {
	if job.err != nil {
		return
	}
	if len(job.spans) > 1 {
		sort.Slice(job.spans, func(i, j int) bool {
			a, b := job.spans[i], job.spans[j]
			if a.shard != b.shard {
				return a.shard < b.shard
			}
			if a.offset != b.offset {
				return a.offset < b.offset
			}
			return a.end-a.begin > b.end-b.begin
		})
	}
	var previous tokenSpan
	havePrevious := false
	for _, span := range job.spans {
		if job.err = ctx.Err(); job.err != nil {
			return
		}
		delta := span.offset - previous.offset
		n := span.end - span.begin
		if havePrevious && span.shard == previous.shard && delta >= 0 && delta+int64(n) <= int64(previous.end-previous.begin) {
			start := previous.begin + int(delta)
			copy(job.buffer.y[span.begin:span.end], job.buffer.y[start:start+n])
			continue
		}
		if job.err = read(ctx, span.shard, span.offset, job.buffer.y[span.begin:span.end]); job.err != nil {
			return
		}
		previous = span
		havePrevious = true
	}
}

func NewTokenPrefetch(ctx context.Context, d *TokenDataset, split string, c PrefetchConfig, cursor DatasetCursor) (*TokenPrefetch, error) {
	return newTokenPrefetch(ctx, d, split, c, cursor, d.ReadTokens)
}
func newTokenPrefetch(ctx context.Context, d *TokenDataset, split string, c PrefetchConfig, cursor DatasetCursor, read tokenReadFunc) (*TokenPrefetch, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	planner, err := newTokenPlanner(d, split, c, cursor)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	p := &TokenPrefetch{ctx: ctx, cancel: cancel, out: make(chan *TokenBatch)}
	free := make(chan *tokenBuffer, c.Depth)
	jobs := make(chan tokenJob, c.Depth)
	type result struct {
		id    uint64
		batch *TokenBatch
	}
	results := make(chan result, c.Depth)
	n := c.Batch * c.Seq
	for i := 0; i < c.Depth; i++ {
		storage := make([]int, 2*n)
		free <- &tokenBuffer{storage[:n:n], storage[n:]}
	}
	p.wg.Add(c.Workers + 2)
	// Only this goroutine advances the speculative cursor and shard-order RNG.
	go func() {
		defer p.wg.Done()
		defer close(jobs)
		for id := uint64(0); ; id++ {
			if ctx.Err() != nil {
				return
			}
			var buffer *tokenBuffer
			select {
			case buffer = <-free:
			case <-ctx.Done():
				return
			}
			spans, after, err := planner.plan(ctx)
			job := tokenJob{id, buffer, spans, after, err}
			select {
			case jobs <- job:
			case <-ctx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()
	for worker := 0; worker < c.Workers; worker++ {
		go func() {
			defer p.wg.Done()
			for {
				if ctx.Err() != nil {
					return
				}
				var job tokenJob
				var ok bool
				select {
				case job, ok = <-jobs:
					if !ok {
						return
					}
				case <-ctx.Done():
					return
				}
				fillTokenJob(ctx, &job, read)
				if job.err == nil {
					for row := 0; row < c.Batch; row++ {
						begin, end := row*c.Seq, (row+1)*c.Seq
						job.buffer.x[begin] = BOS
						copy(job.buffer.x[begin+1:end], job.buffer.y[begin:end-1])
					}
				}
				buffer := job.buffer
				b := &TokenBatch{X: buffer.x, Y: buffer.y, After: job.after, err: job.err, release: func() {
					select {
					case free <- buffer:
					case <-ctx.Done():
					}
				}}
				select {
				case results <- result{job.id, b}:
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	go func() {
		defer p.wg.Done()
		defer close(p.out)
		defer cancel()
		pending := make(map[uint64]*TokenBatch)
		for next := uint64(0); ; {
			var r result
			select {
			case r = <-results:
			case <-ctx.Done():
				return
			}
			pending[r.id] = r.batch
			for {
				b, ok := pending[next]
				if !ok {
					break
				}
				select {
				case p.out <- b:
				case <-ctx.Done():
					return
				}
				delete(pending, next)
				next++
				if b.err != nil {
					return
				}
			}
		}
	}()
	return p, nil
}

// Parallel kernels partition independent output rows; reductions have a fixed
// order, so changing GOMAXPROCS does not change arithmetic or resume trajectories.
func parallel(n, work int, fn func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers < 2 || work < 131072 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := n*w/workers, n*(w+1)/workers
		wg.Add(1)
		go func() { defer wg.Done(); fn(lo, hi) }()
	}
	wg.Wait()
}

// A kernelExecutor belongs to one engine actor. Its fixed workers borrow one
// kernel's disjoint output ranges, then join before the actor advances. There
// is no request backlog here: parallel and close must have a single caller,
// and callbacks must not recursively dispatch work through this executor.
type kernelExecutor struct {
	workers   int
	jobs      chan kernelJob
	completed chan struct{}
	lifetime  sync.WaitGroup
}

type kernelJob struct {
	lo, hi int
	fn     func(lo, hi int)
}

// workers is validated by newEngine (1..256). One worker runs on the actor.
func newKernelExecutor(workers int) *kernelExecutor {
	e := &kernelExecutor{workers: workers}
	if workers > 1 {
		e.jobs = make(chan kernelJob, workers)
		e.completed = make(chan struct{}, workers)
		e.lifetime.Add(workers)
		for i := 0; i < workers; i++ {
			go e.run()
		}
	}
	return e
}

func (e *kernelExecutor) run() {
	defer e.lifetime.Done()
	for job := range e.jobs {
		job.fn(job.lo, job.hi)
		// A completion acknowledges both the writes and the end of borrowing.
		// Idle workers must not retain a closure over pooled scratch or KV pages.
		job = kernelJob{}
		e.completed <- struct{}{}
	}
}

func (e *kernelExecutor) parallel(n, work int, fn func(lo, hi int)) {
	if e == nil {
		parallel(n, work, fn)
		return
	}
	workers := min(e.workers, n)
	if workers < 2 || work < 131072 {
		fn(0, n)
		return
	}
	for w := 0; w < workers; w++ {
		e.jobs <- kernelJob{n * w / workers, n * (w + 1) / workers, fn}
	}
	for w := 0; w < workers; w++ {
		<-e.completed
	}
}

// Called once by the actor after its final joined kernel. No worker outlives
// Engine.Close, including engines that were closed without serving a request.
func (e *kernelExecutor) close() {
	if e.jobs != nil {
		close(e.jobs)
		e.lifetime.Wait()
	}
}

type Param struct {
	Name             string
	Rows, Cols       int
	Decay            bool
	Data, Grad, M, V []float32
	Q8               *q8Weights // Inference-only; Data is nil when present.
	Q4               *q4Weights // Packed inference-only weights; no FP32 copy.
}

const q8Block = 64

// Symmetric weight-only quantization. Blocks never cross an output row.
// Activations, ordinary accumulation, normalization, biases and KV stay FP32.
type q8Weights struct {
	Values []int8
	Scales []float32
}

// Two signed four-bit weights per byte, low nibble first. Odd-width rows have
// a zero high padding nibble; blocks and packing never cross a row boundary.
type q4Weights struct {
	Values []byte
	Scales []float32
}

// Read-only after initialization. A shared 2 KiB table decodes two weights at
// once; four prompt/session rows reuse each pair. No per-call LUT or scratch.
var q4Pairs = func() (table [256][2]float32) {
	for i := range table {
		table[i][0] = float32(int8(byte(i)<<4) >> 4)
		table[i][1] = float32(int8(byte(i)) >> 4)
	}
	return
}()

func (q *q4Weights) put(cols, row, col int, v int8) {
	i, shift := row*((cols+1)/2)+col/2, uint(col%2)*4
	q.Values[i] = q.Values[i]&^(15<<shift) | (byte(v)&15)<<shift
}

func (p *Param) quantized() bool { return p.Q8 != nil || p.Q4 != nil }

func (p *Param) quantScales() []float32 {
	if p.Q4 != nil {
		return p.Q4.Scales
	}
	return p.Q8.Scales
}

func (p *Param) quantScale(amax float32) float32 {
	if p.Q4 != nil {
		return lowBitScale(amax, 7)
	}
	return q8Scale(amax)
}

func (p *Param) putQuantized(row, col int, v, scale float32) {
	if p.Q4 != nil {
		q := int8(0)
		if scale != 0 {
			q = int8(max(-7, min(7, math.Round(float64(v)/float64(scale)))))
		}
		p.Q4.put(p.Cols, row, col, q)
	} else {
		p.Q8.Values[row*p.Cols+col] = q8Value(v, scale)
	}
}

func newInferenceParamFormat(name string, rows, cols int, matrix bool, format string) *Param {
	format = hfMatrixFormat(format, name)
	if matrix && format == "q4" {
		return &Param{Name: name, Rows: rows, Cols: cols, Q4: &q4Weights{
			Values: make([]byte, rows*((cols+1)/2)),
			Scales: make([]float32, rows*((cols+q8Block-1)/q8Block)),
		}}
	}
	return newInferenceParam(name, rows, cols, matrix && format == "q8")
}

func newInferenceParam(name string, rows, cols int, quantized bool) *Param {
	p := &Param{Name: name, Rows: rows, Cols: cols}
	if quantized {
		p.Q8 = &q8Weights{Values: make([]int8, rows*cols), Scales: make([]float32, rows*((cols+q8Block-1)/q8Block))}
	} else {
		p.Data = make([]float32, rows*cols)
	}
	return p
}

func q8Scale(amax float32) float32 {
	return lowBitScale(amax, 127)
}

func lowBitScale(amax float32, levels float64) float32 {
	if amax == 0 {
		return 0
	}
	scale := float32(float64(amax) / levels)
	if scale == 0 {
		return math.SmallestNonzeroFloat32
	}
	// Round the scale downward if needed so reconstruction stays finite even
	// when a finite source value is near MaxFloat32.
	if scale > math.SmallestNonzeroFloat32 && float64(scale)*levels > float64(amax) {
		scale = math.Nextafter32(scale, 0)
	}
	return scale
}

func q8Value(v, scale float32) int8 {
	if scale == 0 {
		return 0
	}
	return int8(max(-127, min(127, math.Round(float64(v)/float64(scale)))))
}

func (p *Param) copyRow(dst []float32, row int) {
	if p.Q4 != nil {
		stride, groups := (p.Cols+1)/2, (p.Cols+q8Block-1)/q8Block
		wr := p.Q4.Values[row*stride : (row+1)*stride]
		for j := 0; j < p.Cols; j++ {
			dst[j] = q4Pairs[wr[j/2]][j%2] * p.Q4.Scales[row*groups+j/q8Block]
		}
		return
	}
	if p.Q8 == nil {
		copy(dst, p.Data[row*p.Cols:(row+1)*p.Cols])
		return
	}
	q := p.Q8
	groups := (p.Cols + q8Block - 1) / q8Block
	for g := 0; g < groups; g++ {
		scale := q.Scales[row*groups+g]
		for j := g * q8Block; j < min((g+1)*q8Block, p.Cols); j++ {
			dst[j] = float32(q.Values[row*p.Cols+j]) * scale
		}
	}
}

// linear computes X W^T; W is contiguous [out,in].
func linear(x []float32, w *Param, n int) []float32 {
	y := make([]float32, n*w.Rows)
	linearInto(y, x, w, n)
	return y
}
func linearInto(y, x []float32, w *Param, n int) {
	linearIntoWithExecutor(nil, y, x, w, n)
}

func addBias(x []float32, bias *Param) {
	if bias == nil {
		return
	}
	for i := range x {
		x[i] += bias.Data[i%len(bias.Data)]
	}
}

func linearIntoWithExecutor(e *kernelExecutor, y, x []float32, w *Param, n int) {
	if w.Q4 != nil {
		linearQ4Into(e, y, x, w, n)
		return
	}
	if w.Q8 != nil {
		linearQ8Into(e, y, x, w, n)
		return
	}
	in, out := w.Cols, w.Rows
	// Four sequence rows share each weight load. Disjoint 4x32 output tiles can
	// run on different goroutines, including for a single long hidden vector.
	outTiles := (out + 31) / 32
	tasks := ((n + 3) / 4) * outTiles
	e.parallel(tasks, n*in*out, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			r := (task / outTiles) * 4
			o0 := (task % outTiles) * 32
			o1 := min(o0+32, out)
			if r+4 <= n {
				x0, x1, x2, x3 := x[r*in:(r+1)*in], x[(r+1)*in:(r+2)*in], x[(r+2)*in:(r+3)*in], x[(r+3)*in:(r+4)*in]
				for o := o0; o < o1; o++ {
					wr := w.Data[o*in : (o+1)*in]
					var a, b, c, d float32
					for j, v := range wr {
						a += x0[j] * v
						b += x1[j] * v
						c += x2[j] * v
						d += x3[j] * v
					}
					y[r*out+o] = a
					y[(r+1)*out+o] = b
					y[(r+2)*out+o] = c
					y[(r+3)*out+o] = d
				}
			} else {
				for row := r; row < n; row++ {
					xr := x[row*in : (row+1)*in]
					for o := o0; o < o1; o++ {
						wr := w.Data[o*in : (o+1)*in]
						var sum float32
						for j, v := range xr {
							sum += v * wr[j]
						}
						y[row*out+o] = sum
					}
				}
			}
		}
	})
}

// Decode reads one byte per weight, with no full matrix dequantization.
// Four-row tiles reuse each integer conversion across prompt/session rows.
func linearQ8Into(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	if q8NeedsWideSum(x[:n*in]) {
		linearQ8WideInto(e, y, x, w, n)
		return
	}
	groups, tiles := (in+q8Block-1)/q8Block, (out+31)/32
	e.parallel(((n+3)/4)*tiles, n*in*out, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			r, o0 := (task/tiles)*4, (task%tiles)*32
			for o := o0; o < min(o0+32, out); o++ {
				wr := w.Q8.Values[o*in : (o+1)*in]
				scales := w.Q8.Scales[o*groups : (o+1)*groups]
				if r+4 <= n {
					x0, x1 := x[r*in:(r+1)*in], x[(r+1)*in:(r+2)*in]
					x2, x3 := x[(r+2)*in:(r+3)*in], x[(r+3)*in:(r+4)*in]
					var s0, s1, s2, s3 float32
					for g, scale := range scales {
						var a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 float32
						j, end := g*q8Block, min((g+1)*q8Block, in)
						for ; j+4 <= end; j += 4 {
							v0, v1, v2, v3 := float32(wr[j]), float32(wr[j+1]), float32(wr[j+2]), float32(wr[j+3])
							a0 += x0[j] * v0
							b0 += x0[j+1] * v1
							c0 += x0[j+2] * v2
							d0 += x0[j+3] * v3
							a1 += x1[j] * v0
							b1 += x1[j+1] * v1
							c1 += x1[j+2] * v2
							d1 += x1[j+3] * v3
							a2 += x2[j] * v0
							b2 += x2[j+1] * v1
							c2 += x2[j+2] * v2
							d2 += x2[j+3] * v3
							a3 += x3[j] * v0
							b3 += x3[j+1] * v1
							c3 += x3[j+2] * v2
							d3 += x3[j+3] * v3
						}
						p0, p1, p2, p3 := (a0+b0)+(c0+d0), (a1+b1)+(c1+d1), (a2+b2)+(c2+d2), (a3+b3)+(c3+d3)
						for ; j < end; j++ {
							v := float32(wr[j])
							p0 += x0[j] * v
							p1 += x1[j] * v
							p2 += x2[j] * v
							p3 += x3[j] * v
						}
						s0 += p0 * scale
						s1 += p1 * scale
						s2 += p2 * scale
						s3 += p3 * scale
					}
					y[r*out+o], y[(r+1)*out+o] = s0, s1
					y[(r+2)*out+o], y[(r+3)*out+o] = s2, s3
				} else {
					for row := r; row < min(r+4, n); row++ {
						xr := x[row*in : (row+1)*in]
						var sum float32
						for g, scale := range scales {
							j, end := g*q8Block, min((g+1)*q8Block, in)
							var a, b, c, d float32
							for ; j+4 <= end; j += 4 {
								a += xr[j] * float32(wr[j])
								b += xr[j+1] * float32(wr[j+1])
								c += xr[j+2] * float32(wr[j+2])
								d += xr[j+3] * float32(wr[j+3])
							}
							part := (a + b) + (c + d)
							for ; j < end; j++ {
								part += xr[j] * float32(wr[j])
							}
							sum += part * scale
						}
						y[row*out+o] = sum
					}
				}
			}
		}
	})
}

// Integer partial sums precede the block scale. A rare, very large activation
// can overflow them even when the scaled result is finite. The factor of two
// leaves rounding headroom; ordinary activations only pay one input scan.
func q8NeedsWideSum(x []float32) bool {
	const limit = float32(math.MaxFloat32 / (2 * q8Block * 127))
	for _, v := range x {
		if v > limit || v < -limit {
			return true
		}
	}
	return false
}

func linearQ8WideInto(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	groups := (in + q8Block - 1) / q8Block
	for row := 0; row < n; row++ {
		xr, yr := x[row*in:(row+1)*in], y[row*out:(row+1)*out]
		if !q8NeedsWideSum(xr) {
			// Select per row, so a neighbouring extreme input never changes
			// the reduction order or result of an ordinary row in the batch.
			linearQ8Into(e, yr, xr, w, 1)
			continue
		}
		e.parallel(out, in*out, func(lo, hi int) {
			for o := lo; o < hi; o++ {
				wr := w.Q8.Values[o*in : (o+1)*in]
				var sum float64
				for g := 0; g < groups; g++ {
					var part float64
					for j := g * q8Block; j < min((g+1)*q8Block, in); j++ {
						part += float64(xr[j]) * float64(wr[j])
					}
					sum += part * float64(w.Q8.Scales[o*groups+g])
				}
				yr[o] = float32(sum)
			}
		})
	}
}

// Packed pairs are decoded once per four input columns and reused by up to
// four rows. Both paths use the same reduction order across batching/workers.
func linearQ4Into(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	if q8NeedsWideSum(x[:n*in]) {
		linearQ4WideInto(e, y, x, w, n)
		return
	}

	groups, tiles := (in+q8Block-1)/q8Block, (out+31)/32
	e.parallel(((n+3)/4)*tiles, n*in*out, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			r, o0 := (task/tiles)*4, (task%tiles)*32
			for o := o0; o < min(o0+32, out); o++ {
				wr := w.Q4.Values[o*((in+1)/2) : (o+1)*((in+1)/2)]
				scales := w.Q4.Scales[o*groups : (o+1)*groups]
				if r+4 <= n {
					x0, x1 := x[r*in:(r+1)*in], x[(r+1)*in:(r+2)*in]
					x2, x3 := x[(r+2)*in:(r+3)*in], x[(r+3)*in:(r+4)*in]
					var s0, s1, s2, s3 float32
					for g, scale := range scales {
						var a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3 float32
						j, end := g*q8Block, min((g+1)*q8Block, in)
						for ; j+4 <= end; j += 4 {
							v0, v1, v2, v3 := q4Pairs[wr[j/2]][0], q4Pairs[wr[j/2]][1], q4Pairs[wr[j/2+1]][0], q4Pairs[wr[j/2+1]][1]
							a0 += x0[j] * v0
							b0 += x0[j+1] * v1
							c0 += x0[j+2] * v2
							d0 += x0[j+3] * v3
							a1 += x1[j] * v0
							b1 += x1[j+1] * v1
							c1 += x1[j+2] * v2
							d1 += x1[j+3] * v3
							a2 += x2[j] * v0
							b2 += x2[j+1] * v1
							c2 += x2[j+2] * v2
							d2 += x2[j+3] * v3
							a3 += x3[j] * v0
							b3 += x3[j+1] * v1
							c3 += x3[j+2] * v2
							d3 += x3[j+3] * v3
						}
						p0, p1, p2, p3 := (a0+b0)+(c0+d0), (a1+b1)+(c1+d1), (a2+b2)+(c2+d2), (a3+b3)+(c3+d3)
						for ; j < end; j++ {
							v := q4Pairs[wr[(j)/2]][(j)%2]
							p0 += x0[j] * v
							p1 += x1[j] * v
							p2 += x2[j] * v
							p3 += x3[j] * v
						}
						s0 += p0 * scale
						s1 += p1 * scale
						s2 += p2 * scale
						s3 += p3 * scale
					}
					y[r*out+o], y[(r+1)*out+o] = s0, s1
					y[(r+2)*out+o], y[(r+3)*out+o] = s2, s3
				} else {
					for row := r; row < min(r+4, n); row++ {
						xr := x[row*in : (row+1)*in]
						var sum float32
						for g, scale := range scales {
							j, end := g*q8Block, min((g+1)*q8Block, in)
							var a, b, c, d float32
							for ; j+4 <= end; j += 4 {
								v0, v1 := q4Pairs[wr[j/2]][0], q4Pairs[wr[j/2]][1]
								v2, v3 := q4Pairs[wr[j/2+1]][0], q4Pairs[wr[j/2+1]][1]
								a += xr[j] * v0
								b += xr[j+1] * v1
								c += xr[j+2] * v2
								d += xr[j+3] * v3
							}
							part := (a + b) + (c + d)
							for ; j < end; j++ {
								part += xr[j] * q4Pairs[wr[(j)/2]][(j)%2]
							}
							sum += part * scale
						}
						y[row*out+o] = sum
					}
				}
			}
		}
	})
}

func linearQ4WideInto(e *kernelExecutor, y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	groups := (in + q8Block - 1) / q8Block
	for row := 0; row < n; row++ {
		xr, yr := x[row*in:(row+1)*in], y[row*out:(row+1)*out]
		if !q8NeedsWideSum(xr) {
			// Select per row, so a neighbouring extreme input never changes
			// the reduction order or result of an ordinary row in the batch.
			linearQ4Into(e, yr, xr, w, 1)
			continue
		}
		e.parallel(out, in*out, func(lo, hi int) {
			for o := lo; o < hi; o++ {
				wr := w.Q4.Values[o*((in+1)/2) : (o+1)*((in+1)/2)]
				var sum float64
				for g := 0; g < groups; g++ {
					var part float64
					for j := g * q8Block; j < min((g+1)*q8Block, in); j++ {
						part += float64(xr[j]) * float64(q4Pairs[wr[j/2]][j%2])
					}
					sum += part * float64(w.Q4.Scales[o*groups+g])
				}
				yr[o] = float32(sum)
			}
		})
	}
}

// Gradients accumulate, which is necessary for tied weights and microbatches.
func linearBackward(x []float32, w *Param, dy []float32, n int) []float32 {
	dx := make([]float32, n*w.Cols)
	linearBackwardInto(dx, x, w, dy, n)
	return dx
}

// dx must be disjoint from x, dy and parameter storage. Clear only scratch:
// parameter gradients retain contributions from tied weights and microbatches.
func linearBackwardInto(dx, x []float32, w *Param, dy []float32, n int) {
	in, out := w.Cols, w.Rows
	clear(dx)
	// Four independent rows share each weight load. Each output element still
	// accumulates o in ascending order, exactly as the scalar kernel did.
	parallel((n+3)/4, n*in*out, func(lo, hi int) {
		for tile := lo; tile < hi; tile++ {
			r := tile * 4
			if r+4 <= n {
				d0, d1, d2, d3 := dx[r*in:(r+1)*in], dx[(r+1)*in:(r+2)*in], dx[(r+2)*in:(r+3)*in], dx[(r+3)*in:(r+4)*in]
				y0, y1, y2, y3 := dy[r*out:(r+1)*out], dy[(r+1)*out:(r+2)*out], dy[(r+2)*out:(r+3)*out], dy[(r+3)*out:(r+4)*out]
				for o := 0; o < out; o++ {
					g0, g1, g2, g3 := y0[o], y1[o], y2[o], y3[o]
					for i, v := range w.Data[o*in : (o+1)*in] {
						d0[i] += g0 * v
						d1[i] += g1 * v
						d2[i] += g2 * v
						d3[i] += g3 * v
					}
				}
			} else {
				for row := r; row < n; row++ {
					d := dx[row*in : (row+1)*in]
					for o := 0; o < out; o++ {
						g := dy[row*out+o]
						for i, v := range w.Data[o*in : (o+1)*in] {
							d[i] += g * v
						}
					}
				}
			}
		}
	})
	// Weight-gradient tiles have disjoint owners and reuse each input load.
	// Existing gradients (tied embeddings / accumulation) are never reset here.
	parallel((out+3)/4, n*in*out, func(lo, hi int) {
		for tile := lo; tile < hi; tile++ {
			o := tile * 4
			if o+4 <= out {
				g0, g1, g2, g3 := w.Grad[o*in:(o+1)*in], w.Grad[(o+1)*in:(o+2)*in], w.Grad[(o+2)*in:(o+3)*in], w.Grad[(o+3)*in:(o+4)*in]
				for r := 0; r < n; r++ {
					y := dy[r*out : (r+1)*out]
					d0, d1, d2, d3 := y[o], y[o+1], y[o+2], y[o+3]
					for i, v := range x[r*in : (r+1)*in] {
						g0[i] += d0 * v
						g1[i] += d1 * v
						g2[i] += d2 * v
						g3[i] += d3 * v
					}
				}
			} else {
				for row := o; row < out; row++ {
					gw := w.Grad[row*in : (row+1)*in]
					for r := 0; r < n; r++ {
						g := dy[r*out+row]
						for i, v := range x[r*in : (r+1)*in] {
							gw[i] += g * v
						}
					}
				}
			}
		}
	})
}
func rms(x []float32, w *Param, n int, eps float64) ([]float32, []float32) {
	d := w.Cols
	y := make([]float32, len(x))
	inv := make([]float32, n)
	for r := 0; r < n; r++ {
		var ss float64
		for _, v := range x[r*d : (r+1)*d] {
			ss += float64(v) * float64(v)
		}
		a := float32(1 / math.Sqrt(ss/float64(d)+eps))
		inv[r] = a
		for j := 0; j < d; j++ {
			y[r*d+j] = x[r*d+j] * a * w.Data[j]
		}
	}
	return y, inv
}
func rmsBackward(x []float32, w *Param, inv, dy []float32, n int) []float32 {
	dx := make([]float32, len(x))
	rmsBackwardInto(dx, x, w, inv, dy, n)
	return dx
}
func rmsBackwardInto(dx, x []float32, w *Param, inv, dy []float32, n int) {
	d := w.Cols
	for r := 0; r < n; r++ {
		var dot float64
		for j := 0; j < d; j++ {
			i := r*d + j
			dot += float64(dy[i]) * float64(w.Data[j]) * float64(x[i])
			w.Grad[j] += dy[i] * x[i] * inv[r]
		}
		a := float64(inv[r])
		correction := dot * a * a / float64(d)
		for j := 0; j < d; j++ {
			i := r*d + j
			dx[i] = float32(a * (float64(dy[i])*float64(w.Data[j]) - float64(x[i])*correction))
		}
	}
}
func add(a, b []float32) []float32 {
	y := make([]float32, len(a))
	for i := range a {
		y[i] = a[i] + b[i]
	}
	return y
}
func addInPlace(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}
func swiglu(g, u []float32) []float32 {
	y := make([]float32, len(g))
	swigluInto(y, g, u)
	return y
}
func swigluInto(y, g, u []float32) {
	for i, v := range g {
		s := 1 / (1 + math.Exp(-float64(v)))
		y[i] = float32(float64(v)*s) * u[i]
	}
}
func swigluBackward(g, u, dy []float32) ([]float32, []float32) {
	dg, du := make([]float32, len(g)), make([]float32, len(g))
	swigluBackwardInto(dg, du, g, u, dy)
	return dg, du
}
func swigluBackwardInto(dg, du, g, u, dy []float32) {
	for i, v := range g {
		x := float64(v)
		s := 1 / (1 + math.Exp(-x))
		dg[i] = dy[i] * u[i] * float32(s+x*s*(1-s))
		du[i] = dy[i] * float32(x*s)
	}
}

// Model owns parameters; each Forward owns its tape and each decoder its cache.
// Immutable models are safe to share between inference requests.
type Block struct{ N1, Q, K, V, O, N2, Gate, Up, Down, QB, KB, VB, N1B, N2B, OB, GateB, DownB *Param }
type Model struct {
	output              *Param // Optional untied HF output head; native models use Emb.
	Config              Config
	Emb, Final          *Param
	Position, FinalBias *Param // GPT-2 absolute positions and affine LayerNorm.
	Blocks              []Block
	Params              []*Param
	Cos, Sin            []float32
	Revision            uint64
	decodePool          sync.Pool
	hf                  *hfConfig // Non-nil marks imported, inference-only model semantics.
}

func (m *Model) parameterCount() int64 {
	if m.hf != nil {
		return hfParameterCount(m.Config, *m.hf)
	}
	return m.Config.ParameterCount()
}

func loadHFModel(ctx context.Context, dir string, limit int, mib int64) (*Model, HFPlan, error) {
	return loadHFModelFormat(ctx, dir, limit, mib, "f32")
}

func loadHFModelFormat(ctx context.Context, dir string, limit int, mib int64, format string) (*Model, HFPlan, error) {
	if err := ctx.Err(); err != nil {
		return nil, HFPlan{}, err
	}
	a, err := openHFFlagsFormat(dir, limit, mib, format)
	if err != nil {
		return nil, HFPlan{}, err
	}
	defer a.Close()
	if a.Config.ModelType == "gpt2" {
		m, err := loadGPT2(ctx, a)
		return m, a.Plan, err
	}
	m := allocateModelStorageFormat(a.Plan.Config, nil, format)
	m.hf = &a.Config
	makeExtra := func(name string, rows, cols int) *Param {
		p := newInferenceParamFormat(name, rows, cols, name == "output", format)
		m.Params = append(m.Params, p)
		return p
	}
	if a.Config.ModelType == "qwen2" {
		for i := range m.Blocks {
			b := &m.Blocks[i]
			b.QB = makeExtra("q_bias", 1, m.Config.Dim)
			b.KB = makeExtra("k_bias", 1, m.Config.KVDim())
			b.VB = makeExtra("v_bias", 1, m.Config.KVDim())
		}
	}
	if !a.Config.Tied {
		m.output = makeExtra("output", m.Config.Vocab, m.Config.Dim)
	}
	// HF computes default RoPE frequencies and position products in FP32.
	hd := m.Config.Dim / m.Config.Heads
	for pos := 0; pos < m.Config.Context; pos++ {
		for j := 0; j < hd/2; j++ {
			inv := float32(1) / float32(math.Pow(m.Config.RopeTheta, float64(float32(2*j)/float32(hd))))
			angle := float64(float32(pos) * inv)
			m.Cos[pos*hd/2+j] = float32(math.Cos(angle))
			m.Sin[pos*hd/2+j] = float32(math.Sin(angle))
		}
	}
	for i, s := range hfProfileSpecs(m.Config, a.Config) {
		if err := a.readParam(ctx, s.Name, m.Params[i]); err != nil {
			return nil, a.Plan, err
		}
	}
	return m, a.Plan, nil
}

func loadGPT2(ctx context.Context, a *hfArchive) (*Model, error) {
	c := a.Plan.Config
	m := &Model{Config: c, hf: &a.Config}
	param := func(name string, rows, cols int, matrix bool) *Param {
		p := newInferenceParamFormat(name, rows, cols, matrix, a.Plan.WeightFormat)
		m.Params = append(m.Params, p)
		return p
	}
	// All names, dimensions and the complete resident storage budget have
	// already been validated by openHFArchive.
	d := c.Dim
	m.Emb = param("wte.weight", c.Vocab, d, true)
	m.Position = param("wpe.weight", a.Config.Context, d, false)
	m.Final = param("ln_f.weight", 1, d, false)
	m.FinalBias = param("ln_f.bias", 1, d, false)
	for _, p := range m.Params {
		if err := a.readParam(ctx, p.Name, p); err != nil {
			return nil, err
		}
	}
	conv := func(name string, dst ...*Param) error {
		// HF Conv1D stores [in,out], while linearInto uses [out,in].
		cols, total := dst[0].Cols, 0
		for _, p := range dst {
			total += p.Rows
		}
		passes := 1
		if dst[0].quantized() {
			passes = 2
		}
		for pass := 0; pass < passes; pass++ {
			// Transposed input arrives column-first. One streaming pass finds
			// block maxima, the second writes integers; no transpose buffer.
			if err := a.readTensorValues(ctx, name, cols*total, func(i int, v float32) error {
				in, out := i/total, i%total
				for _, p := range dst {
					if out < p.Rows {
						if !p.quantized() {
							p.Data[out*cols+in] = v
						} else {
							g := out*((cols+q8Block-1)/q8Block) + in/q8Block
							if pass == 0 {
								p.quantScales()[g] = max(p.quantScales()[g], float32(math.Abs(float64(v))))
							} else {
								p.putQuantized(out, in, v, p.quantScales()[g])
							}
						}
						return nil
					}
					out -= p.Rows
				}
				return errors.New("GPT-2 projection destination mismatch")
			}); err != nil {
				return err
			}
			if passes == 2 && pass == 0 {
				for _, p := range dst {
					for g, amax := range p.quantScales() {
						p.quantScales()[g] = p.quantScale(amax)
					}
				}
			}
		}
		return nil
	}
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("h.%d.", i)
		b := Block{
			N1: param(p+"ln_1.weight", 1, d, false), N1B: param(p+"ln_1.bias", 1, d, false),
			Q: param(p+"q", d, d, true), K: param(p+"k", d, d, true), V: param(p+"v", d, d, true),
			QB: param(p+"qb", 1, d, false), KB: param(p+"kb", 1, d, false), VB: param(p+"vb", 1, d, false),
			O: param(p+"attn.c_proj.weight", d, d, true), OB: param(p+"attn.c_proj.bias", 1, d, false),
			N2: param(p+"ln_2.weight", 1, d, false), N2B: param(p+"ln_2.bias", 1, d, false),
			Gate: param(p+"mlp.c_fc.weight", c.Hidden, d, true), GateB: param(p+"mlp.c_fc.bias", 1, c.Hidden, false),
			Down: param(p+"mlp.c_proj.weight", d, c.Hidden, true), DownB: param(p+"mlp.c_proj.bias", 1, d, false),
		}
		for _, v := range []*Param{b.N1, b.N1B, b.OB, b.N2, b.N2B, b.GateB, b.DownB} {
			if err := a.readTensor(ctx, v.Name, v.Data); err != nil {
				return nil, err
			}
		}
		if err := conv(p+"attn.c_attn.weight", b.Q, b.K, b.V); err != nil {
			return nil, err
		}
		if err := a.readTensorValues(ctx, p+"attn.c_attn.bias", 3*d, func(j int, v float32) error {
			[]*Param{b.QB, b.KB, b.VB}[j/d].Data[j%d] = v
			return nil
		}); err != nil {
			return nil, err
		}
		for _, v := range []*Param{b.O, b.Gate, b.Down} {
			if err := conv(v.Name, v); err != nil {
				return nil, err
			}
		}
		mask := p + "attn.bias"
		if t, ok := a.tensors[mask]; ok {
			if err := a.readTensorValues(ctx, mask, t.count, func(j int, v float32) error {
				want := float32(0)
				if j%a.Config.Context <= j/a.Config.Context {
					want = 1
				}
				if v != want {
					return errors.New("invalid GPT-2 legacy causal mask values")
				}
				return nil
			}); err != nil {
				return nil, err
			}
		}
		m.Blocks = append(m.Blocks, b)
	}
	return m, ctx.Err()
}

func layerNormInto(dst, x []float32, weight, bias *Param, rows int, eps float64) {
	d := weight.Cols
	for row := 0; row < rows; row++ {
		start := row * d
		var mean, variance float64
		for _, v := range x[start : start+d] {
			mean += float64(v)
		}
		mean /= float64(d)
		for _, v := range x[start : start+d] {
			z := float64(v) - mean
			variance += z * z
		}
		inv := 1 / math.Sqrt(variance/float64(d)+eps)
		for j := 0; j < d; j++ {
			dst[start+j] = float32((float64(x[start+j])-mean)*inv)*weight.Data[j] + bias.Data[j]
		}
	}
}

func inferenceNorm(dst, x []float32, weight, bias *Param, rows int, eps float64) {
	if bias != nil {
		layerNormInto(dst, x, weight, bias, rows, eps)
	} else {
		rmsInto(dst, x, weight, rows, eps)
	}
}

func NewModel(c Config, seed uint64) (*Model, error) { r := &RNG{seed}; return allocateModel(c, r) }
func allocateModel(c Config, rng *RNG) (*Model, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return allocateModelStorage(c, rng), nil
}

// Callers must validate dimensions and their native or HF allocation budget.
func allocateModelStorage(c Config, rng *RNG) *Model {
	return allocateModelStorageFormat(c, rng, "f32")
}

func allocateModelStorageFormat(c Config, rng *RNG, format string) *Model {
	m := &Model{Config: c}
	makeParam := func(name string, rows, cols int, norm bool, scale float64) *Param {
		p := newInferenceParamFormat(name, rows, cols, !norm, format)
		p.Decay = !norm
		if rng != nil {
			for i := range p.Data {
				if norm {
					p.Data[i] = 1
				} else {
					p.Data[i] = float32(rng.Normal() * scale)
				}
			}
		}
		m.Params = append(m.Params, p)
		return p
	}
	d, k, h := c.Dim, c.KVDim(), c.Hidden
	m.Emb = makeParam("token_embedding", c.Vocab, d, false, .02)
	for l := 0; l < c.Layers; l++ {
		prefix := fmt.Sprintf("layers.%d.", l)
		b := Block{}
		b.N1 = makeParam(prefix+"attention_norm", 1, d, true, 0)
		b.Q = makeParam(prefix+"q", d, d, false, 1/math.Sqrt(float64(d)))
		b.K = makeParam(prefix+"k", k, d, false, 1/math.Sqrt(float64(d)))
		b.V = makeParam(prefix+"v", k, d, false, 1/math.Sqrt(float64(d)))
		b.O = makeParam(prefix+"o", d, d, false, 1/math.Sqrt(float64(2*c.Layers*d)))
		b.N2 = makeParam(prefix+"ffn_norm", 1, d, true, 0)
		b.Gate = makeParam(prefix+"gate", h, d, false, 1/math.Sqrt(float64(d)))
		b.Up = makeParam(prefix+"up", h, d, false, 1/math.Sqrt(float64(d)))
		b.Down = makeParam(prefix+"down", d, h, false, 1/math.Sqrt(float64(2*c.Layers*h)))
		m.Blocks = append(m.Blocks, b)
	}
	m.Final = makeParam("final_norm", 1, d, true, 0)
	hd := d / c.Heads
	m.Cos = make([]float32, c.Context*hd/2)
	m.Sin = make([]float32, len(m.Cos))
	for pos := 0; pos < c.Context; pos++ {
		for j := 0; j < hd/2; j++ {
			angle := float64(pos) / math.Pow(c.RopeTheta, float64(2*j)/float64(hd))
			i := pos*hd/2 + j
			m.Cos[i] = float32(math.Cos(angle))
			m.Sin[i] = float32(math.Sin(angle))
		}
	}
	return m
}
func (m *Model) ZeroGrad() {
	for _, p := range m.Params {
		if len(p.Grad) != len(p.Data) {
			p.Grad = make([]float32, len(p.Data))
		} else {
			clear(p.Grad)
		}
	}
}
func (m *Model) rope(x []float32, batch, seq, heads, offset int, inverse bool) {
	hd := m.Config.Dim / m.Config.Heads
	width := heads * hd
	for b := 0; b < batch; b++ {
		for t := 0; t < seq; t++ {
			for h := 0; h < heads; h++ {
				for j := 0; j < hd/2; j++ {
					i := (b*seq+t)*width + h*hd + 2*j
					zidx := i + 1
					if m.hf != nil {
						i = (b*seq+t)*width + h*hd + j
						zidx = i + hd/2
					}
					ri := (offset+t)*hd/2 + j
					c, s := m.Cos[ri], m.Sin[ri]
					if inverse {
						s = -s
					}
					a, z := x[i], x[zidx]
					x[i] = a*c - z*s
					x[zidx] = a*s + z*c
				}
			}
		}
	}
}

// Only log-normalizers survive forward. Attention probabilities are recomputed
// in backward, removing the B*H*T*T allocation without changing dense attention.
func attend(q, out, scores []float32, length int, kv func(int) ([]float32, []float32)) float64 {
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
func (m *Model) attention(q, k, v []float32, batch, seq int) ([]float32, []float64) {
	c := m.Config
	d, kd, hd := c.Dim, c.KVDim(), c.Dim/c.Heads
	group := c.Heads / c.KVHeads
	out := make([]float32, batch*seq*d)
	lse := make([]float64, batch*c.Heads*seq)
	parallel(batch*c.Heads, batch*c.Heads*seq*seq*hd, func(lo, hi int) {
		scores := make([]float32, seq)
		for bh := lo; bh < hi; bh++ {
			b, h := bh/c.Heads, bh%c.Heads
			kh := h / group
			kv := func(s int) ([]float32, []float32) { i := (b*seq+s)*kd + kh*hd; return k[i : i+hd], v[i : i+hd] }
			for t := 0; t < seq; t++ {
				qi := (b*seq+t)*d + h*hd
				lse[bh*seq+t] = attend(q[qi:qi+hd], out[qi:qi+hd], scores, t+1, kv)
			}
		}
	})
	return out, lse
}
func (m *Model) attentionBackward(q, k, v, a []float32, lse []float64, dy []float32, batch, seq int) ([]float32, []float32, []float32) {
	dq, dk, dv := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v))
	m.attentionBackwardInto(dq, dk, dv, q, k, v, a, lse, dy, batch, seq)
	return dq, dk, dv
}
func (m *Model) attentionBackwardInto(dq, dk, dv, q, k, v, a []float32, lse []float64, dy []float32, batch, seq int) {
	clear(dq)
	clear(dk)
	clear(dv)
	c := m.Config
	d, kd, hd := c.Dim, c.KVDim(), c.Dim/c.Heads
	group := c.Heads / c.KVHeads
	scale := float32(1 / math.Sqrt(float64(hd)))
	// Partition by KV head, not Q head: grouped queries share K/V gradients.
	parallel(batch*c.KVHeads, batch*c.Heads*seq*seq*hd, func(lo, hi int) {
		for bkh := lo; bkh < hi; bkh++ {
			b, kh := bkh/c.KVHeads, bkh%c.KVHeads
			for h := kh * group; h < (kh+1)*group; h++ {
				for t := 0; t < seq; t++ {
					qi := (b*seq+t)*d + h*hd
					var delta float32
					for j := 0; j < hd; j++ {
						delta += dy[qi+j] * a[qi+j]
					}
					for s := 0; s <= t; s++ {
						ki := (b*seq+s)*kd + kh*hd
						var dot, dp float32
						for j := 0; j < hd; j++ {
							dot += q[qi+j] * k[ki+j]
							dp += dy[qi+j] * v[ki+j]
						}
						p := float32(math.Exp(float64(dot*scale) - lse[(b*c.Heads+h)*seq+t]))
						g := p * (dp - delta) * scale
						for j := 0; j < hd; j++ {
							dq[qi+j] += g * k[ki+j]
							dk[ki+j] += g * q[qi+j]
							dv[ki+j] += p * dy[qi+j]
						}
					}
				}
			}
		}
	})
}

type LayerTape struct {
	X, N1, R1, Q, K, V, A, U, N2, R2, Gate, Up, H []float32
	LSE                                           []float64
}
type Tape struct {
	Model                *Model
	Revision             uint64
	IDs                  []int
	Batch, Seq           int
	Layers               []LayerTape
	X, Norm, Inv, Logits []float32
	Used                 bool
}

// Conservative FP32-equivalent bound for a Forward tape and allocating Backward.
// This is a managed activation guard, not an estimate of peak process RSS.
func activationElements(c Config, batch, seq int) int64 {
	return int64(batch)*int64(seq)*(int64(c.Layers)*(12*int64(c.Dim)+4*int64(c.KVDim())+6*int64(c.Hidden))+4*int64(c.Vocab)) + 2*int64(batch)*int64(c.Heads)*int64(seq)*int64(c.Layers)
}

// The serial training loop owns this fixed-shape scratch for its entire run.
// It never retains a tape or input lease and is not part of Model or TrainState.
// Buffers are disjoint; Into kernels finish all workers before reuse. Like
// parameter mutation, workspace use requires exclusive access by its trainer.
type backwardWorkspace struct {
	model                      *Model
	config                     Config
	batch, seq                 int
	loss                       []float32
	residual, projection, temp []float32
	dq, dh, dg, du, dk, dv     []float32
}

func backwardWorkspaceElements(c Config, batch, seq int) int64 {
	return int64(batch) * int64(seq) * (int64(c.Vocab) + 4*int64(c.Dim) + 3*int64(c.Hidden) + 2*int64(c.KVDim()))
}

func validateTrainingShape(c Config, batch, seq int) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if batch < 1 || batch > 4096 || seq < 1 || seq > c.Context {
		return errors.New("invalid workspace shape or context length")
	}
	// Keep the existing conservative guard and charge persistent scratch too.
	// Evaluation owns its workspace only within each validation invocation.
	// Its batch=1 payload is smaller than this activation estimate, so it also
	// fits while backward scratch is retained. Shapes near the old limit may fail.
	if activationElements(c, batch, seq)+backwardWorkspaceElements(c, batch, seq) > maxActivationElements {
		return errors.New("activation budget including backward workspace exceeded; reduce batch, sequence, or model size")
	}
	return nil
}

func newBackwardWorkspace(m *Model, batch, seq int) (*backwardWorkspace, error) {
	if m == nil {
		return nil, errors.New("workspace requires a model")
	}
	if m.hf != nil {
		return nil, errors.New("HF models are inference-only")
	}
	c := m.Config
	if err := validateTrainingShape(c, batch, seq); err != nil {
		return nil, err
	}
	n := batch * seq
	return &backwardWorkspace{
		model: m, config: c, batch: batch, seq: seq,
		loss: make([]float32, n*c.Vocab), residual: make([]float32, n*c.Dim),
		projection: make([]float32, n*c.Dim), temp: make([]float32, n*c.Dim), dq: make([]float32, n*c.Dim),
		dh: make([]float32, n*c.Hidden), dg: make([]float32, n*c.Hidden), du: make([]float32, n*c.Hidden),
		dk: make([]float32, n*c.KVDim()), dv: make([]float32, n*c.KVDim()),
	}, nil
}

func (w *backwardWorkspace) validate(m *Model, batch, seq int) error {
	if w == nil || w.model != m || w.config != m.Config || w.batch != batch || w.seq != seq {
		return errors.New("backward workspace owner, configuration, or shape mismatch")
	}
	n, c := batch*seq, w.config
	for _, buffer := range []struct {
		data []float32
		size int
	}{
		{w.loss, n * c.Vocab}, {w.residual, n * c.Dim}, {w.projection, n * c.Dim}, {w.temp, n * c.Dim}, {w.dq, n * c.Dim},
		{w.dh, n * c.Hidden}, {w.dg, n * c.Hidden}, {w.du, n * c.Hidden}, {w.dk, n * c.KVDim()}, {w.dv, n * c.KVDim()},
	} {
		if len(buffer.data) != buffer.size || cap(buffer.data) != buffer.size {
			return errors.New("backward workspace storage mismatch")
		}
	}
	return nil
}

func (m *Model) Forward(ids []int, batch, seq int) (*Tape, error) {
	if m.hf != nil {
		return nil, errors.New("HF models are inference-only; use PrefillBatch/DecodeBatch")
	}
	c := m.Config
	if batch < 1 || batch > 4096 || seq < 1 || seq > c.Context || len(ids) != batch*seq {
		return nil, errors.New("invalid input shape or context length")
	}
	n := batch * seq
	if activationElements(c, batch, seq) > maxActivationElements {
		return nil, errors.New("activation budget exceeded; reduce batch, sequence, or model size")
	}
	x := make([]float32, n*c.Dim)
	for r, id := range ids {
		if id < 0 || id >= c.Vocab {
			return nil, fmt.Errorf("token %d outside vocabulary", id)
		}
		copy(x[r*c.Dim:(r+1)*c.Dim], m.Emb.Data[id*c.Dim:(id+1)*c.Dim])
	}
	tape := &Tape{Model: m, Revision: m.Revision, IDs: append([]int(nil), ids...), Batch: batch, Seq: seq}
	for _, b := range m.Blocks {
		l := LayerTape{X: x}
		l.N1, l.R1 = rms(x, b.N1, n, c.NormEps)
		l.Q, l.K, l.V = linear(l.N1, b.Q, n), linear(l.N1, b.K, n), linear(l.N1, b.V, n)
		m.rope(l.Q, batch, seq, c.Heads, 0, false)
		m.rope(l.K, batch, seq, c.KVHeads, 0, false)
		l.A, l.LSE = m.attention(l.Q, l.K, l.V, batch, seq)
		l.U = add(x, linear(l.A, b.O, n))
		l.N2, l.R2 = rms(l.U, b.N2, n, c.NormEps)
		l.Gate, l.Up = linear(l.N2, b.Gate, n), linear(l.N2, b.Up, n)
		l.H = swiglu(l.Gate, l.Up)
		x = add(l.U, linear(l.H, b.Down, n))
		tape.Layers = append(tape.Layers, l)
	}
	tape.X = x
	tape.Norm, tape.Inv = rms(x, m.Final, n, c.NormEps)
	tape.Logits = linear(tape.Norm, m.Emb, n)
	return tape, nil
}

// Each evaluation invocation exclusively owns its workspace. Unlike Forward,
// it retains neither a tape nor per-layer activations. Separate workspaces may
// read an immutable model concurrently; parameter mutation requires exclusion.
type evaluationWorkspace struct {
	model                                  *Model
	config                                 Config
	maxBatch, maxSeq                       int
	x, norm, q, k, v, attention            []float32
	gate, up, hidden, temp, logits, scores []float32
}

func evaluationWorkspaceElements(c Config, batch, seq int) int64 {
	return int64(batch) * int64(seq) * (5*int64(c.Dim) + 2*int64(c.KVDim()) + 3*int64(c.Hidden) + int64(c.Vocab) + int64(c.Heads))
}

func newEvaluationWorkspace(m *Model, maxBatch, maxSeq int) (*evaluationWorkspace, error) {
	if m == nil {
		return nil, errors.New("evaluation workspace requires a model")
	}
	if m.hf != nil {
		return nil, errors.New("HF models require token-ID inference; native evaluation uses BOS framing")
	}
	c := m.Config
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if maxBatch < 1 || maxBatch > 4096 || maxSeq < 1 || maxSeq > c.Context {
		return nil, errors.New("invalid evaluation workspace shape or context length")
	}
	if evaluationWorkspaceElements(c, maxBatch, maxSeq) > maxActivationElements {
		return nil, errors.New("evaluation workspace budget exceeded; reduce batch, sequence, or model size")
	}
	n := maxBatch * maxSeq
	return &evaluationWorkspace{
		model: m, config: c, maxBatch: maxBatch, maxSeq: maxSeq,
		x: make([]float32, n*c.Dim), norm: make([]float32, n*c.Dim), q: make([]float32, n*c.Dim),
		k: make([]float32, n*c.KVDim()), v: make([]float32, n*c.KVDim()), attention: make([]float32, n*c.Dim),
		gate: make([]float32, n*c.Hidden), up: make([]float32, n*c.Hidden), hidden: make([]float32, n*c.Hidden),
		temp: make([]float32, n*c.Dim), logits: make([]float32, n*c.Vocab), scores: make([]float32, n*c.Heads),
	}, nil
}

func (w *evaluationWorkspace) validate(m *Model, batch, seq int) error {
	if w == nil || m == nil || w.model != m || w.config != m.Config || w.maxBatch < 1 || w.maxBatch > 4096 || w.maxSeq < 1 || w.maxSeq > m.Config.Context || batch < 1 || batch > w.maxBatch || seq < 1 || seq > w.maxSeq {
		return errors.New("evaluation workspace owner, configuration, or shape mismatch")
	}
	n, c := w.maxBatch*w.maxSeq, w.config
	for _, buffer := range []struct {
		data []float32
		size int
	}{
		{w.x, n * c.Dim}, {w.norm, n * c.Dim}, {w.q, n * c.Dim}, {w.attention, n * c.Dim}, {w.temp, n * c.Dim},
		{w.k, n * c.KVDim()}, {w.v, n * c.KVDim()}, {w.gate, n * c.Hidden}, {w.up, n * c.Hidden}, {w.hidden, n * c.Hidden},
		{w.logits, n * c.Vocab}, {w.scores, n * c.Heads},
	} {
		if len(buffer.data) != buffer.size || cap(buffer.data) != buffer.size {
			return errors.New("evaluation workspace storage mismatch")
		}
	}
	return nil
}

// The result borrows w.logits until the next call using w. Callers score it
// synchronously or copy it before reusing the workspace. No input is retained.
func (m *Model) forwardEvaluation(ctx context.Context, ids []int, batch, seq int, w *evaluationWorkspace) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("nil evaluation context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := w.validate(m, batch, seq); err != nil {
		return nil, err
	}
	if len(ids) != batch*seq {
		return nil, errors.New("invalid evaluation input shape")
	}
	c := m.Config
	for _, id := range ids {
		if id < 0 || id >= c.Vocab {
			return nil, fmt.Errorf("token %d outside vocabulary", id)
		}
	}
	n, d, kd, hd := batch*seq, c.Dim, c.KVDim(), c.Dim/c.Heads
	x, norm, q := w.x[:n*d], w.norm[:n*d], w.q[:n*d]
	k, v, a := w.k[:n*kd], w.v[:n*kd], w.attention[:n*d]
	gate, up, hidden := w.gate[:n*c.Hidden], w.up[:n*c.Hidden], w.hidden[:n*c.Hidden]
	temp, logits := w.temp[:n*d], w.logits[:n*c.Vocab:n*c.Vocab]
	for r, id := range ids {
		copy(x[r*d:(r+1)*d], m.Emb.Data[id*d:(id+1)*d])
	}
	group := c.Heads / c.KVHeads
	for _, b := range m.Blocks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		rmsInto(norm, x, b.N1, n, c.NormEps)
		linearInto(q, norm, b.Q, n)
		linearInto(k, norm, b.K, n)
		linearInto(v, norm, b.V, n)
		m.rope(q, batch, seq, c.Heads, 0, false)
		m.rope(k, batch, seq, c.KVHeads, 0, false)
		parallel(batch*c.Heads, batch*c.Heads*seq*seq*hd, func(lo, hi int) {
			for bh := lo; bh < hi; bh++ {
				batchIndex, h := bh/c.Heads, bh%c.Heads
				kh := h / group
				// Each head owns a fixed score region, regardless of worker count
				// or the shorter sequence length of this particular window.
				scores := w.scores[bh*w.maxSeq : bh*w.maxSeq+seq]
				kv := func(s int) ([]float32, []float32) {
					i := (batchIndex*seq+s)*kd + kh*hd
					return k[i : i+hd], v[i : i+hd]
				}
				for t := 0; t < seq; t++ {
					qi := (batchIndex*seq+t)*d + h*hd
					attend(q[qi:qi+hd], a[qi:qi+hd], scores, t+1, kv)
				}
			}
		})
		linearInto(temp, a, b.O, n)
		addInPlace(x, temp)
		rmsInto(norm, x, b.N2, n, c.NormEps)
		linearInto(gate, norm, b.Gate, n)
		linearInto(up, norm, b.Up, n)
		swigluInto(hidden, gate, up)
		linearInto(temp, hidden, b.Down, n)
		addInPlace(x, temp)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	rmsInto(norm, x, m.Final, n, c.NormEps)
	linearInto(logits, norm, m.Emb, n)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return logits, nil
}

// CrossEntropy uses log-sum-exp, including for extremely unlikely targets.
// Returned derivatives are scaled means, allowing exact microbatch weighting.
func CrossEntropy(logits []float32, targets []int, vocab int, scale float64, gradient bool) (float64, []float32, error) {
	if len(targets) == 0 || vocab < 1 || len(logits) != len(targets)*vocab || !finite(scale) {
		return 0, nil, errors.New("invalid loss input")
	}
	var grad []float32
	if gradient {
		grad = make([]float32, len(logits))
	}
	loss, err := crossEntropyInto(grad, logits, targets, vocab, scale)
	if err != nil {
		return 0, nil, err
	}
	return loss, grad, nil
}

// A nil destination scores only; a non-nil destination is overwritten in full.
func crossEntropyInto(grad, logits []float32, targets []int, vocab int, scale float64) (float64, error) {
	if len(targets) == 0 || vocab < 1 || len(logits) != len(targets)*vocab || !finite(scale) || (grad != nil && len(grad) != len(logits)) {
		return 0, errors.New("invalid loss input")
	}
	var loss float64
	for r, target := range targets {
		if target < 0 || target >= vocab {
			return 0, errors.New("target outside vocabulary")
		}
		row := logits[r*vocab : (r+1)*vocab]
		maxLogit := math.Inf(-1)
		for _, v := range row {
			x := float64(v)
			if !finite(x) {
				return 0, errors.New("nonfinite logit")
			}
			if x > maxLogit {
				maxLogit = x
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v) - maxLogit)
		}
		loss += math.Log(sum) + maxLogit - float64(row[target])
		if grad != nil {
			for j, v := range row {
				g := math.Exp(float64(v)-maxLogit) / sum
				if j == target {
					g--
				}
				grad[r*vocab+j] = float32(g * scale / float64(len(targets)))
			}
		}
	}
	return loss / float64(len(targets)), nil
}
func (m *Model) Backward(t *Tape, targets []int, scale float64) (float64, error) {
	if m.hf != nil {
		return 0, errors.New("HF models are inference-only")
	}
	return m.backward(t, targets, scale, nil)
}
func (m *Model) backward(t *Tape, targets []int, scale float64, w *backwardWorkspace) (float64, error) {
	if t == nil || t.Model != m || t.Revision != m.Revision || t.Used {
		return 0, errors.New("tape is foreign, stale, or already consumed")
	}
	for _, p := range m.Params {
		if len(p.Grad) != len(p.Data) {
			return 0, errors.New("call ZeroGrad before Backward")
		}
	}
	var loss float64
	var dy []float32
	var err error
	if w == nil {
		loss, dy, err = CrossEntropy(t.Logits, targets, m.Config.Vocab, scale, true)
	} else {
		if err = w.validate(m, t.Batch, t.Seq); err != nil {
			return 0, err
		}
		loss, err = crossEntropyInto(w.loss, t.Logits, targets, m.Config.Vocab, scale)
	}
	if err != nil {
		return 0, err
	}
	t.Used = true
	if w != nil {
		m.backwardInto(t, w)
		return loss, nil
	}
	n := t.Batch * t.Seq
	dx := linearBackward(t.Norm, m.Emb, dy, n)
	dx = rmsBackward(t.X, m.Final, t.Inv, dx, n)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		b, l := m.Blocks[i], t.Layers[i]
		dh := linearBackward(l.H, b.Down, dx, n)
		dg, du := swigluBackward(l.Gate, l.Up, dh)
		dn2 := linearBackward(l.N2, b.Gate, dg, n)
		addInPlace(dn2, linearBackward(l.N2, b.Up, du, n))
		addInPlace(dx, rmsBackward(l.U, b.N2, l.R2, dn2, n))
		da := linearBackward(l.A, b.O, dx, n)
		dq, dk, dv := m.attentionBackward(l.Q, l.K, l.V, l.A, l.LSE, da, t.Batch, t.Seq)
		m.rope(dq, t.Batch, t.Seq, m.Config.Heads, 0, true)
		m.rope(dk, t.Batch, t.Seq, m.Config.KVHeads, 0, true)
		dn1 := linearBackward(l.N1, b.Q, dq, n)
		addInPlace(dn1, linearBackward(l.N1, b.K, dk, n))
		addInPlace(dn1, linearBackward(l.N1, b.V, dv, n))
		addInPlace(dx, rmsBackward(l.X, b.N1, l.R1, dn1, n))
	}
	d := m.Config.Dim
	for r, id := range t.IDs {
		for j := 0; j < d; j++ {
			m.Emb.Grad[id*d+j] += dx[r*d+j]
		}
	}
	return loss, nil
}

func (m *Model) backwardInto(t *Tape, w *backwardWorkspace) {
	n := t.Batch * t.Seq
	linearBackwardInto(w.temp, t.Norm, m.Emb, w.loss, n)
	rmsBackwardInto(w.residual, t.X, m.Final, t.Inv, w.temp, n)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		b, l := m.Blocks[i], t.Layers[i]
		// The residual gradient lives across the whole block. The projection
		// sum survives each temporary contribution and its following RMSNorm.
		linearBackwardInto(w.dh, l.H, b.Down, w.residual, n)
		swigluBackwardInto(w.dg, w.du, l.Gate, l.Up, w.dh)
		linearBackwardInto(w.projection, l.N2, b.Gate, w.dg, n)
		linearBackwardInto(w.temp, l.N2, b.Up, w.du, n)
		addInPlace(w.projection, w.temp)
		rmsBackwardInto(w.temp, l.U, b.N2, l.R2, w.projection, n)
		addInPlace(w.residual, w.temp)
		linearBackwardInto(w.temp, l.A, b.O, w.residual, n)
		m.attentionBackwardInto(w.dq, w.dk, w.dv, l.Q, l.K, l.V, l.A, l.LSE, w.temp, t.Batch, t.Seq)
		m.rope(w.dq, t.Batch, t.Seq, m.Config.Heads, 0, true)
		m.rope(w.dk, t.Batch, t.Seq, m.Config.KVHeads, 0, true)
		// Attention has finished reading temp; K/V projections may now reuse it.
		linearBackwardInto(w.projection, l.N1, b.Q, w.dq, n)
		linearBackwardInto(w.temp, l.N1, b.K, w.dk, n)
		addInPlace(w.projection, w.temp)
		linearBackwardInto(w.temp, l.N1, b.V, w.dv, n)
		addInPlace(w.projection, w.temp)
		rmsBackwardInto(w.temp, l.X, b.N1, l.R1, w.projection, n)
		addInPlace(w.residual, w.temp)
	}
	d := m.Config.Dim
	for r, id := range t.IDs {
		for j := 0; j < d; j++ {
			m.Emb.Grad[id*d+j] += w.residual[r*d+j]
		}
	}
}

// KV pages are allocated lazily. The inference actor alone owns its page pool;
// reservation of the full request budget prevents mid-generation pool deadlock.
const pageTokens = 32

type kvPage struct{ Data []float32 }
type pagePool struct {
	PageElements, Limit, Reserved, Allocated int
	Free                                     []*kvPage
}

func (p *pagePool) acquire() *kvPage {
	if len(p.Free) > 0 {
		i := len(p.Free) - 1
		v := p.Free[i]
		p.Free = p.Free[:i]
		return v
	}
	if p.Allocated >= p.Limit {
		panic("KV reservation invariant violated")
	}
	p.Allocated++
	return &kvPage{make([]float32, p.PageElements)}
}

type KVCache struct {
	Model    *Model
	Revision uint64
	Pos      int
	Pages    []*kvPage
	pool     *pagePool
	reserved int
	released bool
}

func (m *Model) NewCache() *KVCache { return &KVCache{Model: m, Revision: m.Revision} }
func (c *KVCache) ensurePage() {
	c.ensurePages(c.Pos + 1)
}
func (c *KVCache) ensurePages(end int) {
	for len(c.Pages) < (end+pageTokens-1)/pageTokens {
		var p *kvPage
		if c.pool != nil {
			p = c.pool.acquire()
		} else {
			p = &kvPage{make([]float32, 2*pageTokens*c.Model.Config.KVDim()*c.Model.Config.Layers)}
		}
		c.Pages = append(c.Pages, p)
	}
}
func (c *KVCache) row(layer, pos int) ([]float32, []float32) {
	kd := c.Model.Config.KVDim()
	p := c.Pages[pos/pageTokens]
	base := layer*2*pageTokens*kd + (pos%pageTokens)*kd
	return p.Data[base : base+kd], p.Data[base+pageTokens*kd : base+pageTokens*kd+kd]
}
func (c *KVCache) Close() {
	if c.released {
		return
	}
	if c.pool != nil {
		c.pool.Free = append(c.pool.Free, c.Pages...)
		c.pool.Reserved -= c.reserved
	}
	c.Pages = nil
	c.released = true
}

type decodePoolUse struct{ pages, reserved int }
type decodeWorkspace struct {
	X, N, Q, K, V, A, G, U, H, Tmp, Logits, Scores []float32
	// Metadata has the same hard bounds as PrefillBatch. Only this workspace's
	// owner may use it; pointer-bearing fields are cleared after kernels join.
	rowCaches [1024]*KVCache
	positions [1024]int
	lastRows  [256]int
	pools     map[*pagePool]decodePoolUse
	seen      map[*KVCache]bool
}

func (w *decodeWorkspace) clearReferences() {
	clear(w.rowCaches[:])
	clear(w.pools)
	clear(w.seen)
}

func (w *decodeWorkspace) floatBytes() int64 {
	var elements int64
	for _, x := range [][]float32{w.X, w.N, w.Q, w.K, w.V, w.A, w.G, w.U, w.H, w.Tmp, w.Logits, w.Scores} {
		elements += int64(cap(x))
	}
	return elements * 4
}

type inferenceArray struct {
	x *[]float32
	n int
}

// Account for both current needs and retained capacities across mixed shapes.
// Extra reserves caller-owned output, even when an internal consumer borrows it.
func resizeInferenceArrays(arrays []inferenceArray, extra, limit int64) error {
	required, retained := extra, extra
	for _, a := range arrays {
		required += int64(a.n)
		retained += int64(max(cap(*a.x), a.n))
	}
	if required > limit {
		return errors.New("inference workspace budget exceeded; reduce prefill tokens or batch size")
	}
	if retained > limit {
		for _, a := range arrays {
			*a.x = nil
		}
	}
	for _, a := range arrays {
		*a.x = resize(*a.x, a.n)
	}
	return nil
}

func resize(x []float32, n int) []float32 {
	if cap(x) < n {
		return make([]float32, n)
	}
	return x[:n]
}
func rmsInto(y, x []float32, w *Param, n int, eps float64) {
	d := w.Cols
	for r := 0; r < n; r++ {
		var ss float64
		for _, v := range x[r*d : (r+1)*d] {
			ss += float64(v) * float64(v)
		}
		a := float32(1 / math.Sqrt(ss/float64(d)+eps))
		for j := 0; j < d; j++ {
			y[r*d+j] = x[r*d+j] * a * w.Data[j]
		}
	}
}

// DecodeBatch shares all matrix projections across independent sessions. Each
// row can have a different position; pages and RoPE offsets remain session-local.
func (m *Model) DecodeBatch(ctx context.Context, caches []*KVCache, tokens []int) ([]float32, error) {
	if len(tokens) != len(caches) || len(caches) < 1 || len(caches) > 256 {
		return nil, errors.New("invalid decode batch (1..256)")
	}
	chunks := make([][]int, len(tokens))
	for i := range tokens {
		chunks[i] = tokens[i : i+1]
	}
	return m.PrefillBatch(ctx, caches, chunks)
}

// PrefillBatch appends one nonempty, causal chunk per session and returns only
// the last-token logits for each session. Flattened rows share projection work;
// attention reads each row's own prefix, including earlier rows in its chunk.
// The caller must exclusively own all caches and their pools for this call.
// Cancellation may populate future KV rows, but never commits cache positions.
func (m *Model) PrefillBatch(ctx context.Context, caches []*KVCache, chunks [][]int) ([]float32, error) {
	return m.prefillBatch(ctx, caches, chunks, nil)
}

func (m *Model) prefillBatch(ctx context.Context, caches []*KVCache, chunks [][]int, executor *kernelExecutor) ([]float32, error) {
	raw := m.decodePool.Get()
	var w *decodeWorkspace
	if raw == nil {
		w = &decodeWorkspace{}
	} else {
		w = raw.(*decodeWorkspace)
	}
	defer m.decodePool.Put(w)
	logits, err := m.prefillBorrowed(ctx, caches, chunks, executor, w)
	if err != nil {
		return nil, err
	}
	// Public callers keep their result across subsequent calls and concurrent
	// engines sharing this model. Only the actor may consume borrowed logits.
	return append([]float32(nil), logits...), nil
}

// The caller exclusively owns w and consumes the result before its next use.
// No callback, response or request may retain the returned logits.
func (m *Model) prefillBorrowed(ctx context.Context, caches []*KVCache, chunks [][]int, executor *kernelExecutor, w *decodeWorkspace) ([]float32, error) {
	c := m.Config
	sessions := len(caches)
	if sessions < 1 || sessions > 256 || len(chunks) != sessions {
		return nil, errors.New("invalid prefill batch (1..256 sessions)")
	}
	if ctx == nil {
		return nil, errors.New("nil inference context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	defer w.clearReferences()
	if w.pools == nil {
		w.pools = make(map[*pagePool]decodePoolUse)
	}
	if w.seen == nil {
		w.seen = make(map[*KVCache]bool, sessions)
	}
	pools, seen := w.pools, w.seen
	n, maxPos := 0, 0
	pageElements := 2 * pageTokens * c.KVDim() * c.Layers
	for i, cache := range caches {
		if cache == nil || cache.Model != m || cache.Revision != m.Revision || cache.released || seen[cache] {
			return nil, errors.New("foreign, stale, released, or duplicate KV cache")
		}
		chunk := chunks[i]
		if len(chunk) < 1 || len(chunk) > 1024-n {
			return nil, errors.New("prefill requires nonempty chunks totaling at most 1024 tokens")
		}
		if cache.Pos < 0 || cache.Pos > c.Context || len(chunk) > c.Context-cache.Pos {
			return nil, errors.New("KV cache context exhausted or invalid position")
		}
		for _, token := range chunk {
			if token < 0 || token >= c.Vocab {
				return nil, errors.New("token outside vocabulary")
			}
		}
		end := cache.Pos + len(chunk)
		pages := (end + pageTokens - 1) / pageTokens
		if len(cache.Pages) < (cache.Pos+pageTokens-1)/pageTokens || len(cache.Pages) > (c.Context+pageTokens-1)/pageTokens {
			return nil, errors.New("invalid KV page coverage")
		}
		for _, page := range cache.Pages {
			if page == nil || len(page.Data) != pageElements {
				return nil, errors.New("invalid KV page storage")
			}
		}
		if p := cache.pool; p != nil {
			if p.PageElements != pageElements || p.Limit < 1 || p.Allocated < 0 || p.Allocated > p.Limit || len(p.Free) > p.Allocated || p.Reserved > p.Limit || cache.reserved < 1 || pages > cache.reserved || len(cache.Pages) > cache.reserved {
				return nil, errors.New("invalid KV page reservation")
			}
			use := pools[p]
			use.pages += max(0, pages-len(cache.Pages))
			use.reserved += cache.reserved
			pools[p] = use
		}
		seen[cache] = true
		n += len(chunk)
		maxPos = max(maxPos, end)
	}
	for p, use := range pools {
		if use.reserved > p.Reserved || use.pages > len(p.Free)+p.Limit-p.Allocated {
			return nil, errors.New("insufficient KV page reservation")
		}
		// Validate only the free pages this call will actually acquire.
		for j := 0; j < min(use.pages, len(p.Free)); j++ {
			page := p.Free[len(p.Free)-1-j]
			if page == nil || len(page.Data) != pageElements {
				return nil, errors.New("invalid free KV page storage")
			}
		}
	}
	d, kd, hd := c.Dim, c.KVDim(), c.Dim/c.Heads
	upElements := n * c.Hidden
	if m.Position != nil {
		upElements = 0
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	group := c.Heads / c.KVHeads
	// Reserve the public result copy in both paths to keep admission limits
	// unchanged. Metadata is separately bounded by 256 sessions / 1024 rows.
	arrays := []inferenceArray{
		{&w.X, n * d}, {&w.N, n * d}, {&w.Q, n * d},
		{&w.K, n * kd}, {&w.V, n * kd}, {&w.A, n * d},
		{&w.Tmp, n * d}, {&w.G, n * c.Hidden},
		{&w.U, upElements}, {&w.H, n * c.Hidden},
		{&w.Logits, sessions * c.Vocab}, {&w.Scores, n * maxPos},
	}
	if err := resizeInferenceArrays(arrays, int64(sessions)*int64(c.Vocab), maxActivationElements); err != nil {
		return nil, err
	}
	// Every session owns consecutive rows, but positions include its prefix.
	rowCaches, positions, lastRows := w.rowCaches[:n], w.positions[:n], w.lastRows[:sessions]
	r := 0
	for i, cache := range caches {
		cache.ensurePages(cache.Pos + len(chunks[i]))
		for j, token := range chunks[i] {
			rowCaches[r], positions[r] = cache, cache.Pos+j
			m.Emb.copyRow(w.X[r*d:(r+1)*d], token)
			if m.Position != nil {
				addInPlace(w.X[r*d:(r+1)*d], m.Position.Data[positions[r]*d:(positions[r]+1)*d])
			}
			r++
		}
		lastRows[i] = r - 1
	}
	for l, b := range m.Blocks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		region := trace.StartRegion(ctx, "monolith.prefill_layer")
		inferenceNorm(w.N, w.X, b.N1, b.N1B, n, c.NormEps)
		linearIntoWithExecutor(executor, w.Q, w.N, b.Q, n)
		linearIntoWithExecutor(executor, w.K, w.N, b.K, n)
		linearIntoWithExecutor(executor, w.V, w.N, b.V, n)
		addBias(w.Q, b.QB)
		addBias(w.K, b.KB)
		addBias(w.V, b.VB)
		for i, cache := range rowCaches {
			if m.Position == nil {
				m.rope(w.Q[i*d:(i+1)*d], 1, 1, c.Heads, positions[i], false)
				m.rope(w.K[i*kd:(i+1)*kd], 1, 1, c.KVHeads, positions[i], false)
			}
			k, v := cache.row(l, positions[i])
			copy(k, w.K[i*kd:(i+1)*kd])
			copy(v, w.V[i*kd:(i+1)*kd])
		}
		executor.parallel(n, n*c.Heads*maxPos*hd, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				cache := rowCaches[i]
				scores := w.Scores[i*maxPos : (i+1)*maxPos]
				for h := 0; h < c.Heads; h++ {
					kh := h / group
					qi := i*d + h*hd
					kv := func(pos int) ([]float32, []float32) {
						k, v := cache.row(l, pos)
						return k[kh*hd : (kh+1)*hd], v[kh*hd : (kh+1)*hd]
					}
					attend(w.Q[qi:qi+hd], w.A[qi:qi+hd], scores, positions[i]+1, kv)
				}
			}
		})
		linearIntoWithExecutor(executor, w.Tmp, w.A, b.O, n)
		addBias(w.Tmp, b.OB)
		addInPlace(w.X, w.Tmp)
		inferenceNorm(w.N, w.X, b.N2, b.N2B, n, c.NormEps)
		linearIntoWithExecutor(executor, w.G, w.N, b.Gate, n)
		if m.Position != nil {
			addBias(w.G, b.GateB)
			for i, v := range w.G {
				x := float64(v)
				w.H[i] = float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
			}
		} else {
			linearIntoWithExecutor(executor, w.U, w.N, b.Up, n)
			for i, v := range w.G {
				w.H[i] = float32(float64(v)/(1+math.Exp(-float64(v)))) * w.U[i]
			}
		}
		linearIntoWithExecutor(executor, w.Tmp, w.H, b.Down, n)
		addBias(w.Tmp, b.DownB)
		addInPlace(w.X, w.Tmp)
		region.End()
	}
	// Gather only each session's final hidden row before the vocabulary head.
	for i, row := range lastRows {
		copy(w.Tmp[i*d:(i+1)*d], w.X[row*d:(row+1)*d])
	}
	inferenceNorm(w.N[:sessions*d], w.Tmp[:sessions*d], m.Final, m.FinalBias, sessions, c.NormEps)
	head := m.Emb
	if m.output != nil {
		head = m.output
	}
	linearIntoWithExecutor(executor, w.Logits, w.N[:sessions*d], head, sessions)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for i, cache := range caches {
		cache.Pos += len(chunks[i])
	}
	return w.Logits, nil
}
func (m *Model) Step(cache *KVCache, token int) ([]float32, error) {
	return m.DecodeBatch(context.Background(), []*KVCache{cache}, []int{token})
}

type TrainSpec struct {
	Steps       int     `json:"steps"`
	Batch       int     `json:"batch"`
	Seq         int     `json:"seq"`
	Accum       int     `json:"accum"`
	LR          float64 `json:"lr"`
	MinLR       float64 `json:"min_lr"`
	Warmup      int     `json:"warmup"`
	WeightDecay float64 `json:"weight_decay"`
	Beta1       float64 `json:"beta1"`
	Beta2       float64 `json:"beta2"`
	Eps         float64 `json:"eps"`
	Clip        float64 `json:"clip"`
	ValFraction float64 `json:"val_fraction"`
}

func defaultTrainSpec() TrainSpec {
	return TrainSpec{Steps: 200, Batch: 2, Seq: 64, Accum: 1, LR: .003, MinLR: .0003, Warmup: 10, WeightDecay: .01, Beta1: .9, Beta2: .999, Eps: 1e-8, Clip: 1, ValFraction: .1}
}
func (s TrainSpec) Validate(c Config) error {
	if s.Steps < 1 || s.Steps > 10_000_000 || s.Batch < 1 || s.Batch > 4096 || s.Seq < 1 || s.Seq > c.Context || s.Accum < 1 || s.Accum > 1024 || s.Warmup < 0 || s.Warmup > s.Steps {
		return errors.New("invalid steps, batch, sequence, accumulation, or warmup")
	}
	vals := []float64{s.LR, s.MinLR, s.WeightDecay, s.Beta1, s.Beta2, s.Eps, s.Clip, s.ValFraction}
	for _, v := range vals {
		if !finite(v) {
			return errors.New("training settings must be finite")
		}
	}
	if s.LR <= 0 || s.LR > 1 || s.MinLR < 0 || s.MinLR > s.LR || s.WeightDecay < 0 || s.WeightDecay > 1 || s.Beta1 < 0 || s.Beta1 >= 1 || s.Beta2 < 0 || s.Beta2 >= 1 || s.Eps <= 0 || s.Eps > 1 || s.Clip <= 0 || s.ValFraction <= 0 || s.ValFraction >= .5 {
		return errors.New("invalid optimizer or validation split settings")
	}
	return nil
}

// An invocation's major-array plan is computed before allocating model tensors,
// corpus token arrays, workspaces or prefetch leases. It is not a process/RSS cap:
// runtime/allocator overhead, metadata, source preparation, checkpoint/profile IO
// and garbage awaiting collection are outside this working-set estimate.
type trainingMemoryInput struct {
	Kind      string `json:"kind"`
	TextBytes int64  `json:"text_bytes,omitempty"`
	Shards    int    `json:"shards,omitempty"`
	Workers   int    `json:"workers,omitempty"`
	Depth     int    `json:"depth,omitempty"`
}
type trainingMemoryComponents struct {
	Weights                   int64 `json:"weights_bytes"`
	Gradients                 int64 `json:"gradients_bytes"`
	AdamMoments               int64 `json:"adam_moments_bytes"`
	RoPE                      int64 `json:"rope_bytes"`
	BackwardWorkspace         int64 `json:"backward_workspace_bytes"`
	CorpusAllowance           int64 `json:"corpus_allowance_bytes"`
	TrainingPrefetchAllowance int64 `json:"training_prefetch_allowance_bytes"`
	TrainingActivationBound   int64 `json:"training_activation_bound_bytes"`
	TrainingInputAllowance    int64 `json:"training_input_allowance_bytes"`
	EvaluationWorkspace       int64 `json:"evaluation_workspace_bytes"`
	EvaluationInputAllowance  int64 `json:"evaluation_input_allowance_bytes"`
}
type trainingMemoryPlan struct {
	Format          string                   `json:"format"`
	Scope           string                   `json:"scope"`
	Config          Config                   `json:"config"`
	Spec            TrainSpec                `json:"spec"`
	Input           trainingMemoryInput      `json:"input"`
	Components      trainingMemoryComponents `json:"components"`
	PersistentBytes int64                    `json:"persistent_bytes"`
	TrainingBytes   int64                    `json:"training_bytes"`
	EvaluationBytes int64                    `json:"evaluation_bytes"`
	RequiredBytes   int64                    `json:"required_bytes"`
	BudgetBytes     int64                    `json:"budget_bytes"`
	FitsBudget      bool                     `json:"fits_budget"`
}

func memoryBudgetBytes(mib int64) (int64, error) {
	if mib < 0 || mib > math.MaxInt64/(1<<20) {
		return 0, errors.New("memory-mib must be nonnegative and fit in int64 bytes")
	}
	return mib * (1 << 20), nil
}

func newTrainingMemoryPlan(c Config, s TrainSpec, input trainingMemoryInput, budget int64) (trainingMemoryPlan, error) {
	p := trainingMemoryPlan{Format: "monolith-training-memory-v1", Scope: "major-array-payload-and-input-staging", Config: c, Spec: s, Input: input, BudgetBytes: budget}
	// Validate scalar limits before multiplying. All subsequent products/sums fit
	// int64 under Config/TrainSpec/prefetch limits, even for rejected shapes.
	if err := validateTrainingShape(c, s.Batch, s.Seq); err != nil {
		return p, err
	}
	if err := s.Validate(c); err != nil {
		return p, err
	}
	if budget < 0 {
		return p, errors.New("negative memory budget")
	}
	n := int64(s.Batch) * int64(s.Seq)
	a := &p.Components
	a.Weights = 4 * c.ParameterCount()
	a.Gradients, a.AdamMoments = a.Weights, 2*a.Weights
	a.RoPE = 4 * int64(c.Context) * int64(c.Dim/c.Heads)
	a.BackwardWorkspace = 4 * backwardWorkspaceElements(c, s.Batch, s.Seq)
	a.TrainingActivationBound = 4 * activationElements(c, s.Batch, s.Seq)
	a.EvaluationWorkspace = 4 * evaluationWorkspaceElements(c, 1, s.Seq)
	// Tape.IDs is copied with append: allow two 64-bit ints per target for
	// capacity growth. Per-microbatch arrays are released before the next one;
	// accumulation increases compute, not the planned live payload.
	a.TrainingInputAllowance = 16 * n
	switch input.Kind {
	case "text":
		if input.TextBytes < 1 || input.TextBytes > 64<<20 || input.Shards != 0 || input.Workers != 0 || input.Depth != 0 {
			return p, errors.New("invalid text memory-plan input")
		}
		// At most one token per raw byte, plus BOS/EOS for each split. Allow
		// capacity growth and keep the source text charged throughout training.
		a.CorpusAllowance = input.TextBytes + 16*(input.TextBytes+4)
		a.TrainingInputAllowance += 16 * n // sampleBatch's X and Y
		a.EvaluationInputAllowance = 8 * int64(s.Seq)
	case "dataset":
		if input.TextBytes != 0 || input.Shards < 1 || input.Shards > maxDatasetShards {
			return p, errors.New("invalid dataset memory-plan input")
		}
		prefetch := PrefetchConfig{Batch: s.Batch, Seq: s.Seq, Workers: input.Workers, Depth: input.Depth}
		if err := prefetch.Validate(); err != nil {
			return p, err
		}
		// The existing 64-byte allowance covers X/Y and span slice growth per
		// outstanding target. Add bounded readers and planner index/order arrays.
		a.TrainingPrefetchAllowance = 64*n*int64(input.Depth) + (16<<10)*int64(input.Workers) + 64*int64(input.Shards)
		// Validation runs its own depth=1 reader while training prefetch remains
		// alive. It also has a second planner to determine the scoring range.
		a.EvaluationInputAllowance = 64*int64(s.Seq) + (16 << 10) + 128*int64(input.Shards)
	default:
		return p, errors.New("unknown memory-plan source kind")
	}
	p.PersistentBytes = a.Weights + a.Gradients + a.AdamMoments + a.RoPE + a.BackwardWorkspace + a.CorpusAllowance + a.TrainingPrefetchAllowance
	p.TrainingBytes = p.PersistentBytes + a.TrainingActivationBound + a.TrainingInputAllowance
	p.EvaluationBytes = p.PersistentBytes + a.EvaluationWorkspace + a.EvaluationInputAllowance
	p.RequiredBytes = max(p.TrainingBytes, p.EvaluationBytes)
	p.FitsBudget = budget == 0 || p.RequiredBytes <= budget
	return p, nil
}

func (p trainingMemoryPlan) checkBudget() error {
	if !p.FitsBudget {
		return fmt.Errorf("training memory plan requires %d bytes (%.2f MiB), exceeds -memory-mib budget %d bytes (%.2f MiB); reduce model, batch, sequence or prefetch, or raise the budget", p.RequiredBytes, float64(p.RequiredBytes)/(1<<20), p.BudgetBytes, float64(p.BudgetBytes)/(1<<20))
	}
	return nil
}

func (s TrainSpec) LearningRate(step int) float64 {
	if s.Warmup > 0 && step <= s.Warmup {
		if step == s.Warmup {
			return s.LR
		}
		return s.LR * float64(step) / float64(s.Warmup)
	}
	if s.Steps <= s.Warmup+1 {
		return s.MinLR
	}
	progress := float64(step-s.Warmup-1) / float64(s.Steps-s.Warmup-1)
	progress = math.Max(0, math.Min(1, progress))
	return s.MinLR + .5*(s.LR-s.MinLR)*(1+math.Cos(math.Pi*progress))
}

type TrainState struct {
	Spec         TrainSpec          `json:"spec"`
	Step         int                `json:"step"`
	TokensSeen   int64              `json:"tokens_seen"`
	RNG          RNG                `json:"rng"`
	CorpusSHA256 string             `json:"corpus_sha256"`
	Dataset      *DatasetTrainState `json:"dataset,omitempty"`
}

type DatasetTrainState struct {
	Version         int           `json:"version"`
	ManifestSHA256  string        `json:"manifest_sha256"`
	TokenizerSHA256 string        `json:"tokenizer_sha256"`
	OrderSeed       uint64        `json:"order_seed"`
	Cursor          DatasetCursor `json:"cursor"`
}

func (s *TrainState) Validate(c Config) error {
	if err := s.Spec.Validate(c); err != nil {
		return err
	}
	if s.Step < 0 || s.Step > s.Spec.Steps || s.TokensSeen != int64(s.Step)*int64(s.Spec.Batch)*int64(s.Spec.Seq)*int64(s.Spec.Accum) {
		return errors.New("inconsistent training step/token count")
	}
	hash, err := hex.DecodeString(s.CorpusSHA256)
	if err != nil || len(hash) != 32 {
		return errors.New("invalid corpus hash")
	}
	if d := s.Dataset; d != nil {
		if d.Version != 1 || !validSHA256(d.ManifestSHA256) || d.ManifestSHA256 != s.CorpusSHA256 || !validSHA256(d.TokenizerSHA256) || d.Cursor.Shard < 0 || d.Cursor.Shard >= maxDatasetShards || d.Cursor.Offset < 1 || d.Cursor.Offset > 1<<50 || d.Cursor.Epoch > uint64(s.TokensSeen)/2 {
			return errors.New("invalid dataset training state")
		}
	}
	return nil
}

// AdamW applies global norm clipping, bias correction and decoupled decay.
// Preflight rejects nonfinite gradients before modifying any parameter/state.
func (m *Model) AdamW(s TrainSpec, step int) (float64, error) {
	if m.hf != nil {
		return 0, errors.New("HF models are inference-only")
	}
	if err := s.Validate(m.Config); err != nil {
		return 0, err
	}
	if step < 1 || step > s.Steps {
		return 0, errors.New("optimizer step outside schedule")
	}
	var squared float64
	for _, p := range m.Params {
		if len(p.Grad) != len(p.Data) {
			return 0, errors.New("missing gradients")
		}
		if len(p.M) != 0 && (len(p.M) != len(p.Data) || len(p.V) != len(p.Data)) || len(p.M) == 0 && len(p.V) != 0 {
			return 0, errors.New("invalid optimizer state shape")
		}
		for i, g := range p.Grad {
			if !finite(float64(g)) || !finite(float64(p.Data[i])) {
				return 0, fmt.Errorf("nonfinite parameter/gradient: %s[%d]", p.Name, i)
			}
			if len(p.M) > 0 && (!finite(float64(p.M[i])) || !finite(float64(p.V[i])) || p.V[i] < 0) {
				return 0, errors.New("invalid Adam moments")
			}
			squared += float64(g) * float64(g)
		}
	}
	norm := math.Sqrt(squared)
	clip := 1.0
	if norm > s.Clip {
		clip = s.Clip / norm
	}
	lr := s.LearningRate(step)
	bc1 := 1 - math.Pow(s.Beta1, float64(step))
	bc2 := 1 - math.Pow(s.Beta2, float64(step))
	for _, p := range m.Params {
		if len(p.M) == 0 {
			p.M = make([]float32, len(p.Data))
			p.V = make([]float32, len(p.Data))
		}
		parallel(len(p.Data), len(p.Data)*8, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				g := float64(p.Grad[i]) * clip
				p.M[i] = float32(s.Beta1*float64(p.M[i]) + (1-s.Beta1)*g)
				p.V[i] = float32(s.Beta2*float64(p.V[i]) + (1-s.Beta2)*g*g)
				update := (float64(p.M[i]) / bc1) / (math.Sqrt(float64(p.V[i])/bc2) + s.Eps)
				if p.Decay {
					update += s.WeightDecay * float64(p.Data[i])
				}
				p.Data[i] -= float32(lr * update)
			}
		})
	}
	m.Revision++
	return norm, nil
}

// Checkpoints contain a bounded JSON header, ordered little-endian FP32 arrays,
// and a SHA-256 trailer over every preceding byte. Save uses fsync + atomic rename.
type checkpointMeta struct {
	Architecture string      `json:"architecture"`
	Config       Config      `json:"config"`
	Tokenizer    Tokenizer   `json:"tokenizer"`
	Training     *TrainState `json:"training,omitempty"`
}

const architecture = "pre-rms-rope-gqa-swiglu-tied-v1"

func strictJSON(data []byte, v any) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || data[0] != '{' {
		return errors.New("expected a JSON object")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return errors.New("expected exactly one JSON object")
	}
	return nil
}
func writeFloats(w io.Writer, x []float32) error {
	buf := make([]byte, 16*1024)
	for start := 0; start < len(x); {
		n := min(len(x)-start, len(buf)/4)
		for i := 0; i < n; i++ {
			v := x[start+i]
			if !finite(float64(v)) {
				return errors.New("cannot save nonfinite tensor")
			}
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
		}
		if _, err := w.Write(buf[:4*n]); err != nil {
			return err
		}
		start += n
	}
	return nil
}
func readFloats(r io.Reader, x []float32, nonnegative bool) error {
	buf := make([]byte, 16*1024)
	for start := 0; start < len(x); {
		n := min(len(x)-start, len(buf)/4)
		if _, err := io.ReadFull(r, buf[:4*n]); err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
			if !finite(float64(v)) || nonnegative && v < 0 {
				return errors.New("invalid checkpoint tensor value")
			}
			x[start+i] = v
		}
		start += n
	}
	return nil
}
func SaveCheckpoint(path string, m *Model, tok Tokenizer, state *TrainState) error {
	if m.hf != nil {
		return errors.New("HF models are inference-only; native checkpoints use a different model/tokenizer contract")
	}
	if err := m.Config.Validate(); err != nil {
		return err
	}
	if err := tok.Validate(); err != nil {
		return err
	}
	if tok.Vocab() != m.Config.Vocab {
		return errors.New("tokenizer/model vocabulary mismatch")
	}
	if state != nil {
		if err := state.Validate(m.Config); err != nil {
			return err
		}
		if state.Dataset != nil && state.Dataset.TokenizerSHA256 != tokenizerHash(tok) {
			return errors.New("checkpoint dataset/tokenizer identity mismatch")
		}
		for _, p := range m.Params {
			if len(p.M) != len(p.Data) || len(p.V) != len(p.Data) {
				return errors.New("missing optimizer moments")
			}
			for _, v := range p.V {
				if v < 0 {
					return errors.New("negative Adam variance")
				}
			}
		}
	}
	meta, err := json.Marshal(checkpointMeta{architecture, m.Config, tok, state})
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err = os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".monolith-*.tmp")
	if err != nil {
		return err
	}
	temp := f.Name()
	defer os.Remove(temp)
	defer f.Close()
	hash := sha256.New()
	w := bufio.NewWriterSize(io.MultiWriter(f, hash), 64*1024)
	if _, err = io.WriteString(w, checkpointMagic); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, uint32(len(meta))); err != nil {
		return err
	}
	if _, err = w.Write(meta); err != nil {
		return err
	}
	for _, p := range m.Params {
		if err = writeFloats(w, p.Data); err != nil {
			return err
		}
	}
	if state != nil {
		for _, p := range m.Params {
			if err = writeFloats(w, p.M); err != nil {
				return err
			}
			if err = writeFloats(w, p.V); err != nil {
				return err
			}
		}
	}
	if err = w.Flush(); err != nil {
		return err
	}
	if _, err = f.Write(hash.Sum(nil)); err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if err = os.Rename(temp, path); err != nil {
		return err
	}
	if d, e := os.Open(dir); e == nil {
		defer d.Close()
		_ = d.Sync()
	}
	return nil
}
func LoadCheckpoint(path string) (*Model, Tokenizer, *TrainState, error) {
	m, tok, state, _, err := loadCheckpointIdentity(path)
	return m, tok, state, err
}
func loadCheckpointIdentity(path string) (*Model, Tokenizer, *TrainState, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Tokenizer{}, nil, "", err
	}
	defer f.Close()
	return loadCheckpointFile(f)
}

// Digest, metadata and tensors come from the same open file, even if a trainer
// atomically replaces the path. In-place mutation remains unsupported.
func loadCheckpointFile(f *os.File) (*Model, Tokenizer, *TrainState, string, error) {
	checked, err := readCheckpointMetadata(f)
	if err != nil {
		return nil, Tokenizer{}, nil, "", err
	}
	m, err := loadCheckpointTensors(f, checked)
	if err != nil {
		return nil, Tokenizer{}, nil, "", err
	}
	return m, checked.meta.Tokenizer, checked.meta.Training, checked.identity, nil
}

type checkedCheckpoint struct {
	meta       checkpointMeta
	dataOffset int64
	identity   string
}

// Verify bounded metadata, expected file size and the complete checksum without
// allocating tensor arrays. Training can check its invocation budget here, then
// either return a plan or load tensors from this same descriptor.
func readCheckpointMetadata(f *os.File) (checkedCheckpoint, error) {
	stat, err := f.Stat()
	if err != nil {
		return checkedCheckpoint{}, err
	}
	r := io.NewSectionReader(f, 0, stat.Size())
	var prefix [12]byte
	if _, err = io.ReadFull(r, prefix[:]); err != nil {
		return checkedCheckpoint{}, err
	}
	if string(prefix[:8]) != checkpointMagic {
		return checkedCheckpoint{}, errors.New("unknown checkpoint magic/version")
	}
	headerLen := binary.LittleEndian.Uint32(prefix[8:])
	if headerLen == 0 || headerLen > 4*1024*1024 {
		return checkedCheckpoint{}, errors.New("checkpoint header outside allocation limit")
	}
	data := make([]byte, int(headerLen))
	if _, err = io.ReadFull(r, data); err != nil {
		return checkedCheckpoint{}, err
	}
	var meta checkpointMeta
	if err = strictJSON(data, &meta); err != nil {
		return checkedCheckpoint{}, err
	}
	if meta.Architecture != architecture {
		return checkedCheckpoint{}, errors.New("unsupported checkpoint architecture")
	}
	if err = meta.Config.Validate(); err != nil {
		return checkedCheckpoint{}, err
	}
	if err = meta.Tokenizer.Validate(); err != nil {
		return checkedCheckpoint{}, err
	}
	if meta.Tokenizer.Vocab() != meta.Config.Vocab {
		return checkedCheckpoint{}, errors.New("checkpoint vocabulary mismatch")
	}
	arrays := int64(1)
	if meta.Training != nil {
		arrays = 3
		if err = meta.Training.Validate(meta.Config); err != nil {
			return checkedCheckpoint{}, err
		}
		if meta.Training.Dataset != nil && meta.Training.Dataset.TokenizerSHA256 != tokenizerHash(meta.Tokenizer) {
			return checkedCheckpoint{}, errors.New("checkpoint dataset/tokenizer identity mismatch")
		}
	}
	expected := 12 + int64(headerLen) + 4*arrays*meta.Config.ParameterCount() + sha256.Size
	if stat.Size() != expected {
		return checkedCheckpoint{}, fmt.Errorf("checkpoint size mismatch: got %d, expected %d", stat.Size(), expected)
	}
	hash := sha256.New()
	if _, err = io.Copy(hash, io.NewSectionReader(f, 0, expected-sha256.Size)); err != nil {
		return checkedCheckpoint{}, err
	}
	var sum [sha256.Size]byte
	if _, err = f.ReadAt(sum[:], expected-sha256.Size); err != nil {
		return checkedCheckpoint{}, err
	}
	if !bytes.Equal(sum[:], hash.Sum(nil)) {
		return checkedCheckpoint{}, errors.New("checkpoint checksum mismatch")
	}
	hash.Write(sum[:])
	identity := hex.EncodeToString(hash.Sum(nil))
	return checkedCheckpoint{meta: meta, dataOffset: 12 + int64(headerLen), identity: identity}, nil
}

func loadCheckpointTensors(f *os.File, checked checkedCheckpoint) (*Model, error) {
	meta := checked.meta
	m, err := allocateModel(meta.Config, nil)
	if err != nil {
		return nil, err
	}
	if _, err = f.Seek(checked.dataOffset, io.SeekStart); err != nil {
		return nil, err
	}
	r := bufio.NewReaderSize(f, 64*1024)
	for _, p := range m.Params {
		if err = readFloats(r, p.Data, false); err != nil {
			return nil, err
		}
	}
	if meta.Training != nil {
		for _, p := range m.Params {
			p.M = make([]float32, len(p.Data))
			p.V = make([]float32, len(p.Data))
			if err = readFloats(r, p.M, false); err != nil {
				return nil, err
			}
			if err = readFloats(r, p.V, true); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}
func (m *Model) DropOptimizer() {
	for _, p := range m.Params {
		p.Grad = nil
		p.M = nil
		p.V = nil
	}
}

func corpusHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}
func splitCorpus(tok Tokenizer, text string, fraction float64, seq int) ([]int, []int, error) {
	cut := int(float64(len(text)) * (1 - fraction))
	train := tok.Encode(text[:cut], true, true)
	val := tok.Encode(text[cut:], true, true)
	if len(train) < seq+1 || len(val) < seq+1 {
		return nil, nil, errors.New("corpus split is shorter than sequence+1; provide more text or reduce -seq")
	}
	return train, val, nil
}

// Every independent training window starts with BOS, exactly like inference.
// Data is [BOS, payload..., EOS]; EOS can be the final supervised target.
func sampleBatch(data []int, batch, seq int, rng *RNG) ([]int, []int) {
	x, y := make([]int, batch*seq), make([]int, batch*seq)
	for b := 0; b < batch; b++ {
		start := 1 + rng.Intn(len(data)-seq)
		x[b*seq] = BOS
		copy(x[b*seq+1:(b+1)*seq], data[start:start+seq-1])
		copy(y[b*seq:(b+1)*seq], data[start:start+seq])
	}
	return x, y
}
func trainUpdate(m *Model, s *TrainState, data []int) (float64, float64, error) {
	return observedTrainUpdate(context.Background(), m, s, data, nil, nil)
}
func observedTrainUpdate(ctx context.Context, m *Model, s *TrainState, data []int, observer *trainingObserver, w *backwardWorkspace) (float64, float64, error) {
	spec := s.Spec
	if w != nil {
		if err := w.validate(m, spec.Batch, spec.Seq); err != nil {
			return 0, 0, err
		}
	}
	_ = observer.phase(ctx, "zero_grad", func() error { m.ZeroGrad(); return nil })
	var loss float64
	for micro := 0; micro < spec.Accum; micro++ {
		var x, y []int
		_ = observer.phase(ctx, "input", func() error { x, y = sampleBatch(data, spec.Batch, spec.Seq, &s.RNG); return nil })
		l, err := observedMicrobatch(ctx, m, spec, x, y, observer, w)
		if err != nil {
			return 0, 0, err
		}
		loss += l / float64(spec.Accum)
	}
	var norm float64
	err := observer.phase(ctx, "optimizer", func() (err error) { norm, err = m.AdamW(spec, s.Step+1); return err })
	if err != nil {
		return 0, 0, err
	}
	s.Step++
	s.TokensSeen += int64(spec.Batch) * int64(spec.Seq) * int64(spec.Accum)
	return loss, norm, nil
}

func datasetTrainingState(d *TokenDataset, seed uint64) *DatasetTrainState {
	return &DatasetTrainState{Version: 1, ManifestSHA256: d.id, TokenizerSHA256: d.manifest.TokenizerSHA256, OrderSeed: seed, Cursor: DatasetCursor{Offset: 1}}
}

// Reconcile position against committed optimizer progress, not speculative IO.
func validateDatasetResume(d *TokenDataset, tok Tokenizer, s *TrainState) error {
	state := s.Dataset
	if state == nil || state.Version != 1 || state.ManifestSHA256 != d.id || s.CorpusSHA256 != d.id {
		return errors.New("resume dataset manifest identity differs from original")
	}
	if state.TokenizerSHA256 != d.manifest.TokenizerSHA256 || tokenizerHash(tok) != d.manifest.TokenizerSHA256 {
		return errors.New("resume dataset tokenizer identity differs from original")
	}
	p, err := newTokenPlanner(d, "train", PrefetchConfig{Batch: 1, Seq: 1, Workers: 1, Depth: 1, Seed: state.OrderSeed, Shuffle: true}, DatasetCursor{Offset: 1})
	if err != nil {
		return err
	}
	p.cursor.Epoch = uint64(s.TokensSeen / p.perEpoch)
	p.setOrder()
	left := s.TokensSeen % p.perEpoch
	for i, index := range p.order {
		n := d.manifest.Shards[index].Tokens - 1
		if left < n {
			p.cursor.Shard = i
			p.cursor.Offset = left + 1
			break
		}
		left -= n
	}
	if state.Cursor != p.cursor {
		return errors.New("dataset cursor does not match committed training tokens")
	}
	return nil
}

func trainDatasetUpdate(ctx context.Context, m *Model, s *TrainState, p *TokenPrefetch) (float64, float64, error) {
	return observedDatasetUpdate(ctx, m, s, p, nil, nil)
}
func observedDatasetUpdate(ctx context.Context, m *Model, s *TrainState, p *TokenPrefetch, observer *trainingObserver, w *backwardWorkspace) (float64, float64, error) {
	spec := s.Spec
	if w != nil {
		if err := w.validate(m, spec.Batch, spec.Seq); err != nil {
			return 0, 0, err
		}
	}
	_ = observer.phase(ctx, "zero_grad", func() error { m.ZeroGrad(); return nil })
	var loss float64
	var after DatasetCursor
	for micro := 0; micro < spec.Accum; micro++ {
		var b *TokenBatch
		err := observer.phase(ctx, "input", func() (err error) { b, err = p.Next(ctx); return err })
		if err != nil {
			return 0, 0, err
		}
		l, err := observedMicrobatch(ctx, m, spec, b.X, b.Y, observer, w)
		after = b.After
		b.Release()
		if err != nil {
			return 0, 0, err
		}
		loss += l / float64(spec.Accum)
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	var norm float64
	err := observer.phase(ctx, "optimizer", func() (err error) { norm, err = m.AdamW(spec, s.Step+1); return err })
	if err != nil {
		return 0, 0, err
	}
	s.Step++
	s.TokensSeen += int64(spec.Batch) * int64(spec.Seq) * int64(spec.Accum)
	s.Dataset.Cursor = after
	return loss, norm, nil
}

func observedMicrobatch(ctx context.Context, m *Model, spec TrainSpec, x, y []int, observer *trainingObserver, w *backwardWorkspace) (float64, error) {
	var tape *Tape
	err := observer.phase(ctx, "forward", func() (err error) { tape, err = m.Forward(x, spec.Batch, spec.Seq); return err })
	var loss float64
	if err == nil {
		err = observer.phase(ctx, "backward", func() (err error) { loss, err = m.backward(tape, y, 1/float64(spec.Accum), w); return err })
	}
	return loss, err
}

func evaluateDataset(ctx context.Context, m *Model, d *TokenDataset, seq, maxBatches int) (float64, int, error) {
	return evaluateDatasetSplit(ctx, m, d, "validation", seq, maxBatches)
}
func evaluateDatasetSplit(ctx context.Context, m *Model, d *TokenDataset, split string, seq, maxBatches int) (float64, int, error) {
	if seq < 1 || seq > m.Config.Context || maxBatches < 0 {
		return 0, 0, errors.New("invalid dataset evaluation settings")
	}
	if ctx == nil {
		return 0, 0, errors.New("nil evaluation context")
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	c := PrefetchConfig{Batch: 1, Seq: seq, Workers: 1, Depth: 1}
	planner, err := newTokenPlanner(d, split, c, DatasetCursor{Offset: 1})
	if err != nil {
		return 0, 0, err
	}
	w, err := newEvaluationWorkspace(m, 1, int(min(int64(seq), planner.perEpoch)))
	if err != nil {
		return 0, 0, err
	}
	p, err := NewTokenPrefetch(ctx, d, split, c, DatasetCursor{Offset: 1})
	if err != nil {
		return 0, 0, err
	}
	defer p.Close()
	left := planner.perEpoch
	var total float64
	tokens := 0
	for batches := 0; left > 0 && (maxBatches == 0 || batches < maxBatches); batches++ {
		b, err := p.Next(ctx)
		if err != nil {
			return 0, tokens, err
		}
		n := int(min(int64(seq), left))
		logits, err := m.forwardEvaluation(ctx, b.X[:n], 1, n, w)
		var loss float64
		if err == nil {
			loss, _, err = CrossEntropy(logits, b.Y[:n], m.Config.Vocab, 1, false)
		}
		b.Release()
		if err != nil {
			return 0, tokens, err
		}
		total += loss * float64(n)
		tokens += n
		left -= int64(n)
	}
	return total / float64(tokens), tokens, nil
}

// Evaluation scores every post-BOS target exactly once in independent, BOS-
// framed windows. A positive maxBatches selects a deterministic leading subset.
func evaluate(ctx context.Context, m *Model, data []int, seq, maxBatches int) (float64, int, error) {
	if len(data) < 2 || data[0] != BOS || seq < 1 || seq > m.Config.Context || maxBatches < 0 {
		return 0, 0, errors.New("invalid evaluation data/settings (BOS-framed input required)")
	}
	if ctx == nil {
		return 0, 0, errors.New("nil evaluation context")
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	w, err := newEvaluationWorkspace(m, 1, min(seq, len(data)-1))
	if err != nil {
		return 0, 0, err
	}
	input := make([]int, w.maxSeq)
	var total float64
	tokens, batches := 0, 0
	for start := 1; start < len(data) && (maxBatches == 0 || batches < maxBatches); start += seq {
		if err := ctx.Err(); err != nil {
			return 0, tokens, err
		}
		end := min(start+seq, len(data))
		x := input[:end-start]
		x[0] = BOS
		copy(x[1:], data[start:end-1])
		logits, err := m.forwardEvaluation(ctx, x, 1, len(x), w)
		if err != nil {
			return 0, tokens, err
		}
		loss, _, err := CrossEntropy(logits, data[start:end], m.Config.Vocab, 1, false)
		if err != nil {
			return 0, tokens, err
		}
		total += loss * float64(end-start)
		tokens += end - start
		batches++
	}
	return total / float64(tokens), tokens, nil
}

// Sampling is per-request and deterministic, independent of cohort membership.
type Sampling struct {
	MaxTokens     int     `json:"max_tokens"`
	Temperature   float64 `json:"temperature"`
	TopK          int     `json:"top_k"`
	TopP          float64 `json:"top_p"`
	RepeatPenalty float64 `json:"repeat_penalty"`
	Seed          uint64  `json:"seed"`
}

func defaultSampling() Sampling {
	return Sampling{MaxTokens: 64, Temperature: .8, TopK: 40, TopP: .95, RepeatPenalty: 1.1, Seed: 1}
}
func (s Sampling) Validate(c Config) error {
	if s.MaxTokens < 1 || s.MaxTokens > c.Context || s.TopK < 0 || !finite(s.Temperature) || s.Temperature < 0 || s.Temperature > 100 || !finite(s.TopP) || s.TopP <= 0 || s.TopP > 1 || !finite(s.RepeatPenalty) || s.RepeatPenalty < 1 || s.RepeatPenalty > 100 {
		return errors.New("invalid sampling settings")
	}
	return nil
}

type candidate struct {
	ID          int
	Logit, Prob float64
}

func sample(logits []float32, history []int, s Sampling, rng *RNG) (int, error) {
	return sampleWithBlocked(logits, history, s, rng, BOS)
}
func sampleWithBlocked(logits []float32, history []int, s Sampling, rng *RNG, blocked int) (int, error) {
	if len(logits) == 0 {
		return 0, errors.New("empty sampling distribution")
	}
	var seen map[int]bool
	if s.Temperature != 0 || s.RepeatPenalty != 1 {
		seen = make(map[int]bool, len(history))
		for _, id := range history {
			seen[id] = true
		}
	}
	if s.Temperature == 0 {
		best, bestLogit := -1, float64(0)
		for id, v := range logits {
			// Validate even masked logits and logits after the eventual winner.
			if !finite(float64(v)) {
				return 0, errors.New("nonfinite sampling logit")
			}
			if id == blocked {
				continue
			}
			x := float64(v)
			if seen[id] {
				if x > 0 {
					x /= s.RepeatPenalty
				} else {
					x *= s.RepeatPenalty
				}
			}
			// Ascending IDs and strict comparison preserve the sorter's tie rule,
			// including signed zero. Greedy sampling never consumes RNG state.
			if best < 0 || x > bestLogit {
				best, bestLogit = id, x
			}
		}
		if best < 0 {
			return 0, errors.New("empty sampling distribution")
		}
		return best, nil
	}
	cs := make([]candidate, 0, len(logits)-1)
	for id, v := range logits {
		if !finite(float64(v)) {
			return 0, errors.New("nonfinite sampling logit")
		}
		if id == blocked {
			continue
		}
		x := float64(v)
		if seen[id] {
			if x > 0 {
				x /= s.RepeatPenalty
			} else {
				x *= s.RepeatPenalty
			}
		}
		cs = append(cs, candidate{ID: id, Logit: x})
	}
	if len(cs) == 0 {
		return 0, errors.New("empty sampling distribution")
	}
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].Logit == cs[j].Logit {
			return cs[i].ID < cs[j].ID
		}
		return cs[i].Logit > cs[j].Logit
	})
	if s.TopK > 0 && s.TopK < len(cs) {
		cs = cs[:s.TopK]
	}
	var total float64
	maxLogit := cs[0].Logit
	for i := range cs {
		cs[i].Prob = math.Exp((cs[i].Logit - maxLogit) / s.Temperature)
		total += cs[i].Prob
	}
	if s.TopP < 1 {
		var mass float64
		for i, c := range cs {
			mass += c.Prob
			if mass >= s.TopP*total {
				cs = cs[:i+1]
				total = mass
				break
			}
		}
	}
	draw := rng.Float64() * total
	for _, c := range cs {
		draw -= c.Prob
		if draw < 0 {
			return c.ID, nil
		}
	}
	return cs[len(cs)-1].ID, nil
}

type Completion struct {
	Text             string `json:"text"`
	TokenIDs         []int  `json:"token_ids"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	FinishReason     string `json:"finish_reason"`
}

func validateInferenceTokenizer(m *Model, tok inferenceTokenizer) error {
	if tok == nil {
		return errors.New("nil inference tokenizer")
	}
	h, ok := tok.(*hfTokenizer)
	if ok && h == nil {
		return errors.New("nil HF tokenizer")
	}
	if (m.hf != nil) != ok || ok && (h.bos != m.hf.BOS || h.eos != m.hf.EOS) {
		return errors.New("incompatible model/tokenizer inference profiles")
	}
	if tok.Vocab() != m.Config.Vocab {
		return errors.New("tokenizer/model mismatch")
	}
	return nil
}
func preparePrompt(ctx context.Context, m *Model, tok inferenceTokenizer, prompt string, s Sampling) ([]int, error) {
	if err := validateInferenceTokenizer(m, tok); err != nil {
		return nil, err
	}
	if err := s.Validate(m.Config); err != nil {
		return nil, err
	}
	if len(prompt) > 1024*1024 {
		return nil, errors.New("prompt exceeds 1 MiB")
	}
	ids, err := tok.encodePrompt(ctx, prompt, m.Config.Context-s.MaxTokens)
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, errors.New("prompt must encode at least one token")
	}
	if len(ids)+s.MaxTokens > m.Config.Context {
		return nil, fmt.Errorf("prompt (%d) + generation (%d) exceeds context (%d)", len(ids), s.MaxTokens, m.Config.Context)
	}
	return ids, nil
}
func completion(tok inferenceTokenizer, output []int, prompt int, reason string) Completion {
	return Completion{strings.ToValidUTF8(tok.Decode(output), "\ufffd"), output, prompt, len(output), reason}
}
func Generate(ctx context.Context, m *Model, tok inferenceTokenizer, prompt string, s Sampling) (Completion, error) {
	ids, err := preparePrompt(ctx, m, tok, prompt, s)
	if err != nil {
		return Completion{}, err
	}
	cache := m.NewCache()
	defer cache.Close()
	var logits []float32
	for _, id := range ids {
		if err = ctx.Err(); err != nil {
			return Completion{}, err
		}
		logits, err = m.DecodeBatch(ctx, []*KVCache{cache}, []int{id})
		if err != nil {
			return Completion{}, err
		}
	}
	history := append([]int(nil), ids...)
	out := make([]int, 0, s.MaxTokens)
	rng := &RNG{s.Seed}
	reason := "length"
	for i := 0; i < s.MaxTokens; i++ {
		if err = ctx.Err(); err != nil {
			return Completion{}, err
		}
		id, e := sampleWithBlocked(logits, history, s, rng, tok.blockedID())
		if e != nil {
			return Completion{}, e
		}
		out = append(out, id)
		history = append(history, id)
		if id == tok.eosID() {
			reason = "eos"
			break
		}
		if i+1 < s.MaxTokens {
			logits, err = m.DecodeBatch(ctx, []*KVCache{cache}, []int{id})
			if err != nil {
				return Completion{}, err
			}
		}
	}
	return completion(tok, out, len(ids), reason), nil
}

// Engine owns sessions and KV pages in one goroutine. Every tick gives each
// active session progress, with a bounded token budget for chunked prefill.
var ErrBusy = errors.New("inference capacity exhausted; retry later")
var ErrClosed = errors.New("inference engine closed")
var ErrSlowConsumer = errors.New("stream consumer exceeded its buffered token capacity")

type EngineConfig struct {
	MaxBatch, Queue                                            int
	CacheBytes                                                 int64
	StreamBuffer, PrefillChunk, MixedPrefillChunk, TokenBudget int
	KernelWorkers                                              int // 0 captures min(GOMAXPROCS,256); 1 runs inline.
}

type TokenEvent struct {
	TokenID int    `json:"token_id"`
	Text    string `json:"text"`
}

// incrementalText preserves strings.ToValidUTF8 semantics across token pieces,
// including one replacement per consecutive invalid-byte run.
type incrementalText struct {
	pending []byte
	invalid bool
	hf      bool
}

func (d *incrementalText) Push(piece []byte, final bool) string {
	if d.hf {
		return d.pushHF(piece, final)
	}
	data := append(d.pending, piece...)
	d.pending = nil
	var out strings.Builder
	for len(data) > 0 {
		if !final && !utf8.FullRune(data) {
			d.pending = append(d.pending, data...)
			break
		}
		r, n := utf8.DecodeRune(data)
		if r == utf8.RuneError && n == 1 {
			if !d.invalid {
				out.WriteRune(utf8.RuneError)
			}
			d.invalid = true
		} else {
			out.Write(data[:n])
			d.invalid = false
		}
		data = data[n:]
	}
	return out.String()
}

// Match the ByteLevel decoder's UTF-8 lossy conversion, including one
// replacement for an incomplete valid prefix and separate invalid lead bytes.
func (d *incrementalText) pushHF(piece []byte, final bool) string {
	data := append(d.pending, piece...)
	d.pending = nil
	var out strings.Builder
	for pos := 0; pos < len(data); {
		b := data[pos]
		need := 1
		switch {
		case b < 0x80:
			out.WriteByte(b)
			pos++
			continue
		case b >= 0xc2 && b <= 0xdf:
			need = 2
		case b >= 0xe0 && b <= 0xef:
			need = 3
		case b >= 0xf0 && b <= 0xf4:
			need = 4
		}
		n := 1
		for n < need && pos+n < len(data) {
			c := data[pos+n]
			if c < 0x80 || c > 0xbf || n == 1 && (b == 0xe0 && c < 0xa0 || b == 0xed && c >= 0xa0 || b == 0xf0 && c < 0x90 || b == 0xf4 && c >= 0x90) {
				break
			}
			n++
		}
		if need > 1 && n == need {
			out.Write(data[pos : pos+n])
			pos += n
			continue
		}
		if need > 1 && pos+n == len(data) && n < need && !final {
			d.pending = append(d.pending, data[pos:]...)
			break
		}
		out.WriteRune(utf8.RuneError)
		pos += n
	}
	return out.String()
}

type engineRequest struct {
	Ctx               context.Context
	IDs               []int
	Sampling          Sampling
	Reply             chan engineReply
	Events            chan TokenEvent
	Cancel            context.CancelFunc
	release           func()
	Started, Enqueued time.Time
	Released          chan struct{}
	task              *trace.Task
	cancelContext     context.CancelFunc
	stopWatches       [2]func()
	lifecycleMu       sync.Mutex
	cancelObserved    time.Time
	terminal          bool
	computeDone       time.Time // Actor-owned; every kernel has joined at this point.
}

// Cancellation timestamps are observations, not the instant a remote peer
// disconnected. Close records synchronously; context callbacks may be scheduled
// later. There are at most two callbacks per admitted request, with no idle
// watcher goroutines for standard cancelable contexts.
func (r *engineRequest) observeCancellation(source string) {
	r.lifecycleMu.Lock()
	defer r.lifecycleMu.Unlock()
	if r.terminal || !r.cancelObserved.IsZero() {
		return
	}
	r.cancelObserved = time.Now()
	r.log("cancel_observed:" + source)
}
func (r *engineRequest) log(event string) {
	if r.task != nil {
		trace.Log(r.Ctx, "lifecycle", event)
	}
}
func (r *engineRequest) watchCancellation(ctx context.Context, source string) func() {
	if ctx == nil { // Only mailbox-only tests construct an engine without a context.
		return func() {}
	}
	done := make(chan struct{})
	stop := context.AfterFunc(ctx, func() {
		defer close(done)
		r.observeCancellation(source)
	})
	return func() {
		if !stop() {
			<-done // AfterFunc's stop alone does not join a running callback.
		}
	}
}
func (r *engineRequest) stopObservation() time.Time {
	for _, stop := range r.stopWatches {
		stop()
	}
	r.lifecycleMu.Lock()
	defer r.lifecycleMu.Unlock()
	r.terminal = true
	return r.cancelObserved
}

type engineReply struct {
	Completion Completion
	Err        error
}
type CompletionStream struct {
	Events  <-chan TokenEvent
	request *engineRequest
}

func (s *CompletionStream) Close() { s.request.Cancel() }

// Released closes after computation and cancellation observers have stopped,
// KV reservations/pages have been returned, and the admission slot is free.
// It does not imply that the caller's event/reply buffers or cached pages have
// been freed by the GC. Unlike Wait, it never completes merely on cancellation.
func (s *CompletionStream) Released() <-chan struct{} { return s.request.Released }

// Wait returns the terminal result. Call once, after draining Events, unless
// abandoning the stream. Cancellation never requires draining its event buffer.
// Wait may return a context error before Released closes.
func (s *CompletionStream) Wait() (Completion, error) {
	select {
	case r := <-s.request.Reply:
		return r.Completion, r.Err
	default:
	}
	select {
	case r := <-s.request.Reply:
		return r.Completion, r.Err
	case <-s.request.Ctx.Done():
		select {
		case r := <-s.request.Reply:
			return r.Completion, r.Err
		default:
			return Completion{}, s.request.Ctx.Err()
		}
	}
}

type engineSession struct {
	Request         *engineRequest
	Cache           *KVCache
	Cursor          int
	Output, History []int
	RNG             RNG
	Text            incrementalText
	LastToken       time.Time
}
type engineCounters struct {
	Submitted, Completed, Canceled, Failed, Rejected, Active, Reserved, Allocated atomic.Int64
	InputTokens, GeneratedTokens, Ticks, SlowConsumers, PeakTickTokens            atomic.Int64
	Released, Computing, KernelWorkers, WorkspaceFloatBytes                       atomic.Int64
}
type EngineStats struct {
	Submitted           int64 `json:"submitted"`
	Completed           int64 `json:"completed"`
	Canceled            int64 `json:"canceled"`
	Failed              int64 `json:"failed"`
	Rejected            int64 `json:"rejected"`
	Active              int64 `json:"active"`
	Queue               int   `json:"queue"`
	KVReservedBytes     int64 `json:"kv_reserved_bytes"`
	KVAllocatedBytes    int64 `json:"kv_allocated_bytes"`
	InputTokens         int64 `json:"input_tokens"`
	GeneratedTokens     int64 `json:"generated_tokens"`
	Ticks               int64 `json:"ticks"`
	SlowConsumers       int64 `json:"slow_consumers"`
	PeakTickTokens      int64 `json:"peak_tick_tokens"`
	Released            int64 `json:"released"`
	Computing           int64 `json:"computing"`
	AdmissionUsed       int   `json:"admission_used"`
	KernelWorkers       int64 `json:"kernel_workers"`
	WorkspaceFloatBytes int64 `json:"workspace_float_bytes"`
}

var latencyBounds = [...]float64{.0001, .0005, .001, .005, .01, .05, .1, .5, 1, 5, 30, 120}

type latencyHistogram struct {
	buckets      [len(latencyBounds)]atomic.Uint64
	count, nanos atomic.Uint64
}

func (h *latencyHistogram) observe(d time.Duration) {
	h.count.Add(1)
	h.nanos.Add(uint64(max(d, 0)))
	for i, bound := range latencyBounds {
		if d.Seconds() <= bound {
			h.buckets[i].Add(1)
			break
		}
	}
}
func (h *latencyHistogram) write(w io.Writer, name string) {
	fmt.Fprintf(w, "# TYPE %s histogram\n", name)
	var cumulative uint64
	for i, bound := range latencyBounds {
		cumulative += h.buckets[i].Load()
		fmt.Fprintf(w, "%s_bucket{le=\"%g\"} %d\n", name, bound, cumulative)
	}
	fmt.Fprintf(w, "%s_bucket{le=\"+Inf\"} %d\n%s_count %d\n%s_sum %.9f\n", name, h.count.Load(), name, h.count.Load(), name, float64(h.nanos.Load())/1e9)
}

type Engine struct {
	Model                                      *Model
	Tokenizer                                  inferenceTokenizer
	Config                                     EngineConfig
	queue                                      chan *engineRequest
	admission                                  chan struct{}
	httpSlots                                  chan struct{}
	ctx                                        context.Context
	cancel                                     context.CancelFunc
	done                                       chan struct{}
	pool                                       pagePool
	stats                                      engineCounters
	pieces                                     [][]byte
	submitMu                                   sync.Mutex
	queueLatency, firstTokenLatency, tokenGap  latencyHistogram
	batchCompute, cancelCompute, cancelRelease latencyHistogram
}

func NewEngine(m *Model, tok inferenceTokenizer, c EngineConfig) (*Engine, error) {
	e, err := newEngine(m, tok, c)
	if err != nil {
		return nil, err
	}
	go e.run()
	return e, nil
}

// Capacity validation is shared by CLI preflight and engine construction.
func normalizeEngineConfig(c EngineConfig) (EngineConfig, error) {
	if c.StreamBuffer == 0 {
		c.StreamBuffer = 32
	}
	if c.PrefillChunk == 0 {
		c.PrefillChunk = 16
	}
	if c.MixedPrefillChunk == 0 {
		c.MixedPrefillChunk = 1
	}
	if c.TokenBudget == 0 {
		c.TokenBudget = max(c.MaxBatch, 64)
	}
	if c.KernelWorkers == 0 {
		c.KernelWorkers = min(runtime.GOMAXPROCS(0), 256)
	}
	if c.KernelWorkers < 1 || c.KernelWorkers > 256 {
		return c, errors.New("invalid kernel worker count (1..256, or 0 for automatic)")
	}
	if c.MaxBatch < 1 || c.MaxBatch > 256 || c.Queue < 1 || c.Queue > 65536 || c.CacheBytes < 1 || c.CacheBytes > 1<<40 || c.StreamBuffer < 1 || c.StreamBuffer > 4096 || c.PrefillChunk < 1 || c.PrefillChunk > 1024 || c.MixedPrefillChunk < 1 || c.MixedPrefillChunk > 1024 || c.TokenBudget < c.MaxBatch || c.TokenBudget > 1024 {
		return c, errors.New("invalid engine capacity or token budget")
	}
	return c, nil
}

// Construct ownership before starting the actor. Tests can stage a complete
// cohort before run, without timing assumptions or replacing model kernels.
func newEngine(m *Model, tok inferenceTokenizer, c EngineConfig) (*Engine, error) {
	if err := validateInferenceTokenizer(m, tok); err != nil {
		return nil, err
	}
	c, err := normalizeEngineConfig(c)
	if err != nil {
		return nil, err
	}
	if err := tok.Validate(); err != nil {
		return nil, err
	}
	if tok.Vocab() != m.Config.Vocab {
		return nil, errors.New("tokenizer/model mismatch")
	}
	elements := 2 * pageTokens * m.Config.KVDim() * m.Config.Layers
	limit := int(c.CacheBytes / int64(elements*4))
	if limit < 1 {
		return nil, errors.New("cache budget cannot hold one KV page")
	}
	ctx, cancel := context.WithCancel(context.Background())
	e := &Engine{Model: m, Tokenizer: tok, Config: c, queue: make(chan *engineRequest, c.Queue), admission: make(chan struct{}, c.Queue+c.MaxBatch), httpSlots: make(chan struct{}, c.Queue+c.MaxBatch), ctx: ctx, cancel: cancel, done: make(chan struct{}), pool: pagePool{PageElements: elements, Limit: limit}, pieces: tok.Pieces()}
	return e, nil
}
func (e *Engine) Stats() EngineStats {
	s := &e.stats
	return EngineStats{Submitted: s.Submitted.Load(), Completed: s.Completed.Load(), Canceled: s.Canceled.Load(), Failed: s.Failed.Load(), Rejected: s.Rejected.Load(), Active: s.Active.Load(), Queue: len(e.queue), KVReservedBytes: s.Reserved.Load(), KVAllocatedBytes: s.Allocated.Load(), InputTokens: s.InputTokens.Load(), GeneratedTokens: s.GeneratedTokens.Load(), Ticks: s.Ticks.Load(), SlowConsumers: s.SlowConsumers.Load(), PeakTickTokens: s.PeakTickTokens.Load(), Released: s.Released.Load(), Computing: s.Computing.Load(), AdmissionUsed: len(e.admission), KernelWorkers: s.KernelWorkers.Load(), WorkspaceFloatBytes: s.WorkspaceFloatBytes.Load()}
}
func (e *Engine) Close() { e.cancel(); <-e.done }
func (e *Engine) enqueue(ctx context.Context, prompt string, s Sampling, streaming bool) (*engineRequest, error) {
	started := time.Now()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case <-e.done:
		return nil, ErrClosed
	default:
	}
	select {
	case e.admission <- struct{}{}:
	default:
		e.stats.Rejected.Add(1)
		return nil, ErrBusy
	}
	handedOff := false
	defer func() {
		if !handedOff {
			<-e.admission
		}
	}()
	ids, err := preparePrompt(ctx, e.Model, e.Tokenizer, prompt, s)
	if err != nil {
		return nil, err
	}
	pages := (len(ids) + s.MaxTokens + pageTokens - 1) / pageTokens
	if pages > e.pool.Limit {
		return nil, errors.New("request exceeds total KV cache budget")
	}
	ctx, cancel := context.WithCancel(ctx)
	r := &engineRequest{Ctx: ctx, IDs: ids, Sampling: s, Reply: make(chan engineReply, 1), cancelContext: cancel, release: func() { <-e.admission }, Started: started, Enqueued: time.Now(), Released: make(chan struct{})}
	if trace.IsEnabled() {
		r.Ctx, r.task = trace.NewTask(ctx, "monolith.request")
	}
	r.Cancel = func() {
		r.observeCancellation("caller")
		cancel()
	}
	r.stopWatches[0] = r.watchCancellation(ctx, "context")
	r.stopWatches[1] = r.watchCancellation(e.ctx, "shutdown")
	defer func() {
		if !handedOff {
			r.stopObservation()
			cancel()
			r.log("not_enqueued")
			if r.task != nil {
				r.task.End()
			}
		}
	}()
	if streaming {
		r.Events = make(chan TokenEvent, e.Config.StreamBuffer)
	}
	// Serialize submission with the shutdown drain, so no caller can enqueue
	// after the actor has stopped and strand an admission slot or terminal reply.
	e.submitMu.Lock()
	defer e.submitMu.Unlock()
	select {
	case <-e.done:
		cancel()
		return nil, ErrClosed
	default:
	}
	if e.ctx != nil && e.ctx.Err() != nil {
		cancel()
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		cancel()
		return nil, err
	}
	r.log("enqueue") // Publication must follow this event; the actor can run immediately.
	select {
	case e.queue <- r:
		handedOff = true
		e.stats.Submitted.Add(1)
		return r, nil
	default:
		cancel()
		e.stats.Rejected.Add(1)
		return nil, ErrBusy
	}
}
func (e *Engine) Stream(ctx context.Context, prompt string, s Sampling) (*CompletionStream, error) {
	r, err := e.enqueue(ctx, prompt, s, true)
	if err != nil {
		return nil, err
	}
	return &CompletionStream{Events: r.Events, request: r}, nil
}
func (e *Engine) Generate(ctx context.Context, prompt string, s Sampling) (Completion, error) {
	r, err := e.enqueue(ctx, prompt, s, false)
	if err != nil {
		return Completion{}, err
	}
	defer r.Cancel()
	return (&CompletionStream{request: r}).Wait()
}

// scheduleTokens guarantees one row per active session, then distributes extra
// prefill rows round-robin. While any session is decoding, a smaller chunk
// limit can protect output cadence. Decoding sessions advance on every tick.
func scheduleTokens(sessions []*engineSession, counts []int, budget, chunk, mixedChunk, start int) int {
	for _, s := range sessions {
		if s.Cursor == len(s.Request.IDs) {
			chunk = min(chunk, mixedChunk)
			break
		}
	}
	total := len(sessions)
	for i := range sessions {
		counts[i] = 1
	}
	for total < budget {
		progress := false
		for j := range sessions {
			i := (start + j) % len(sessions)
			remaining := len(sessions[i].Request.IDs) - sessions[i].Cursor
			if counts[i] < min(chunk, remaining) {
				counts[i]++
				total++
				progress = true
				if total == budget {
					break
				}
			}
		}
		if !progress {
			break
		}
	}
	return total
}
func (e *Engine) run() {
	defer close(e.done)
	executor := newKernelExecutor(e.Config.KernelWorkers)
	var workspace decodeWorkspace
	if executor.workers > 1 {
		e.stats.KernelWorkers.Store(int64(executor.workers))
	}
	defer func() {
		executor.close()
		workspace = decodeWorkspace{}
		e.stats.KernelWorkers.Store(0)
		e.stats.WorkspaceFloatBytes.Store(0)
	}()
	sessions := make([]*engineSession, 0, e.Config.MaxBatch)
	caches := make([]*KVCache, e.Config.MaxBatch)
	chunks := make([][]int, e.Config.MaxBatch)
	counts := make([]int, e.Config.MaxBatch)
	rotation := 0
	accounting := func() {
		e.stats.Reserved.Store(int64(e.pool.Reserved * e.pool.PageElements * 4))
		e.stats.Allocated.Store(int64(e.pool.Allocated * e.pool.PageElements * 4))
	}
	finishRequest := func(r *engineRequest, out Completion, err error) {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrClosed) {
			r.observeCancellation("actor")
		}
		canceled := r.stopObservation()
		outcome := "completed"
		if err != nil {
			_, outcome = inferenceError(err)
		}
		r.log("terminal:" + outcome)
		if err == nil {
			e.stats.Completed.Add(1)
		} else if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrClosed) {
			e.stats.Canceled.Add(1)
		} else if errors.Is(err, ErrBusy) {
			e.stats.Rejected.Add(1)
		} else {
			e.stats.Failed.Add(1)
		}
		if errors.Is(err, ErrSlowConsumer) {
			e.stats.SlowConsumers.Add(1)
		}
		r.release()
		released := time.Now()
		if !canceled.IsZero() {
			e.cancelCompute.observe(max(r.computeDone.Sub(canceled), 0))
			e.cancelRelease.observe(released.Sub(canceled))
		}
		e.stats.Released.Add(1)
		r.log("released")
		if r.task != nil {
			r.task.End()
		}
		// The terminal mailbox is independent of the bounded token mailbox.
		// Neither delivery nor closing it can block the actor on a network reader.
		r.Reply <- engineReply{out, err}
		if r.Events != nil {
			close(r.Events)
		}
		r.cancelContext() // Publish the reply before cleanup can wake Wait.
		close(r.Released)
	}
	finish := func(s *engineSession, out Completion, err error) {
		s.Cache.Close()
		e.stats.Active.Add(-1)
		accounting()
		finishRequest(s.Request, out, err)
	}
	for {
		if e.ctx.Err() != nil {
			e.submitMu.Lock()
			for _, s := range sessions {
				finish(s, Completion{}, ErrClosed)
			}
			for {
				select {
				case r := <-e.queue:
					finishRequest(r, Completion{}, ErrClosed)
				default:
					e.submitMu.Unlock()
					return
				}
			}
		}
		alive := sessions[:0]
		for _, s := range sessions {
			if err := s.Request.Ctx.Err(); err != nil {
				finish(s, Completion{}, err)
			} else {
				alive = append(alive, s)
			}
		}
		clear(sessions[len(alive):])
		sessions = alive
	fill:
		for len(sessions) < e.Config.MaxBatch {
			var r *engineRequest
			if len(sessions) == 0 {
				select {
				case r = <-e.queue:
				case <-e.ctx.Done():
					break fill
				}
			} else {
				select {
				case r = <-e.queue:
				default:
					break fill
				}
			}
			if err := r.Ctx.Err(); err != nil {
				finishRequest(r, Completion{}, err)
				continue
			}
			pages := (len(r.IDs) + r.Sampling.MaxTokens + pageTokens - 1) / pageTokens
			if e.pool.Reserved+pages > e.pool.Limit {
				finishRequest(r, Completion{}, ErrBusy)
				continue
			}
			cache := e.Model.NewCache()
			cache.pool, cache.reserved = &e.pool, pages
			e.pool.Reserved += pages
			sessions = append(sessions, &engineSession{Request: r, Cache: cache, History: append([]int(nil), r.IDs...), RNG: RNG{r.Sampling.Seed}, Text: incrementalText{hf: e.Model.hf != nil}})
			e.stats.Active.Add(1)
			r.log("admitted")
			e.queueLatency.observe(time.Since(r.Enqueued))
			accounting()
		}
		if len(sessions) == 0 {
			continue
		}
		n := len(sessions)
		total := scheduleTokens(sessions, counts, e.Config.TokenBudget, e.Config.PrefillChunk, e.Config.MixedPrefillChunk, rotation%n)
		rotation = (rotation + 1) % n
		for i, s := range sessions {
			caches[i] = s.Cache
			if s.Cursor < len(s.Request.IDs) {
				chunks[i] = s.Request.IDs[s.Cursor : s.Cursor+counts[i]]
			} else {
				chunks[i] = s.Output[len(s.Output)-1:]
			}
		}
		tick := e.stats.Ticks.Load() + 1
		for i, s := range sessions {
			if s.Request.task != nil {
				trace.Logf(s.Request.Ctx, "batch", "%d begin rows=%d", tick, counts[i])
			}
		}
		e.stats.Computing.Store(int64(n))
		computeStart := time.Now()
		var logits []float32
		var err error
		trace.WithRegion(e.ctx, "monolith.batch", func() {
			logits, err = e.Model.prefillBorrowed(e.ctx, caches[:n], chunks[:n], executor, &workspace)
		})
		e.stats.WorkspaceFloatBytes.Store(workspace.floatBytes())
		computeDone := time.Now() // PrefillBatch joins all kernels before returning.
		e.batchCompute.observe(computeDone.Sub(computeStart))
		e.stats.Computing.Store(0)
		for _, s := range sessions {
			s.Request.computeDone = computeDone
			if s.Request.task != nil {
				trace.Logf(s.Request.Ctx, "batch", "%d compute_done", tick)
			}
		}
		clear(caches[:n])
		clear(chunks[:n])
		e.stats.Ticks.Add(1)
		if int64(total) > e.stats.PeakTickTokens.Load() {
			e.stats.PeakTickTokens.Store(int64(total))
		}
		accounting()
		if err != nil {
			if e.ctx.Err() != nil {
				err = ErrClosed
			}
			for _, s := range sessions {
				finish(s, Completion{}, err)
			}
			clear(sessions)
			sessions = sessions[:0]
			continue
		}
		alive = sessions[:0]
		for i, s := range sessions {
			if err = s.Request.Ctx.Err(); err != nil {
				finish(s, Completion{}, err)
				continue
			}
			if s.Cursor < len(s.Request.IDs) {
				s.Cursor += counts[i]
				e.stats.InputTokens.Add(int64(counts[i]))
			}
			if s.Cursor < len(s.Request.IDs) {
				alive = append(alive, s)
				continue
			}
			id, err := sampleWithBlocked(logits[i*e.Model.Config.Vocab:(i+1)*e.Model.Config.Vocab], s.History, s.Request.Sampling, &s.RNG, e.Tokenizer.blockedID())
			if err != nil {
				finish(s, Completion{}, err)
				continue
			}
			now := time.Now()
			if s.LastToken.IsZero() {
				e.firstTokenLatency.observe(now.Sub(s.Request.Started))
			} else {
				e.tokenGap.observe(now.Sub(s.LastToken))
			}
			s.LastToken = now
			e.stats.GeneratedTokens.Add(1)
			s.Output = append(s.Output, id)
			s.History = append(s.History, id)
			reason := ""
			if id == e.Tokenizer.eosID() {
				reason = "eos"
			} else if len(s.Output) == s.Request.Sampling.MaxTokens {
				reason = "length"
			}
			if s.Request.Events != nil {
				event := TokenEvent{TokenID: id, Text: s.Text.Push(e.pieces[id], reason != "")}
				select {
				case s.Request.Events <- event:
				default:
					finish(s, Completion{}, ErrSlowConsumer)
					continue
				}
			}
			if reason != "" {
				finish(s, completion(e.Tokenizer, s.Output, len(s.Request.IDs), reason), nil)
			} else {
				alive = append(alive, s)
			}
		}
		clear(sessions[len(alive):])
		sessions = alive
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func inferenceError(err error) (int, string) {
	switch {
	case errors.Is(err, ErrBusy):
		return http.StatusTooManyRequests, "busy"
	case errors.Is(err, ErrClosed):
		return http.StatusServiceUnavailable, "closed"
	case errors.Is(err, ErrSlowConsumer):
		return http.StatusRequestTimeout, "slow_consumer"
	case errors.Is(err, context.DeadlineExceeded):
		return http.StatusRequestTimeout, "timeout"
	case errors.Is(err, context.Canceled):
		return http.StatusRequestTimeout, "canceled"
	default:
		return http.StatusBadRequest, "invalid_request"
	}
}
func writeInferenceError(w http.ResponseWriter, err error) {
	status, code := inferenceError(err)
	if status == http.StatusTooManyRequests {
		w.Header().Set("Retry-After", "1")
	}
	writeJSON(w, status, map[string]string{"error": err.Error(), "code": code})
}
func (e *Engine) serveStream(w http.ResponseWriter, r *http.Request, ctx context.Context, prompt string, sampling Sampling) {
	if _, ok := w.(http.Flusher); !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "response writer does not support streaming"})
		return
	}
	stream, err := e.Stream(ctx, prompt, sampling)
	if err != nil {
		writeInferenceError(w, err)
		return
	}
	defer stream.Close()
	rc := http.NewResponseController(w)
	write := func(event string, v any) error {
		if err := rc.SetWriteDeadline(time.Now().Add(5 * time.Second)); err != nil && !errors.Is(err, http.ErrNotSupported) {
			return err
		}
		// Bound a blocked network write, not idle prefill/queue time. In
		// HTTP/2 an armed write deadline also resets an otherwise idle stream.
		defer rc.SetWriteDeadline(time.Time{})
		if event == "" {
			if _, err := io.WriteString(w, ": keep-alive\n\n"); err != nil {
				return err
			}
		} else {
			b, err := json.Marshal(v)
			if err != nil {
				return err
			}
			if _, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b); err != nil {
				return err
			}
		}
		return rc.Flush()
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	if err = write("", nil); err != nil {
		return
	}
	heartbeat := time.NewTicker(10 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				out, err := stream.Wait()
				if err != nil {
					_, code := inferenceError(err)
					_ = write("error", map[string]string{"error": err.Error(), "code": code})
				} else {
					_ = write("done", out)
				}
				return
			}
			if err = write("token", event); err != nil {
				return
			}
		case <-heartbeat.C:
			if err = write("", nil); err != nil {
				return
			}
		case <-ctx.Done():
			// Client disconnects terminate delivery; deadlines on connected
			// requests get a terminal error when the socket is still writable.
			if r.Context().Err() == nil {
				_, code := inferenceError(ctx.Err())
				_ = write("error", map[string]string{"error": ctx.Err().Error(), "code": code})
			}
			return
		}
	}
}
func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(405)
			return
		}
		select {
		case <-e.done:
			writeJSON(w, 503, map[string]string{"status": "closed"})
		default:
			writeJSON(w, 200, map[string]any{"status": "ok", "parameters": e.Model.parameterCount(), "stats": e.Stats()})
		}
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(405)
			return
		}
		s := e.Stats()
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(w, "monolith_submitted_total %d\nmonolith_completed_total %d\nmonolith_canceled_total %d\nmonolith_failed_total %d\nmonolith_rejected_total %d\nmonolith_active %d\nmonolith_queue_depth %d\nmonolith_kv_reserved_bytes %d\nmonolith_kv_allocated_bytes %d\n", s.Submitted, s.Completed, s.Canceled, s.Failed, s.Rejected, s.Active, s.Queue, s.KVReservedBytes, s.KVAllocatedBytes)
		fmt.Fprintf(w, "monolith_input_tokens_total %d\nmonolith_generated_tokens_total %d\nmonolith_ticks_total %d\nmonolith_slow_consumers_total %d\nmonolith_peak_tick_tokens %d\n", s.InputTokens, s.GeneratedTokens, s.Ticks, s.SlowConsumers, s.PeakTickTokens)
		fmt.Fprintf(w, "monolith_released_total %d\nmonolith_computing_sessions %d\nmonolith_admission_used %d\n", s.Released, s.Computing, s.AdmissionUsed)
		fmt.Fprintf(w, "monolith_kernel_workers %d\nmonolith_workspace_float_bytes %d\n", s.KernelWorkers, s.WorkspaceFloatBytes)
		e.queueLatency.write(w, "monolith_queue_wait_seconds")
		e.firstTokenLatency.write(w, "monolith_first_token_seconds")
		e.tokenGap.write(w, "monolith_token_gap_seconds")
		e.batchCompute.write(w, "monolith_batch_compute_seconds")
		e.cancelCompute.write(w, "monolith_cancel_observed_to_compute_done_seconds")
		e.cancelRelease.write(w, "monolith_cancel_observed_to_release_seconds")
		var mem runtime.MemStats
		runtime.ReadMemStats(&mem)
		fmt.Fprintf(w, "monolith_heap_alloc_bytes %d\nmonolith_heap_sys_bytes %d\nmonolith_gc_cycles_total %d\nmonolith_gc_pause_seconds_total %.9f\nmonolith_goroutines %d\n", mem.HeapAlloc, mem.HeapSys, mem.NumGC, float64(mem.PauseTotalNs)/1e9, runtime.NumGoroutine())
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(405)
			return
		}
		select {
		case e.httpSlots <- struct{}{}:
			defer func() { <-e.httpSlots }()
		default:
			e.stats.Rejected.Add(1)
			writeInferenceError(w, ErrBusy)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
		if err != nil {
			writeJSON(w, 413, map[string]string{"error": "request body exceeds 1 MiB"})
			return
		}
		req := struct {
			Prompt   *string       `json:"prompt"`
			Messages []ChatMessage `json:"messages"`
			Stream   bool          `json:"stream"`
			Sampling
		}{Sampling: defaultSampling()}
		if err = strictJSON(body, &req); err != nil {
			writeInferenceError(w, err)
			return
		}
		prompt := ""
		if req.Prompt != nil {
			prompt = *req.Prompt
		}
		if req.Messages != nil {
			t, ok := e.Tokenizer.(*hfTokenizer)
			if !ok || req.Prompt != nil {
				writeInferenceError(w, errors.New("messages require HF chat support and cannot be combined with prompt"))
				return
			}
			prompt, err = t.formatChat(req.Messages)
			if err != nil {
				writeInferenceError(w, err)
				return
			}
		}
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
		defer cancel()
		if req.Stream {
			e.serveStream(w, r, ctx, prompt, req.Sampling)
			return
		}
		out, err := e.Generate(ctx, prompt, req.Sampling)
		if err != nil {
			writeInferenceError(w, err)
			return
		}
		writeJSON(w, 200, out)
	})
	return mux
}

// The CLI is part of the same binary. No Python, model download, or service is
// involved in training; only explicitly supplied text (or the demo fixture).
func readText(path string, limit int64) (string, error) {
	if path == "" {
		return "", errors.New("-data is required")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return "", err
	}
	if int64(len(b)) > limit {
		return "", fmt.Errorf("input exceeds %d bytes", limit)
	}
	if len(b) == 0 {
		return "", errors.New("empty input")
	}
	return string(b), nil
}
func readJSONFile(path string, v any) error {
	s, err := readText(path, 4<<20)
	if err != nil {
		return err
	}
	return strictJSON([]byte(s), v)
}
func writeJSONFile(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0644)
}
func flags(name string) *flag.FlagSet {
	f := flag.NewFlagSet(name, flag.ContinueOnError)
	f.SetOutput(os.Stderr)
	return f
}
func parse(f *flag.FlagSet, args []string) error {
	if err := f.Parse(args); err != nil {
		return err
	}
	if f.NArg() > 0 {
		return fmt.Errorf("unexpected positional arguments: %v", f.Args())
	}
	return nil
}

// The source wrapper and direct commands share the action's real flag parser.
// Preflight stops before files, stdin, network, worker changes or model loading.
func parseInferenceAction(f *flag.FlagSet, args []string, validateOnly bool) error {
	if err := parse(f, args); err != nil {
		return err
	}
	if validateOnly {
		for _, name := range []string{"model", "hf-model", "context", "weight-mib", "weight-format"} {
			if flagSet(f, name) {
				return errors.New("put model source and HF sizing flags before the infer local action")
			}
		}
	}
	return validateInferenceFlags(f)
}
func validateInferenceFlags(f *flag.FlagSet) error {
	if v := f.Lookup("weight-format"); v != nil && !validHFWeightFormat(v.Value.String()) {
		return errors.New("weight-format must be f32, q8, q4 or q4-mixed")
	}
	for _, bound := range []struct {
		name   string
		lo, hi int64
	}{{"threads", 1, 1024}, {"context", 0, 32768}, {"weight-mib", 1, 1 << 20}, {"kv-mib", 1, 1 << 20}} {
		if v := f.Lookup(bound.name); v != nil {
			n, err := strconv.ParseInt(v.Value.String(), 10, 64)
			if err != nil || n < bound.lo || n > bound.hi {
				return fmt.Errorf("-%s must be %d..%d", bound.name, bound.lo, bound.hi)
			}
		}
	}
	value := func(name string) string {
		if v := f.Lookup(name); v != nil {
			return v.Value.String()
		}
		return ""
	}
	if value("prompt-file") != "" && (flagSet(f, "prompt") || value("messages") != "") {
		return errors.New("-prompt-file is mutually exclusive with -prompt/-messages")
	}
	if value("messages") != "" && (flagSet(f, "prompt") || value("chat") == "true") {
		return errors.New("-messages is mutually exclusive with -prompt/-chat")
	}
	if len(value("prompt")) > 1<<20 {
		return errors.New("prompt exceeds 1 MiB")
	}
	if system := value("system"); len(system) > 1<<20 || !utf8.ValidString(system) {
		return errors.New("system message must be valid UTF-8 and at most 1 MiB")
	}
	return nil
}

func runInferenceAction(ctx context.Context, action string, args []string, validateOnly bool) error {
	switch action {
	case "check-hf":
		return runCheckHF(args, validateOnly)
	case "inspect-hf":
		return runInspectHF(args, validateOnly)
	case "tokenize-hf":
		return runTokenizeHF(ctx, args, validateOnly)
	case "infer-hf":
		return runInferHF(ctx, args, validateOnly)
	case "generate":
		return runGenerate(ctx, args, validateOnly)
	case "chat":
		return runChat(ctx, args, validateOnly)
	case "serve":
		return runServe(ctx, args, validateOnly)
	default:
		return errors.New("unknown inference action")
	}
}
func threadFlag(f *flag.FlagSet) *int {
	return f.Int("threads", min(4, runtime.GOMAXPROCS(0)), "maximum CPU workers (GOMAXPROCS)")
}
func setThreads(n int) error {
	if n < 1 || n > 1024 {
		return errors.New("threads must be 1..1024")
	}
	runtime.GOMAXPROCS(n)
	return nil
}
func samplingFlags(f *flag.FlagSet, s *Sampling) {
	f.IntVar(&s.MaxTokens, "max-tokens", s.MaxTokens, "maximum generated tokens")
	f.Float64Var(&s.Temperature, "temperature", s.Temperature, "temperature; 0 means greedy")
	f.IntVar(&s.TopK, "top-k", s.TopK, "top-k; 0 disables")
	f.Float64Var(&s.TopP, "top-p", s.TopP, "nucleus probability mass")
	f.Float64Var(&s.RepeatPenalty, "repeat-penalty", s.RepeatPenalty, "penalty for already seen tokens (>=1)")
	f.Uint64Var(&s.Seed, "seed", s.Seed, "sampling seed")
}
func readTokenizer(tokenizerPath string) (Tokenizer, error) {
	tok := Tokenizer{}
	if tokenizerPath != "" {
		if err := readJSONFile(tokenizerPath, &tok); err != nil {
			return tok, err
		}
		if err := tok.Validate(); err != nil {
			return tok, err
		}
	}
	return tok, nil
}
func modelConfig(presetName, configPath string, tok Tokenizer) (Config, error) {
	c, err := preset(presetName)
	if err != nil {
		return c, err
	}
	if configPath != "" {
		if err = readJSONFile(configPath, &c); err != nil {
			return c, err
		}
	}
	c.Vocab = tok.Vocab()
	return c, c.Validate()
}
func ensureMoments(m *Model) {
	for _, p := range m.Params {
		if len(p.M) == 0 {
			p.M = make([]float32, len(p.Data))
			p.V = make([]float32, len(p.Data))
		}
	}
}

// Telemetry belongs to one invocation, never to TrainState or a checkpoint.
// The training goroutine owns phases/JSON; IO workers only touch atomic counters.
type trainingPhases struct {
	Input      int64 `json:"input_ns"`
	ZeroGrad   int64 `json:"zero_grad_ns"`
	Forward    int64 `json:"forward_ns"`
	Backward   int64 `json:"backward_ns"`
	Optimizer  int64 `json:"optimizer_ns"`
	Evaluation int64 `json:"evaluation_ns"`
	Checkpoint int64 `json:"checkpoint_ns"`
}
type trainingRun struct {
	Config             Config              `json:"config"`
	Spec               TrainSpec           `json:"spec"`
	Source             EvaluationSource    `json:"source"`
	TokenizerSHA256    string              `json:"tokenizer_sha256"`
	ResumeSHA256       string              `json:"resume_sha256,omitempty"`
	InitializationSeed *uint64             `json:"initialization_seed,omitempty"`
	GoVersion          string              `json:"go_version"`
	Threads            int                 `json:"threads"`
	DataWorkers        int                 `json:"data_workers"`
	Prefetch           int                 `json:"prefetch"`
	StopAfter          int                 `json:"stop_after"`
	EvalEvery          int                 `json:"eval_every"`
	EvalBatches        int                 `json:"eval_batches"`
	SaveEvery          int                 `json:"save_every"`
	LogEvery           int                 `json:"log_every"`
	MemoryPlan         *trainingMemoryPlan `json:"memory_plan,omitempty"`
}
type trainingEvent struct {
	Format     string         `json:"format"`
	Event      string         `json:"event"`
	Step       int            `json:"step"`
	TokensSeen int64          `json:"tokens_seen"`
	Cursor     *DatasetCursor `json:"cursor,omitempty"`
	ElapsedNS  int64          `json:"elapsed_ns"`
	Phases     trainingPhases `json:"phases"`
	ReadNS     int64          `json:"prefetch_read_ns"`
	ReadTokens int64          `json:"prefetch_read_tokens"`
	AllocBytes uint64         `json:"alloc_bytes"`
	HeapBytes  uint64         `json:"heap_bytes"`
	GCs        uint32         `json:"gc_cycles"`
	Loss       *float64       `json:"loss,omitempty"`
	GradNorm   *float64       `json:"grad_norm,omitempty"`
	Targets    int            `json:"targets,omitempty"`
	Status     string         `json:"status,omitempty"`
	Run        *trainingRun   `json:"run,omitempty"`
}
type telemetryOptions struct{ Metrics, CPUProfile, AllocProfile, Checkpoint string }

// The CPU profiler writes asynchronously. Read err only after StopCPUProfile
// has joined its writer; the runtime otherwise discards output write errors.
type profileWriter struct {
	io.Writer
	err error
}

func (w *profileWriter) Write(p []byte) (int, error) {
	n, err := w.Writer.Write(p)
	if err != nil && w.err == nil {
		w.err = err
	}
	return n, err
}

type trainingObserver struct {
	encoder            *json.Encoder
	files              []*os.File
	cpu                bool
	started            bool
	cpuWriter          *profileWriter
	alloc              *os.File
	start              time.Time
	baseline           runtime.MemStats
	phases             trainingPhases
	readNS, readTokens atomic.Int64
	run                trainingRun
}

// Resolve parent symlinks even when the final checkpoint directory does not yet
// exist. SaveCheckpoint replaces a directory entry, not a final symlink target.
func outputIdentity(path string) (string, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	parent, base := filepath.Dir(path), filepath.Base(path)
	resolved, err := filepath.EvalSymlinks(parent)
	if errors.Is(err, os.ErrNotExist) && parent != path {
		resolved, err = outputIdentity(parent)
	}
	if err != nil {
		return "", err
	}
	return filepath.Join(resolved, base), nil
}
func openTrainingObserver(o telemetryOptions) (*trainingObserver, error) {
	if o.Metrics == "" && o.CPUProfile == "" && o.AllocProfile == "" {
		return nil, nil
	}
	paths := []string{o.Metrics, o.CPUProfile, o.AllocProfile}
	seen := make(map[string]bool)
	checkpoint, err := outputIdentity(o.Checkpoint)
	if err != nil {
		return nil, err
	}
	seen[checkpoint] = true
	for _, path := range paths {
		if path == "" {
			continue
		}
		id, err := outputIdentity(path)
		if err != nil {
			return nil, err
		}
		if seen[id] {
			return nil, errors.New("telemetry outputs must differ from each other and the checkpoint")
		}
		seen[id] = true
	}
	observer := &trainingObserver{}
	success := false
	defer func() {
		if !success {
			for _, f := range observer.files {
				_ = f.Close()
				_ = os.Remove(f.Name())
			}
		}
	}()
	for i, path := range paths {
		if path == "" {
			continue
		}
		// Never truncate an input, existing receipt or existing profile.
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
		if err != nil {
			return nil, err
		}
		observer.files = append(observer.files, f)
		switch i {
		case 0:
			observer.encoder = json.NewEncoder(f)
		case 1: // Start only after every requested file has been created.
		case 2:
			observer.alloc = f
		}
	}
	// String canonicalization alone cannot detect case-folding filesystems.
	if info, err := os.Stat(o.Checkpoint); err == nil {
		for _, f := range observer.files {
			created, err := f.Stat()
			if err != nil {
				return nil, err
			}
			if os.SameFile(info, created) {
				return nil, errors.New("telemetry output aliases the checkpoint")
			}
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if o.CPUProfile != "" {
		for _, f := range observer.files {
			if f.Name() == o.CPUProfile {
				observer.cpuWriter = &profileWriter{Writer: f}
				if err := pprof.StartCPUProfile(observer.cpuWriter); err != nil {
					return nil, err
				}
				observer.cpu = true
			}
		}
	}
	observer.start = time.Now()
	runtime.ReadMemStats(&observer.baseline)
	success = true
	return observer, nil
}
func (o *trainingObserver) close() error {
	if o == nil {
		return nil
	}
	if o.cpu {
		pprof.StopCPUProfile()
	}
	var err error
	if o.cpuWriter != nil {
		err = o.cpuWriter.err
	}
	if o.alloc != nil {
		// Flush recent samples after the measured invocation, including dead
		// training tapes. alloc_space is process-wide, not live tensor memory.
		runtime.GC()
		err = errors.Join(err, pprof.Lookup("allocs").WriteTo(o.alloc, 0))
	}
	for _, f := range o.files {
		err = errors.Join(err, f.Close())
	}
	if !o.started {
		// CLI/source settings can still be rejected while constructing the
		// loop. Those attempts must not reserve the next corrected run's paths.
		for _, f := range o.files {
			err = errors.Join(err, os.Remove(f.Name()))
		}
	}
	return err
}
func (o *trainingObserver) phase(ctx context.Context, name string, f func() error) error {
	if o == nil {
		return f()
	}
	start := time.Now()
	var err error
	if o.cpu {
		pprof.Do(ctx, pprof.Labels("phase", name), func(context.Context) { err = f() })
	} else {
		err = f()
	}
	n := time.Since(start).Nanoseconds()
	switch name {
	case "input":
		o.phases.Input += n
	case "zero_grad":
		o.phases.ZeroGrad += n
	case "forward":
		o.phases.Forward += n
	case "backward":
		o.phases.Backward += n
	case "optimizer":
		o.phases.Optimizer += n
	case "evaluation":
		o.phases.Evaluation += n
	case "checkpoint":
		o.phases.Checkpoint += n
	}
	return err
}
func (o *trainingObserver) reader(read tokenReadFunc) tokenReadFunc {
	if o == nil {
		return read
	}
	return func(ctx context.Context, shard int, offset int64, dst []int) error {
		start := time.Now()
		var err error
		if o.cpu {
			pprof.Do(ctx, pprof.Labels("phase", "prefetch_read"), func(context.Context) { err = read(ctx, shard, offset, dst) })
		} else {
			err = read(ctx, shard, offset, dst)
		}
		o.readNS.Add(time.Since(start).Nanoseconds())
		if err == nil {
			o.readTokens.Add(int64(len(dst)))
		}
		return err
	}
}
func (o *trainingObserver) emit(s *TrainState, e trainingEvent) error {
	if o == nil || o.encoder == nil {
		return nil
	}
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	e.Format, e.Step, e.TokensSeen = "monolith-train-v1", s.Step, s.TokensSeen
	e.ElapsedNS, e.Phases = time.Since(o.start).Nanoseconds(), o.phases
	e.ReadNS, e.ReadTokens = o.readNS.Load(), o.readTokens.Load()
	e.AllocBytes, e.HeapBytes, e.GCs = mem.TotalAlloc-o.baseline.TotalAlloc, mem.HeapAlloc, mem.NumGC-o.baseline.NumGC
	if s.Dataset != nil {
		cursor := s.Dataset.Cursor
		e.Cursor = &cursor
	}
	if e.Event == "start" {
		e.Run = &o.run
	}
	if err := o.encoder.Encode(e); err != nil {
		return fmt.Errorf("write training metrics: %w", err)
	}
	return nil
}

type loopOptions struct {
	Out                                                    string
	StopAfter, SaveEvery, EvalEvery, EvalBatches, LogEvery int
	Observer                                               *trainingObserver
}

func (o loopOptions) Validate() error {
	if o.StopAfter < 0 || o.SaveEvery < 0 || o.EvalEvery < 0 || o.EvalBatches < 0 || o.LogEvery < 1 {
		return errors.New("invalid logging/save/evaluation interval")
	}
	return nil
}

func trainLoop(ctx context.Context, m *Model, tok Tokenizer, s *TrainState, train, val []int, o loopOptions) error {
	if err := s.Validate(m.Config); err != nil {
		return err
	}
	w, err := newBackwardWorkspace(m, s.Spec.Batch, s.Spec.Seq)
	if err != nil {
		return err
	}
	return trainingLoop(ctx, m, tok, s, int64(len(train)), int64(len(val)), o,
		func() (float64, float64, error) { return observedTrainUpdate(ctx, m, s, train, o.Observer, w) },
		func() (float64, int, error) { return evaluate(ctx, m, val, s.Spec.Seq, o.EvalBatches) })
}
func trainDatasetLoop(ctx context.Context, m *Model, tok Tokenizer, s *TrainState, d *TokenDataset, workers, depth int, o loopOptions) error {
	if err := s.Validate(m.Config); err != nil {
		return err
	}
	if err := validateDatasetResume(d, tok, s); err != nil {
		return err
	}
	w, err := newBackwardWorkspace(m, s.Spec.Batch, s.Spec.Seq)
	if err != nil {
		return err
	}
	c := PrefetchConfig{Batch: s.Spec.Batch, Seq: s.Spec.Seq, Workers: workers, Depth: depth, Seed: s.Dataset.OrderSeed, Shuffle: true}
	p, err := newTokenPrefetch(ctx, d, "train", c, s.Dataset.Cursor, o.Observer.reader(d.ReadTokens))
	if err != nil {
		return err
	}
	defer p.Close()
	counts := make(map[string]int64)
	for _, shard := range d.manifest.Shards {
		if counts[shard.Split] > math.MaxInt64-shard.Tokens {
			return errors.New("dataset token count overflows int64")
		}
		counts[shard.Split] += shard.Tokens
	}
	return trainingLoop(ctx, m, tok, s, counts["train"], counts["validation"], o,
		func() (float64, float64, error) { return observedDatasetUpdate(ctx, m, s, p, o.Observer, w) },
		func() (float64, int, error) { return evaluateDataset(ctx, m, d, s.Spec.Seq, o.EvalBatches) })
}
func trainingLoop(ctx context.Context, m *Model, tok Tokenizer, s *TrainState, trainTokens, valTokens int64, o loopOptions, update func() (float64, float64, error), eval func() (float64, int, error)) (result error) {
	if err := s.Validate(m.Config); err != nil {
		return err
	}
	if err := o.Validate(); err != nil {
		return err
	}
	if o.Observer != nil {
		o.Observer.started = true
	}
	ensureMoments(m)
	observer := o.Observer
	defer func() {
		status := "completed"
		if s.Step < s.Spec.Steps {
			status = "stopped"
		}
		if ctx.Err() != nil {
			status = "canceled"
		}
		if result != nil {
			status = "failed"
		}
		result = errors.Join(result, observer.emit(s, trainingEvent{Event: "end", Status: status}))
	}()
	save := func() error {
		return observer.phase(ctx, "checkpoint", func() error { return SaveCheckpoint(o.Out, m, tok, s) })
	}
	// An output error at a completed boundary stops training and saves that
	// boundary. No logger goroutine, unbounded queue or silently dropped record.
	record := func(e trainingEvent) error {
		if err := observer.emit(s, e); err != nil {
			return errors.Join(err, save())
		}
		return nil
	}
	if err := record(trainingEvent{Event: "start"}); err != nil {
		return err
	}
	observedEval := func() (float64, int, error) {
		var value float64
		var count int
		err := observer.phase(ctx, "evaluation", func() (err error) { value, count, err = eval(); return err })
		if err == nil {
			err = record(trainingEvent{Event: "evaluation", Loss: &value, Targets: count})
		}
		return value, count, err
	}
	initial, nt, err := observedEval()
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "parameters=%d fp32_weights_mib=%.2f train_tokens=%d val_tokens=%d workers=%d\n", m.Config.ParameterCount(), float64(m.Config.ParameterCount()*4)/(1<<20), trainTokens, valTokens, runtime.GOMAXPROCS(0))
	fmt.Fprintf(os.Stderr, "step=%d val_loss=%.6f val_targets=%d\n", s.Step, initial, nt)
	start := time.Now()
	initialTokens := s.TokensSeen
	target := s.Spec.Steps
	if o.StopAfter > 0 && o.StopAfter < target-s.Step {
		target = s.Step + o.StopAfter
	}
	for s.Step < target {
		if ctx.Err() != nil {
			break
		}
		loss, norm, err := update()
		if err != nil {
			if ctx.Err() != nil && errors.Is(err, ctx.Err()) {
				break
			}
			return err
		}
		if s.Step%o.LogEvery == 0 || s.Step == target {
			fmt.Fprintf(os.Stderr, "step=%d/%d loss=%.6f lr=%.7f grad_norm=%.4f tokens_per_sec=%.1f\n", s.Step, s.Spec.Steps, loss, s.Spec.LearningRate(s.Step), norm, float64(s.TokensSeen-initialTokens)/time.Since(start).Seconds())
			if err = record(trainingEvent{Event: "update", Loss: &loss, GradNorm: &norm}); err != nil {
				return err
			}
		}
		if o.EvalEvery > 0 && s.Step%o.EvalEvery == 0 {
			value, count, e := observedEval()
			if e != nil {
				if ctx.Err() != nil && errors.Is(e, ctx.Err()) {
					break
				}
				return e
			}
			fmt.Fprintf(os.Stderr, "step=%d val_loss=%.6f val_targets=%d\n", s.Step, value, count)
		}
		if o.SaveEvery > 0 && s.Step%o.SaveEvery == 0 {
			if err = save(); err != nil {
				return err
			}
			if err = record(trainingEvent{Event: "checkpoint"}); err != nil {
				return err
			}
		}
	}
	if err = save(); err != nil {
		return err
	}
	if err = record(trainingEvent{Event: "checkpoint"}); err != nil {
		return err
	}
	if ctx.Err() == nil {
		value, count, e := observedEval()
		if e != nil {
			return e
		}
		fmt.Fprintf(os.Stderr, "final_step=%d val_loss=%.6f val_targets=%d checkpoint=%s\n", s.Step, value, count, o.Out)
	} else {
		fmt.Fprintf(os.Stderr, "interrupted; checkpoint saved at completed update %d: %s\n", s.Step, o.Out)
	}
	return nil
}
func runTrain(ctx context.Context, args []string) (result error) {
	f := flags("train")
	spec := defaultTrainSpec()
	data := f.String("data", "", "UTF-8 or arbitrary byte text file (max 64 MiB)")
	datasetPath := f.String("dataset", "", "prepared manifest.json; mutually exclusive with -data")
	dataWorkers := f.Int("data-workers", 2, "dataset IO workers (1..prefetch)")
	prefetch := f.Int("prefetch", 4, "outstanding dataset microbatches (1..64)")
	memoryMiB := f.Int64("memory-mib", 0, "major-array/input working-set budget in MiB; 0 disables this extra cap (not RSS)")
	memoryPlanOnly := f.Bool("memory-plan", false, "print JSON memory plan and exit before tensor allocation/training")
	out := f.String("out", "runs/model.mglm", "atomic checkpoint destination")
	resume := f.String("resume", "", "resume a checkpoint with its original schedule")
	pre := f.String("preset", "tiny", "model preset: demo, tiny, small, base")
	config := f.String("config", "", "model configuration JSON (overrides preset fields)")
	tokenizer := f.String("tokenizer", "", "BPE tokenizer JSON; default is byte vocabulary")
	seed := f.Uint64("seed", 1, "initialization and data sampling seed")
	threads := threadFlag(f)
	f.IntVar(&spec.Steps, "steps", spec.Steps, "total optimizer steps in saved LR schedule")
	f.IntVar(&spec.Batch, "batch", spec.Batch, "sequences per microbatch")
	f.IntVar(&spec.Seq, "seq", spec.Seq, "tokens per sequence")
	f.IntVar(&spec.Accum, "accum", spec.Accum, "gradient accumulation microbatches")
	f.Float64Var(&spec.LR, "lr", spec.LR, "peak learning rate")
	f.Float64Var(&spec.MinLR, "min-lr", spec.MinLR, "final learning rate")
	f.IntVar(&spec.Warmup, "warmup", spec.Warmup, "warmup steps")
	f.Float64Var(&spec.WeightDecay, "weight-decay", spec.WeightDecay, "decoupled matrix weight decay")
	f.Float64Var(&spec.Clip, "clip", spec.Clip, "global gradient norm limit")
	f.Float64Var(&spec.ValFraction, "val-fraction", spec.ValFraction, "held-out trailing raw byte fraction")
	o := loopOptions{}
	telemetry := telemetryOptions{}
	f.StringVar(&telemetry.Metrics, "metrics", "", "new JSONL file for cumulative phase timing and training events")
	f.StringVar(&telemetry.CPUProfile, "cpu-profile", "", "new Go pprof CPU profile (after source/model loading)")
	f.StringVar(&telemetry.AllocProfile, "alloc-profile", "", "new Go pprof sampled allocation profile (process-wide)")
	f.IntVar(&o.StopAfter, "stop-after", 0, "stop after this many additional updates, retaining full schedule")
	f.IntVar(&o.SaveEvery, "save-every", 50, "checkpoint interval; 0 disables periodic saves")
	f.IntVar(&o.EvalEvery, "eval-every", 25, "validation interval; 0 disables periodic validation")
	f.IntVar(&o.EvalBatches, "eval-batches", 8, "deterministic validation windows; 0 means all")
	f.IntVar(&o.LogEvery, "log-every", 10, "training log interval")
	if err := parse(f, args); err != nil {
		return err
	}
	budget, err := memoryBudgetBytes(*memoryMiB)
	if err != nil {
		return err
	}
	if err := o.Validate(); err != nil {
		return err
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	if (*data == "") == (*datasetPath == "") {
		return errors.New("provide exactly one of -data or -dataset")
	}
	visited := make(map[string]bool)
	f.Visit(func(v *flag.Flag) { visited[v.Name] = true })
	if *datasetPath != "" && (visited["tokenizer"] || visited["val-fraction"]) {
		return errors.New("-dataset supplies its own tokenizer and validation split")
	}
	if *datasetPath == "" && (visited["data-workers"] || visited["prefetch"]) {
		return errors.New("-data-workers and -prefetch require -dataset")
	}
	if *resume != "" {
		allowed := map[string]bool{"resume": true, "data": true, "dataset": true, "data-workers": true, "prefetch": true, "out": true, "threads": true, "stop-after": true, "save-every": true, "eval-every": true, "eval-batches": true, "log-every": true, "metrics": true, "cpu-profile": true, "alloc-profile": true, "memory-mib": true, "memory-plan": true}
		var bad string
		f.Visit(func(v *flag.Flag) {
			if !allowed[v.Name] {
				bad = v.Name
			}
		})
		if bad != "" {
			return fmt.Errorf("-%s cannot override a resumed run; its schedule/tokenizer/model are saved", bad)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	var text string
	var dataset *TokenDataset
	if *datasetPath != "" {
		dataset, err = OpenTokenDataset(ctx, *datasetPath)
	} else {
		text, err = readText(*data, 64<<20)
	}
	if err != nil {
		return err
	}
	o.Out = *out
	var c Config
	var tok Tokenizer
	var state *TrainState
	var checkpoint *os.File
	var checked checkedCheckpoint
	if *resume != "" {
		checkpoint, err = os.Open(*resume)
		if err != nil {
			return err
		}
		defer func() {
			if checkpoint != nil {
				_ = checkpoint.Close()
			}
		}()
		checked, err = readCheckpointMetadata(checkpoint)
		if err != nil {
			return err
		}
		c, tok, state = checked.meta.Config, checked.meta.Tokenizer, checked.meta.Training
		if state == nil {
			return errors.New("checkpoint has no training state")
		}
		if (state.Dataset == nil) != (dataset == nil) {
			return errors.New("resume must retain the original data source kind (-data or -dataset)")
		}
		if dataset != nil {
			if err = validateDatasetResume(dataset, tok, state); err != nil {
				return err
			}
		} else if state.CorpusSHA256 != corpusHash(text) {
			return errors.New("resume corpus SHA-256 differs from original")
		}
		if state.Step >= state.Spec.Steps {
			return errors.New("saved training schedule is already complete")
		}
	} else {
		if dataset != nil {
			tok = dataset.manifest.Tokenizer
		} else {
			tok, err = readTokenizer(*tokenizer)
			if err != nil {
				return err
			}
		}
		c, err = modelConfig(*pre, *config, tok)
		if err != nil {
			return err
		}
		state = &TrainState{Spec: spec, RNG: RNG{*seed ^ 0xd1b54a32d192ed03}, CorpusSHA256: corpusHash(text)}
		if dataset != nil {
			state.CorpusSHA256 = dataset.id
			state.Dataset = datasetTrainingState(dataset, state.RNG.State)
		}
	}
	input := trainingMemoryInput{Kind: "text", TextBytes: int64(len(text))}
	if dataset != nil {
		input = trainingMemoryInput{Kind: "dataset", Shards: len(dataset.manifest.Shards), Workers: *dataWorkers, Depth: *prefetch}
	}
	plan, err := newTrainingMemoryPlan(c, state.Spec, input, budget)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if *memoryPlanOnly {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(plan); err != nil {
			return err
		}
		return plan.checkBudget()
	}
	if err := plan.checkBudget(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "memory plan: %.2f MiB managed working set; not process RSS (-memory-mib=%d)\n", float64(plan.RequiredBytes)/(1<<20), *memoryMiB)
	var train, val []int
	if dataset == nil {
		train, val, err = splitCorpus(tok, text, state.Spec.ValFraction, state.Spec.Seq)
		if err != nil {
			return err
		}
	}
	var m *Model
	if checkpoint != nil {
		m, err = loadCheckpointTensors(checkpoint, checked)
		err = errors.Join(err, checkpoint.Close())
		checkpoint = nil
	} else {
		m, err = NewModel(c, *seed)
	}
	if err != nil {
		return err
	}
	telemetry.Checkpoint = o.Out
	o.Observer, err = openTrainingObserver(telemetry)
	if err != nil {
		return err
	}
	defer func() { result = errors.Join(result, o.Observer.close()) }()
	if observer := o.Observer; observer != nil {
		observer.run = trainingRun{Config: m.Config, Spec: state.Spec, Source: EvaluationSource{Kind: "text", SHA256: state.CorpusSHA256}, TokenizerSHA256: tokenizerHash(tok), ResumeSHA256: checked.identity, MemoryPlan: &plan, GoVersion: runtime.Version(), Threads: runtime.GOMAXPROCS(0)}
		observer.run.StopAfter, observer.run.EvalEvery, observer.run.EvalBatches, observer.run.SaveEvery, observer.run.LogEvery = o.StopAfter, o.EvalEvery, o.EvalBatches, o.SaveEvery, o.LogEvery
		if *resume == "" {
			observer.run.InitializationSeed = seed
		}
		if dataset != nil {
			observer.run.Source.Kind, observer.run.Source.Split = "dataset", "train"
			observer.run.DataWorkers, observer.run.Prefetch = *dataWorkers, *prefetch
		}
	}
	if dataset != nil {
		return trainDatasetLoop(ctx, m, tok, state, dataset, *dataWorkers, *prefetch, o)
	}
	return trainLoop(ctx, m, tok, state, train, val, o)
}

const demoCorpus = "the spiral remembers the rain.\nthe cat follows the spiral.\nthe rain returns to the sea.\nthe sea reflects the moon.\nthe moon lights the garden.\nthe garden shelters the cat.\nthe cat listens to the rain.\nthe spiral returns to the garden.\n"

func runDemo(ctx context.Context, args []string) error {
	f := flags("demo")
	steps := f.Int("steps", 500, "optimizer steps")
	out := f.String("out", "runs/demo.mglm", "checkpoint destination")
	threads := threadFlag(f)
	if err := parse(f, args); err != nil {
		return err
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	c, _ := preset("demo")
	m, err := NewModel(c, 7)
	if err != nil {
		return err
	}
	tok := Tokenizer{}
	spec := defaultTrainSpec()
	spec.Steps = *steps
	spec.Warmup = min(5, *steps)
	spec.Seq = 96
	spec.LR = .005
	spec.MinLR = .001
	spec.Batch = 2
	text := strings.Repeat(demoCorpus, 80)
	state := &TrainState{Spec: spec, RNG: RNG{77}, CorpusSHA256: corpusHash(text)}
	train, val, err := splitCorpus(tok, text, spec.ValFraction, spec.Seq)
	if err != nil {
		return err
	}
	if err = trainLoop(ctx, m, tok, state, train, val, loopOptions{Out: *out, EvalEvery: 25, EvalBatches: 8, LogEvery: 10}); err != nil {
		return err
	}
	if ctx.Err() != nil {
		return nil
	}
	s := defaultSampling()
	s.Temperature = 0
	s.RepeatPenalty = 1
	s.MaxTokens = 80
	result, err := Generate(ctx, m, tok, "the spiral", s)
	if err != nil {
		return err
	}
	fmt.Println("the spiral" + result.Text)
	return nil
}
func runGenerate(ctx context.Context, args []string, validateOnly bool) error {
	f := flags("generate")
	model := f.String("model", "runs/model.mglm", "checkpoint")
	dir, limit, budget, format := hfFlags(f)
	chat, messages := chatFlags(f)
	prompt := f.String("prompt", "", "text prompt")
	f.String("prompt-file", "", "prompt file; - reads stdin (at most 1 MiB)")
	asJSON := f.Bool("json", false, "emit structured completion")
	threads := threadFlag(f)
	s := defaultSampling()
	samplingFlags(f, &s)
	if err := parseInferenceAction(f, args, validateOnly); err != nil {
		return err
	}
	if err := s.Validate(Config{Context: 32768}); err != nil {
		return err
	}
	if validateOnly {
		return nil
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	m, tok, err := loadInference(ctx, f, *model, *dir, *limit, *budget, *format)
	if err != nil {
		return err
	}
	m.DropOptimizer()
	formatted, err := inferencePrompt(ctx, f, tok, *prompt, *chat, *messages)
	if err != nil {
		return err
	}
	out, err := Generate(ctx, m, tok, formatted, s)
	if err != nil {
		return err
	}
	if *asJSON {
		return json.NewEncoder(os.Stdout).Encode(out)
	}
	if *chat || *messages != "" {
		fmt.Println(out.Text)
	} else {
		fmt.Print(formatted + out.Text + "\n")
	}
	return nil
}

// chatLoop owns only bounded text history. The engine retains immutable model
// weights and workers across turns; every stream is released before the next.
func chatLoop(ctx context.Context, e *Engine, tok *hfTokenizer, system string, s Sampling, in io.Reader, out, status io.Writer) error {
	out, closeOutput := cliOutput(ctx, out)
	defer closeOutput()
	status, closeStatus := cliOutput(ctx, status)
	defer closeStatus()
	initial := []ChatMessage{}
	if system != "" {
		initial = append(initial, ChatMessage{Role: "system", Content: system})
	}
	history := append([]ChatMessage(nil), initial...)
	scan := bufio.NewScanner(in)
	scan.Buffer(make([]byte, 4096), (1<<20)+2)
	if _, err := fmt.Fprintln(status, "Chat ready. One message per line; /reset clears history, /exit quits. Ctrl-C cancels and exits."); err != nil {
		return err
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if _, err := fmt.Fprint(status, "you> "); err != nil {
			return err
		}
		if !scan.Scan() {
			if err := ctx.Err(); err != nil {
				return err
			}
			return scan.Err()
		}
		line := scan.Text()
		if len(line) > 1<<20 {
			return errors.New("chat input exceeds 1 MiB")
		}
		switch strings.TrimSpace(line) {
		case "":
			continue
		case "/exit", "/quit":
			return nil
		case "/reset":
			history = append([]ChatMessage(nil), initial...)
			if _, err := fmt.Fprintln(status, "History cleared."); err != nil {
				return err
			}
			continue
		}
		trial := append(append([]ChatMessage(nil), history...), ChatMessage{Role: "user", Content: line})
		prompt, err := tok.formatChat(trial)
		if err != nil {
			if _, err := fmt.Fprintf(status, "Cannot add turn: %v. Use /reset to start again.\n", err); err != nil {
				return err
			}
			continue
		}
		stream, err := e.Stream(ctx, prompt, s)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if _, err := fmt.Fprintf(status, "Cannot add turn: %v. Use /reset or reduce -max-tokens.\n", err); err != nil {
				return err
			}
			continue
		}
		if _, err := fmt.Fprint(status, "assistant> "); err != nil {
			stream.Close()
			<-stream.Released()
			return err
		}
		var writeErr error
		for event := range stream.Events {
			if _, writeErr = io.WriteString(out, event.Text); writeErr != nil {
				break
			}
		}
		if writeErr != nil {
			stream.Close()
			<-stream.Released()
			return writeErr
		}
		result, err := stream.Wait()
		stream.Close()
		<-stream.Released()
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintln(out); err != nil {
			return err
		}
		history = append(trial, ChatMessage{Role: "assistant", Content: result.Text})
	}
}

func runChat(ctx context.Context, args []string, validateOnly bool) error {
	f := flags("chat")
	dir, limit, budget, format := hfFlags(f)
	system := f.String("system", "", "system message; empty uses the model template default")
	memory := f.Int64("kv-mib", 256, "KV cache budget in MiB")
	workers := f.Int("kernel-workers", 0, "fixed kernel workers; 0 selects automatically")
	threads := threadFlag(f)
	s := defaultSampling()
	samplingFlags(f, &s)
	if err := parseInferenceAction(f, args, validateOnly); err != nil {
		return err
	}
	if err := s.Validate(Config{Context: 32768}); err != nil {
		return err
	}
	if _, err := normalizeEngineConfig(EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: *memory << 20, KernelWorkers: *workers}); err != nil {
		return err
	}
	if validateOnly {
		return nil
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	if *dir == "" || *memory < 1 || *memory > 1<<20 {
		return errors.New("chat requires -hf-model and valid -kv-mib")
	}
	report, err := checkHFFormat(*dir, *limit, *budget, true, *format)
	if err != nil {
		return err
	}
	if err := s.Validate(report.Weights.Config); err != nil {
		return err
	}
	if !utf8.ValidString(*system) || len(*system) > 1<<20 {
		return errors.New("system message must be valid UTF-8 and at most 1 MiB")
	}
	m, tok, err := loadInference(ctx, f, "", *dir, *limit, *budget, *format)
	if err != nil {
		return err
	}
	e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: *memory << 20, KernelWorkers: *workers})
	if err != nil {
		return err
	}
	defer e.Close()
	in, closeInput := cliInput(ctx, os.Stdin)
	defer closeInput()
	return chatLoop(ctx, e, tok.(*hfTokenizer), *system, s, in, os.Stdout, os.Stderr)
}

type evaluationOptions struct {
	Model, Data, Dataset, Split string
	Seq, MaxBatches             int
}
type EvaluationSource struct {
	Kind   string `json:"kind"`
	SHA256 string `json:"sha256"`
	Split  string `json:"split,omitempty"`
}
type TargetRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}
type EvaluationReport struct {
	Format           string           `json:"format"`
	CheckpointSHA256 string           `json:"checkpoint_sha256"`
	TokenizerSHA256  string           `json:"tokenizer_sha256"`
	Source           EvaluationSource `json:"source"`
	Scoring          string           `json:"scoring"`
	SequenceLength   int              `json:"sequence_length"`
	MaxBatches       int              `json:"max_batches"`
	TargetRange      TargetRange      `json:"target_range"`
	Targets          int              `json:"targets"`
	Loss             float64          `json:"loss"`
	Perplexity       any              `json:"perplexity"`
	GoVersion        string           `json:"go_version"`
}

func evaluationReport(ctx context.Context, o evaluationOptions) (EvaluationReport, error) {
	r := EvaluationReport{}
	if err := ctx.Err(); err != nil {
		return r, err
	}
	if (o.Data == "") == (o.Dataset == "") {
		return r, errors.New("provide exactly one of -data or -dataset")
	}
	if o.MaxBatches < 0 || o.Seq < 0 {
		return r, errors.New("evaluation sequence and batch limits must be nonnegative")
	}
	if o.Dataset == "" && o.Split != "" {
		return r, errors.New("-split requires -dataset")
	}
	if o.Dataset != "" && o.Split != "" && o.Split != "train" && o.Split != "validation" {
		return r, errors.New("-split must be train or validation")
	}
	m, tok, _, digest, err := loadCheckpointIdentity(o.Model)
	if err != nil {
		return r, err
	}
	m.DropOptimizer()
	if o.Seq == 0 {
		o.Seq = min(128, m.Config.Context)
	}
	r = EvaluationReport{Format: "monolith-eval-v1", CheckpointSHA256: digest, TokenizerSHA256: tokenizerHash(tok), SequenceLength: o.Seq, MaxBatches: o.MaxBatches, GoVersion: runtime.Version()}
	if o.Dataset != "" {
		d, err := OpenTokenDataset(ctx, o.Dataset)
		if err != nil {
			return r, err
		}
		if d.manifest.TokenizerSHA256 != r.TokenizerSHA256 {
			return r, errors.New("evaluation dataset/tokenizer identity mismatch")
		}
		if o.Split == "" {
			o.Split = "validation"
		}
		r.Source = EvaluationSource{Kind: "dataset", SHA256: d.id, Split: o.Split}
		r.Scoring = "packed-shards-bos-window-v1"
		r.Loss, r.Targets, err = evaluateDatasetSplit(ctx, m, d, o.Split, o.Seq, o.MaxBatches)
		if err != nil {
			return r, err
		}
	} else {
		text, err := readText(o.Data, 64<<20)
		if err != nil {
			return r, err
		}
		r.Source = EvaluationSource{Kind: "text", SHA256: corpusHash(text)}
		r.Scoring = "text-bos-window-v1"
		r.Loss, r.Targets, err = evaluate(ctx, m, tok.Encode(text, true, true), o.Seq, o.MaxBatches)
		if err != nil {
			return r, err
		}
	}
	if err := ctx.Err(); err != nil {
		return r, err
	}
	r.TargetRange = TargetRange{Start: 0, End: r.Targets}
	ppl := math.Exp(r.Loss)
	if finite(ppl) {
		r.Perplexity = ppl
	} else {
		r.Perplexity = "overflow"
	}
	return r, nil
}
func runEval(ctx context.Context, args []string) error {
	f := flags("eval")
	o := evaluationOptions{}
	f.StringVar(&o.Model, "model", "runs/model.mglm", "checkpoint")
	f.StringVar(&o.Data, "data", "", "evaluation text (max 64 MiB)")
	f.StringVar(&o.Dataset, "dataset", "", "prepared manifest.json; mutually exclusive with -data")
	f.StringVar(&o.Split, "split", "", "dataset split: train or validation (default validation)")
	f.IntVar(&o.Seq, "seq", 0, "context window (default min(context,128))")
	f.IntVar(&o.MaxBatches, "batches", 0, "maximum leading windows; 0 evaluates all targets")
	threads := threadFlag(f)
	if err := parse(f, args); err != nil {
		return err
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	report, err := evaluationReport(ctx, o)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(report)
}
func runTokenizer(args []string) error {
	f := flags("tokenizer")
	data := f.String("data", "", "tokenizer training text, at most 4 MiB")
	vocab := f.Int("vocab", 512, "maximum vocabulary size, 258..8192")
	out := f.String("out", "runs/tokenizer.json", "tokenizer JSON destination")
	if err := parse(f, args); err != nil {
		return err
	}
	text, err := readText(*data, 4<<20)
	if err != nil {
		return err
	}
	tok, err := TrainTokenizer(text, *vocab)
	if err != nil {
		return err
	}
	if err = writeJSONFile(*out, tok); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "vocab=%d bytes=%d tokens=%d tokenizer=%s\n", tok.Vocab(), len(text), len(tok.Encode(text, false, false)), *out)
	return nil
}
func runInspect(args []string) error {
	f := flags("inspect")
	model := f.String("model", "runs/model.mglm", "checkpoint")
	if err := parse(f, args); err != nil {
		return err
	}
	m, tok, state, err := LoadCheckpoint(*model)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(map[string]any{"architecture": architecture, "config": m.Config, "parameters": m.Config.ParameterCount(), "weight_bytes": m.Config.ParameterCount() * 4, "kv_bytes_per_token": 8 * m.Config.Layers * m.Config.KVDim(), "tokenizer_merges": len(tok.Merges), "training": state})
}

// Stop accepting HTTP work immediately and share one deadline between HTTP
// draining and engine completion. An in-flight kernel may outlive cancellation;
// the CLI must still be able to return when its shutdown deadline expires.
func shutdownInferenceServer(ctx context.Context, server *http.Server, e *Engine) error {
	e.cancel()
	if err := server.Shutdown(ctx); err != nil {
		_ = server.Close()
		return err
	}
	select {
	case <-e.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

type inferenceTrace struct {
	file   *os.File
	writer *profileWriter
}

func startInferenceTrace(path string) (*inferenceTrace, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return nil, err
	}
	t := &inferenceTrace{file: f, writer: &profileWriter{Writer: f}}
	if err := trace.Start(t.writer); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return nil, err
	}
	return t, nil
}
func (t *inferenceTrace) close() error {
	if t == nil {
		return nil
	}
	trace.Stop() // Joins the trace writer before checking its error or closing the file.
	return errors.Join(t.writer.err, t.file.Close())
}

func runServe(ctx context.Context, args []string, validateOnly bool) (err error) {
	f := flags("serve")
	model := f.String("model", "runs/model.mglm", "checkpoint")
	dir, limit, weightBudget, format := hfFlags(f)
	addr := f.String("addr", "127.0.0.1:8080", "HTTP listen address")
	batch := f.Int("max-batch", 8, "maximum simultaneous sessions per scheduling tick")
	chunk := f.Int("prefill-chunk", 16, "maximum prompt tokens per session per tick; 1 is the sequential baseline")
	mixedChunk := f.Int("mixed-prefill-chunk", 1, "maximum prompt tokens per session when another session is generating")
	budget := f.Int("token-budget", 0, "total tokens per tick; 0 selects max(max-batch,64)")
	buffer := f.Int("stream-buffer", 32, "buffered token events per stream before slow-consumer termination")
	queue := f.Int("queue", 32, "bounded request queue capacity")
	memory := f.Int64("kv-mib", 256, "hard KV page budget in MiB")
	tracePath := f.String("trace", "", "new Go execution trace file for this serve invocation (process-wide)")
	workers := f.Int("kernel-workers", 0, "reusable compute workers per engine (1..256); 0 captures min(threads,256), 1 runs inline")
	threads := threadFlag(f)
	if err := parseInferenceAction(f, args, validateOnly); err != nil {
		return err
	}
	engineConfig := EngineConfig{MaxBatch: *batch, Queue: *queue, CacheBytes: *memory << 20, PrefillChunk: *chunk, MixedPrefillChunk: *mixedChunk, TokenBudget: *budget, StreamBuffer: *buffer, KernelWorkers: *workers}
	if _, err := normalizeEngineConfig(engineConfig); err != nil {
		return err
	}
	if validateOnly {
		return nil
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	if *memory < 1 || *memory > 1<<20 {
		return errors.New("invalid -kv-mib")
	}
	m, tok, err := loadInference(ctx, f, *model, *dir, *limit, *weightBudget, *format)
	if err != nil {
		return err
	}
	m.DropOptimizer()
	e, err := NewEngine(m, tok, engineConfig)
	if err != nil {
		return err
	}
	defer e.cancel() // A deferred blocking Close would bypass the shutdown deadline.
	recorder, err := startInferenceTrace(*tracePath)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, recorder.close()) }()
	server := &http.Server{Addr: *addr, Handler: e.Handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 125 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	defer server.Close()
	result := make(chan error, 1)
	go func() { result <- server.ListenAndServe() }()
	fmt.Fprintf(os.Stderr, "serving %s parameters=%d max_batch=%d queue=%d kv_mib=%d threads=%d kernel_workers=%d prefill_chunk=%d mixed_prefill_chunk=%d token_budget=%d stream_buffer=%d\n", *addr, m.parameterCount(), *batch, *queue, *memory, runtime.GOMAXPROCS(0), e.Config.KernelWorkers, e.Config.PrefillChunk, e.Config.MixedPrefillChunk, e.Config.TokenBudget, e.Config.StreamBuffer)
	select {
	case err = <-result:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return shutdownInferenceServer(shutdown, server, e)
}

// HF source resolution is separate from inference. Only explicitly selected
// local commands download weights; remote commands never enter this path.
type hfHubFile struct {
	Name string `json:"rfilename"`
	Size int64  `json:"size"`
	Blob string `json:"blobId"`
	LFS  *struct {
		SHA  string `json:"sha256"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}
type hfHubInfo struct {
	SHA   string      `json:"sha"`
	Files []hfHubFile `json:"siblings"`
}
type hfCachedFile struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	SHA  string `json:"sha256"`
}
type hfSnapshot struct {
	Format   string         `json:"format"`
	Repo     string         `json:"repo"`
	Revision string         `json:"revision"`
	Files    []hfCachedFile `json:"files"`
}
type hfSource struct {
	Repo, Revision, Cache    string
	WeightFormat             string
	Offline                  bool
	DownloadBytes, WeightMiB int64
	Context                  int
}
type hfHub struct {
	Base, Token string
	Client      *http.Client
}

var errHFOffline = errors.New("HF_HUB_OFFLINE disables HF network requests; use a verified local cache or unset HF_HUB_OFFLINE")

// Match huggingface_hub's boolean convention. Read at the operation boundary
// rather than at initialization so embedded callers can configure it as well.
func hfOffline() bool {
	switch strings.ToUpper(os.Getenv("HF_HUB_OFFLINE")) {
	case "1", "ON", "YES", "TRUE":
		return true
	default:
		return false
	}
}

func newHFClient(timeout time.Duration) *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.MaxConnsPerHost, t.MaxIdleConnsPerHost = 4, 4
	t.ResponseHeaderTimeout = 30 * time.Second
	return &http.Client{Transport: t, Timeout: timeout, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 {
			return errors.New("HF redirect limit")
		}
		if req.URL.Scheme != "https" {
			return errors.New("HF redirect must use HTTPS")
		}
		if len(via) > 0 && req.URL.Host != via[0].URL.Host {
			req.Header.Del("Authorization")
		}
		return nil
	}}
}
func hfHTTPError(status int) error {
	switch status {
	case 401:
		return errors.New("HF authentication failed; set HF_TOKEN with the required permissions")
	case 403:
		return errors.New("HF access denied; check model access, gating and token permissions")
	case 404:
		return errors.New("HF model, revision or endpoint not found")
	case 429:
		return errors.New("HF rate limit reached; no automatic retry was made")
	default:
		return fmt.Errorf("HF request returned HTTP %d; no automatic retry was made", status)
	}
}
func hfDo(ctx context.Context, client *http.Client, req *http.Request) (*http.Response, error) {
	if hfOffline() {
		return nil, errHFOffline
	}
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		// Transport errors can contain signed download URLs. Keep them out of logs.
		return nil, errors.New("HF request failed; check connectivity and timeout")
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, hfHTTPError(resp.StatusCode)
	}
	return resp, nil
}
func hfReadJSON(r io.Reader, limit int64, dst any) error {
	b, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return err
	}
	if int64(len(b)) > limit {
		return errors.New("HF JSON response exceeds size limit")
	}
	return uniqueJSON(b, dst)
}
func (h hfHub) get(ctx context.Context, path string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(h.Base, "/")+path, nil)
	if err != nil {
		return nil, errors.New("invalid HF request URL")
	}
	if h.Token != "" {
		req.Header.Set("Authorization", "Bearer "+h.Token)
	}
	req.Header.Set("User-Agent", "sr-go-tranformer/1")
	return hfDo(ctx, h.Client, req)
}
func validHFRepo(repo string) bool {
	parts := strings.Split(repo, "/")
	if len(parts) != 2 {
		return false
	}
	for _, p := range parts {
		if p == "" || len(p) > 96 || strings.Contains(p, "..") || strings.HasPrefix(p, ".") || strings.HasSuffix(p, ".") {
			return false
		}
		for _, c := range p {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
				return false
			}
		}
	}
	return true
}
func validHFCommit(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 20 && s == strings.ToLower(s)
}
func hfKey(s string) string { b := sha256.Sum256([]byte(s)); return hex.EncodeToString(b[:]) }
func hfFileHash(ctx context.Context, path string, size int64) (string, error) {
	st, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	if !st.Mode().IsRegular() || st.Size() != size {
		return "", errors.New("HF cache file size/type mismatch")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hash := sha256.New()
	buf := make([]byte, 64<<10)
	r := io.LimitReader(f, size+1)
	var read int64
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, err := r.Read(buf)
		read += int64(n)
		if n > 0 {
			hash.Write(buf[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}
	if read != size {
		return "", errors.New("HF cache changed size during checksum verification")
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
func validateHFSnapshot(ctx context.Context, dir string, s hfSource, sha string) (hfSnapshot, error) {
	var m hfSnapshot
	st, err := os.Lstat(dir)
	if err != nil {
		return m, err
	}
	if !st.IsDir() {
		return m, errors.New("HF snapshot must be a directory, not a symlink")
	}
	if err := readStrictBoundedJSON(filepath.Join(dir, "snapshot.json"), 1<<20, &m); err != nil {
		return m, err
	}
	if m.Format != "monolith-hf-snapshot-v1" || m.Repo != s.Repo || m.Revision != sha || len(m.Files) < 4 || len(m.Files) > 134 {
		return m, errors.New("HF snapshot identity mismatch")
	}
	seen := map[string]bool{}
	var size int64
	for _, f := range m.Files {
		if f.Name == "" || filepath.Base(f.Name) != f.Name || strings.ContainsAny(f.Name, `/\`) || f.Name == "snapshot.json" || seen[f.Name] || f.Size < 1 || f.Size > s.DownloadBytes-size || !validSHA256(f.SHA) {
			return m, errors.New("invalid HF snapshot file or download budget")
		}
		seen[f.Name] = true
		size += f.Size
		sum, err := hfFileHash(ctx, filepath.Join(dir, f.Name), f.Size)
		if err != nil {
			return m, err
		}
		if sum != f.SHA {
			return m, errors.New("HF cache checksum mismatch; use a fresh cache directory")
		}
	}
	if !seen["config.json"] || !seen["tokenizer.json"] || !seen["tokenizer_config.json"] || !seen["model.safetensors"] && !seen["model.safetensors.index.json"] {
		return m, errors.New("incomplete HF snapshot")
	}
	return m, nil
}
func writeHFJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".json-")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(append(b, '\n')); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(f.Name(), path)
}
func (h hfHub) download(ctx context.Context, dir, repo, sha string, f hfHubFile) (hfCachedFile, error) {
	out := hfCachedFile{Name: f.Name, Size: f.Size}
	resp, err := h.get(ctx, "/"+repo+"/resolve/"+sha+"/"+url.PathEscape(f.Name))
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.ContentLength >= 0 && resp.ContentLength != f.Size {
		return out, errors.New("HF download size differs from pinned metadata")
	}
	dst, err := os.OpenFile(filepath.Join(dir, f.Name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return out, err
	}
	defer dst.Close()
	sum := sha256.New()
	gitBlob := sha1.New() // Git blob identity for non-LFS files, not a security primitive.
	fmt.Fprintf(gitBlob, "blob %d%c", f.Size, 0)
	n, err := io.CopyBuffer(io.MultiWriter(dst, sum, gitBlob), io.LimitReader(resp.Body, f.Size+1), make([]byte, 64<<10))
	if err != nil {
		if ctx.Err() != nil {
			return out, ctx.Err()
		}
		return out, errors.New("HF file download interrupted")
	}
	if n != f.Size {
		return out, errors.New("HF download size mismatch")
	}
	out.SHA = hex.EncodeToString(sum.Sum(nil))
	if f.LFS != nil {
		if f.LFS.Size != f.Size || out.SHA != f.LFS.SHA {
			return out, errors.New("HF LFS SHA-256 mismatch")
		}
	} else if hex.EncodeToString(gitBlob.Sum(nil)) != f.Blob {
		return out, errors.New("HF Git blob identity mismatch")
	}
	if err = dst.Sync(); err != nil {
		return out, err
	}
	return out, dst.Close()
}

func (h hfHub) resolve(ctx context.Context, s hfSource) (string, hfSnapshot, error) {
	var empty hfSnapshot
	if s.WeightFormat == "" {
		s.WeightFormat = "f32"
	}
	if !validHFWeightFormat(s.WeightFormat) {
		return "", empty, errors.New("weight-format must be f32, q8, q4 or q4-mixed")
	}
	s.Offline = s.Offline || hfOffline()
	if !validHFRepo(s.Repo) || s.Revision == "" || len(s.Revision) > 200 || strings.ContainsAny(s.Revision, "\x00\r\n") || s.DownloadBytes < 1 || s.DownloadBytes > 16<<30 || s.WeightMiB < 1 || s.WeightMiB > 1<<20 {
		return "", empty, errors.New("invalid HF source or byte budget")
	}
	root := filepath.Join(s.Cache, hfKey(s.Repo))
	ref := filepath.Join(root, "refs", hfKey(s.Revision)+".json")
	sha := s.Revision
	var info hfHubInfo
	if s.Offline {
		if !validHFCommit(sha) {
			var pin struct{ Repo, Ref, SHA string }
			if err := readStrictBoundedJSON(ref, 4096, &pin); err != nil {
				return "", empty, errors.New("offline revision is not cached; provide a cached commit SHA")
			}
			if pin.Repo != s.Repo || pin.Ref != s.Revision || !validHFCommit(pin.SHA) {
				return "", empty, errors.New("invalid cached HF revision")
			}
			sha = pin.SHA
		}
	} else {
		resp, err := h.get(ctx, "/api/models/"+s.Repo+"/revision/"+url.PathEscape(s.Revision)+"?blobs=true")
		if err != nil {
			return "", empty, err
		}
		err = hfReadJSON(resp.Body, 2<<20, &info)
		resp.Body.Close()
		if err != nil {
			return "", empty, err
		}
		if !validHFCommit(info.SHA) || validHFCommit(sha) && info.SHA != sha {
			return "", empty, errors.New("HF did not resolve the requested commit")
		}
		sha = info.SHA
	}
	dir := filepath.Join(root, sha)
	if _, err := os.Lstat(dir); err == nil {
		m, err := validateHFSnapshot(ctx, dir, s, sha)
		if err == nil && !s.Offline {
			if err = os.MkdirAll(filepath.Dir(ref), 0700); err == nil {
				err = writeHFJSON(ref, struct{ Repo, Ref, SHA string }{s.Repo, s.Revision, sha})
			}
		}
		return dir, m, err
	} else if !os.IsNotExist(err) {
		return "", empty, err
	}
	if s.Offline {
		return "", empty, errors.New("HF snapshot is not cached; offline mode made no network requests")
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return "", empty, err
	}
	stage, err := os.MkdirTemp(root, ".download-")
	if err != nil {
		return "", empty, err
	}
	defer os.RemoveAll(stage)
	files := map[string]hfHubFile{}
	for _, f := range info.Files {
		if _, exists := files[f.Name]; exists {
			return "", empty, errors.New("duplicate HF file metadata")
		}
		files[f.Name] = f
	}
	m := hfSnapshot{Format: "monolith-hf-snapshot-v1", Repo: s.Repo, Revision: sha}
	var total int64
	take := func(name string, maxSize int64) error {
		f, ok := files[name]
		if !ok {
			return fmt.Errorf("HF repository lacks %s", name)
		}
		if name == "" || filepath.Base(name) != name || strings.ContainsAny(name, `/\`) || f.Size < 1 || f.Size > maxSize || f.Size > s.DownloadBytes-total || (f.LFS != nil && (!validSHA256(f.LFS.SHA) || f.LFS.Size != f.Size)) || (f.LFS == nil && !validHFCommit(f.Blob)) {
			return errors.New("HF file metadata or download budget rejected")
		}
		total += f.Size
		got, err := h.download(ctx, stage, s.Repo, sha, f)
		if err == nil {
			m.Files = append(m.Files, got)
		}
		return err
	}
	// Reject unsupported computation and tokenizer profiles before weight traffic.
	if err = take("config.json", 1<<20); err != nil {
		return "", empty, err
	}
	hcfg, c, err := readHFConfig(stage, s.Context)
	if err != nil {
		return "", empty, fmt.Errorf("local Go profile: %w (choose infer remote explicitly for other architectures)", err)
	}
	storage, _, _, err := hfWeightStorage(c, hcfg, s.WeightFormat)
	if err != nil {
		return "", empty, err
	}
	if storage > s.WeightMiB<<20 {
		return "", empty, errors.New("HF resident weight budget exceeded before weight download")
	}
	if err = take("tokenizer.json", 32<<20); err != nil {
		return "", empty, err
	}
	if err = take("tokenizer_config.json", 1<<20); err != nil {
		return "", empty, err
	}
	if _, err = loadHFTokenizer(stage, hcfg); err != nil {
		return "", empty, err
	}
	var weights []string
	if _, ok := files["model.safetensors.index.json"]; ok {
		if err = take("model.safetensors.index.json", 4<<20); err != nil {
			return "", empty, err
		}
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err = readBoundedJSON(filepath.Join(stage, "model.safetensors.index.json"), 4<<20, &index); err != nil {
			return "", empty, err
		}
		shards := map[string]bool{}
		for _, name := range index.WeightMap {
			shards[name] = true
		}
		if len(shards) < 1 || len(shards) > 128 {
			return "", empty, errors.New("HF shard count limit")
		}
		for name := range shards {
			if !strings.HasSuffix(name, ".safetensors") || filepath.Base(name) != name || strings.ContainsAny(name, `/\`) {
				return "", empty, errors.New("invalid HF shard name")
			}
			weights = append(weights, name)
		}
		sort.Strings(weights)
	} else {
		weights = []string{"model.safetensors"}
	}
	remaining := s.DownloadBytes - total
	for _, name := range weights {
		f, ok := files[name]
		if !ok || f.Size < 1 || f.Size > remaining {
			return "", empty, errors.New("HF weights exceed download budget or are missing")
		}
		remaining -= f.Size
	}
	for _, name := range weights {
		if err = take(name, s.DownloadBytes); err != nil {
			return "", empty, err
		}
	}
	a, err := openHFArchiveFormat(stage, s.Context, s.WeightMiB<<20, s.WeightFormat)
	if err != nil {
		return "", empty, err
	}
	a.Close()
	if err = writeHFJSON(filepath.Join(stage, "snapshot.json"), m); err != nil {
		return "", empty, err
	}
	if err = os.Rename(stage, dir); err != nil {
		// A concurrent invocation may have atomically published the same snapshot.
		existing, checkErr := validateHFSnapshot(ctx, dir, s, sha)
		if checkErr != nil {
			return "", empty, err
		}
		m = existing
	}
	if err = os.MkdirAll(filepath.Dir(ref), 0700); err != nil {
		return "", empty, err
	}
	if err = writeHFJSON(ref, struct{ Repo, Ref, SHA string }{s.Repo, s.Revision, sha}); err != nil {
		return "", empty, err
	}
	return dir, m, nil
}

func runLearn(ctx context.Context, args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		fmt.Fprintln(os.Stdout, "learn: native Go model training and evaluation\nusage: monolith learn <demo|train|eval|tokenizer|prepare|inspect> [flags]\nHF imports and remote inference are available under infer.")
		return nil
	}
	switch args[0] {
	case "demo", "train", "eval", "tokenizer", "prepare", "inspect":
		return run(ctx, args)
	default:
		return errors.New("unknown learn command; use learn -h")
	}
}
func runInferLocal(ctx context.Context, args []string) error {
	f := flags("infer local")
	repo := f.String("repo", "", "HF namespace/model to fetch automatically")
	revision := f.String("revision", "main", "HF revision; resolved to an immutable commit")
	cache := f.String("cache-dir", "", "cache directory; default is the OS user cache")
	offline := f.Bool("offline", false, "use a verified cached snapshot without network; also enabled by HF_HUB_OFFLINE")
	download := f.Int64("download-mib", 2048, "maximum source files in MiB")
	dir := f.String("dir", "", "existing local HF model directory")
	checkpoint := f.String("checkpoint", "", "native .mglm checkpoint")
	limit := f.Int("context", 0, "HF context cap; 0 selects at most 2048")
	weight := f.Int64("weight-mib", 2048, "maximum resident HF weight payload in MiB, not RSS")
	format := f.String("weight-format", "f32", "resident HF weights: f32, q8, q4 or q4-mixed (Q8 embeddings/head, Q4 projections)")
	f.Usage = func() {
		fmt.Fprintln(f.Output(), "usage: monolith infer local [source flags] <check|inspect|tokens|ids|generate|chat|serve> [action flags]")
		f.PrintDefaults()
	}
	if err := f.Parse(args); err != nil {
		return err
	}
	rest := f.Args()
	if len(rest) == 0 {
		f.Usage()
		return nil
	}
	action := map[string]string{"check": "check-hf", "inspect": "inspect-hf", "tokens": "tokenize-hf", "ids": "infer-hf", "generate": "generate", "chat": "chat", "serve": "serve"}[rest[0]]
	if action == "" {
		return errors.New("unknown infer local action")
	}
	if len(rest) == 2 && (rest[1] == "-h" || rest[1] == "--help") {
		return run(ctx, []string{action, "-h"})
	}
	sources := 0
	for _, s := range []string{*repo, *dir, *checkpoint} {
		if s != "" {
			sources++
		}
	}
	if sources != 1 {
		return errors.New("infer local requires exactly one of -repo, -dir or -checkpoint")
	}
	if err := validateInferenceFlags(f); err != nil {
		return err
	}
	if err := runInferenceAction(ctx, action, rest[1:], true); err != nil {
		return err
	}
	if *checkpoint != "" {
		if action != "generate" && action != "serve" && rest[0] != "inspect" {
			return errors.New("native checkpoints support generate, serve and inspect")
		}
		if flagSet(f, "context") || flagSet(f, "weight-mib") || flagSet(f, "weight-format") || flagSet(f, "revision") || flagSet(f, "cache-dir") || flagSet(f, "download-mib") || *offline {
			return errors.New("HF source flags do not apply to a native checkpoint")
		}
		if rest[0] == "inspect" {
			action = "inspect"
		}
		return run(ctx, append([]string{action, "-model", *checkpoint}, rest[1:]...))
	}
	if *repo != "" {
		if *download < 1 || *download > 16384 {
			return errors.New("download-mib must be 1..16384")
		}
		if *cache == "" {
			base, err := os.UserCacheDir()
			if err != nil {
				return err
			}
			*cache = filepath.Join(base, "sr_go_tranformer", "hf")
		}
		client := newHFClient(15 * time.Minute)
		defer client.CloseIdleConnections()
		h := hfHub{Base: "https://huggingface.co", Token: os.Getenv("HF_TOKEN"), Client: client}
		path, m, err := h.resolve(ctx, hfSource{Repo: *repo, Revision: *revision, Cache: *cache, Offline: *offline, DownloadBytes: *download << 20, WeightMiB: *weight, WeightFormat: *format, Context: *limit})
		if err != nil {
			return err
		}
		*dir = path
		status, closeStatus := cliOutput(ctx, os.Stderr)
		_, err = fmt.Fprintf(status, "local Go: %s @ %s\n", m.Repo, m.Revision)
		closeStatus()
		if err != nil {
			return err
		}
	} else if *offline || flagSet(f, "revision") || flagSet(f, "cache-dir") || flagSet(f, "download-mib") {
		return errors.New("cache/download flags require -repo")
	}
	injected := []string{action, "-hf-model", *dir, "-context", strconv.Itoa(*limit)}
	if action != "tokenize-hf" {
		injected = append(injected, "-weight-mib", strconv.FormatInt(*weight, 10), "-weight-format", *format)
	}
	return run(ctx, append(injected, rest[1:]...))
}

type hfRemote struct {
	Base, Token, Model, API string
	Client                  *http.Client
	MaxTokens               int
	Temperature, TopP       float64
}
type hfRemoteResult struct {
	Backend        string          `json:"backend"`
	RequestedModel string          `json:"requested_model"`
	Model          string          `json:"model,omitempty"`
	Text           string          `json:"text"`
	FinishReason   string          `json:"finish_reason"`
	Usage          json.RawMessage `json:"usage,omitempty"`
}
type hfRemoteEnvelope struct {
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Content   string            `json:"content"`
			Refusal   string            `json:"refusal"`
			ToolCalls []json.RawMessage `json:"tool_calls"`
		} `json:"message"`
		Delta struct {
			Content   string            `json:"content"`
			ToolCalls []json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
		Finish *string `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
	Error json.RawMessage `json:"error"`
}

func validateHFEndpoint(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.User != nil || u.Host == "" || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("invalid HF endpoint URL")
	}
	host := u.Hostname()
	if u.Scheme != "https" || !(host == "router.huggingface.co" || strings.HasSuffix(host, ".endpoints.huggingface.cloud")) {
		return errors.New("endpoint must use HTTPS on router.huggingface.co or a dedicated *.endpoints.huggingface.cloud host")
	}
	return nil
}
func validateRemoteMessages(messages []ChatMessage) error {
	if len(messages) < 1 || len(messages) > 128 {
		return errors.New("remote input requires 1..128 text messages")
	}
	total := 0
	for _, m := range messages {
		if m.Role != "system" && m.Role != "user" && m.Role != "assistant" {
			return errors.New("remote message role must be system, user or assistant")
		}
		total += len(m.Content) + len(m.Role) + 64
		if total > 1<<20 || !utf8.ValidString(m.Content) {
			return errors.New("remote input exceeds UTF-8/1 MiB limit")
		}
	}
	return nil
}
func remoteInput(ctx context.Context, f *flag.FlagSet, prompt, source, messages, system string) ([]ChatMessage, error) {
	if messages != "" {
		if source != "" || flagSet(f, "prompt") || flagSet(f, "system") {
			return nil, errors.New("-messages cannot be combined with prompt or system flags")
		}
		b, err := readCLIInput(ctx, messages, os.Stdin)
		if err != nil {
			return nil, err
		}
		var out []ChatMessage
		if err = uniqueJSON(b, &out); err != nil {
			return nil, err
		}
		d := json.NewDecoder(bytes.NewReader(b))
		d.DisallowUnknownFields()
		if err = d.Decode(&out); err != nil {
			return nil, err
		}
		return out, validateRemoteMessages(out)
	}
	if source != "" {
		if flagSet(f, "prompt") {
			return nil, errors.New("-prompt and -prompt-file are mutually exclusive")
		}
		b, err := readCLIInput(ctx, source, os.Stdin)
		if err != nil {
			return nil, err
		}
		prompt = string(b)
	}
	if prompt == "" {
		return nil, errors.New("provide a nonempty prompt, prompt file or messages")
	}
	var out []ChatMessage
	if system != "" {
		out = append(out, ChatMessage{Role: "system", Content: system})
	}
	out = append(out, ChatMessage{Role: "user", Content: prompt})
	return out, validateRemoteMessages(out)
}
func (r hfRemote) requestContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if r.Client != nil && r.Client.Timeout > 0 {
		return context.WithTimeout(ctx, r.Client.Timeout)
	}
	return context.WithCancel(ctx)
}
func (r hfRemote) complete(ctx context.Context, messages []ChatMessage, out io.Writer) (hfRemoteResult, error) {
	// http.Client.Timeout alone only interrupts network I/O. Carry the same
	// deadline through delivery so a stalled output cannot outlive a request.
	ctx, cancel := r.requestContext(ctx)
	defer cancel()
	if out != nil {
		var closeOutput func()
		out, closeOutput = cliOutput(ctx, out)
		defer closeOutput()
	}
	result := hfRemoteResult{Backend: "hf-remote", RequestedModel: r.Model}
	if err := validateRemoteMessages(messages); err != nil {
		return result, err
	}
	if r.Token == "" {
		return result, errors.New("set HF_TOKEN for remote inference; no request was sent")
	}
	if r.MaxTokens < 1 || r.MaxTokens > 4096 || !finite(r.Temperature) || r.Temperature < 0 || r.Temperature > 2 || !finite(r.TopP) || r.TopP <= 0 || r.TopP > 1 {
		return result, errors.New("invalid remote sampling options")
	}
	if r.Model == "" || len(r.Model) > 256 || strings.ContainsAny(r.Model, "\x00\r\n") {
		return result, errors.New("provide a valid remote -model ID (optionally :provider)")
	}
	var payload any
	endpoint := strings.TrimRight(r.Base, "/") + "/chat/completions"
	if r.API == "text" {
		if out != nil || len(messages) != 1 || messages[0].Role != "user" {
			return result, errors.New("text endpoints require one raw prompt and do not support chat/stream here")
		}
		parameters := map[string]any{"max_new_tokens": r.MaxTokens, "return_full_text": false, "do_sample": r.Temperature > 0}
		if r.Temperature > 0 {
			parameters["temperature"] = r.Temperature
			parameters["top_p"] = r.TopP
		}
		payload = map[string]any{"inputs": messages[0].Content, "parameters": parameters}
		endpoint = r.Base
	} else if r.API == "chat" {
		payload = map[string]any{"model": r.Model, "messages": messages, "max_tokens": r.MaxTokens, "temperature": r.Temperature, "top_p": r.TopP, "stream": out != nil}
	} else {
		return result, errors.New("remote api must be chat or text")
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return result, err
	}
	if len(b) > 2<<20 {
		return result, errors.New("encoded remote request exceeds 2 MiB")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(b))
	if err != nil {
		return result, errors.New("invalid remote endpoint")
	}
	req.Header.Set("Authorization", "Bearer "+r.Token)
	req.Header.Set("Content-Type", "application/json")
	if out != nil {
		req.Header.Set("Accept", "text/event-stream")
	}
	resp, err := hfDo(ctx, r.Client, req)
	if err != nil {
		return result, err
	}
	defer resp.Body.Close()
	if r.API == "text" {
		var rows []struct {
			Text *string `json:"generated_text"`
		}
		if err = hfReadJSON(resp.Body, 8<<20, &rows); err != nil {
			if ctx.Err() != nil {
				return result, ctx.Err()
			}
			return result, errors.New("invalid or oversized HF text response")
		}
		if len(rows) != 1 || rows[0].Text == nil {
			return result, errors.New("HF text endpoint did not return one generated_text")
		}
		result.Text = *rows[0].Text
		result.FinishReason = "unspecified"
		return result, nil
	}
	var text strings.Builder
	apply := func(e hfRemoteEnvelope, stream bool) error {
		if len(e.Error) > 0 && string(e.Error) != "null" {
			return errors.New("HF provider reported a generation error")
		}
		if e.Model != "" {
			result.Model = e.Model
		}
		if len(e.Usage) > 0 && string(e.Usage) != "null" {
			result.Usage = e.Usage
		}
		if len(e.Choices) == 0 && stream {
			return nil
		}
		if len(e.Choices) != 1 || e.Choices[0].Index != 0 {
			return errors.New("HF response must contain exactly one choice")
		}
		c := e.Choices[0]
		if len(c.Message.ToolCalls) > 0 || len(c.Delta.ToolCalls) > 0 {
			return errors.New("tool-call responses are not supported by the text CLI")
		}
		piece := c.Message.Content
		if piece == "" && !stream {
			piece = c.Message.Refusal
		}
		if stream {
			piece = c.Delta.Content
		}
		if len(result.Text)+len(piece) > 4<<20 {
			return errors.New("HF output exceeds 4 MiB text limit")
		}
		if result.FinishReason != "" && piece != "" {
			return errors.New("HF stream sent text after completion")
		}
		text.WriteString(piece)
		result.Text = text.String()
		if out != nil && piece != "" {
			if _, err := io.WriteString(out, piece); err != nil {
				return err
			}
		}
		if c.Finish != nil {
			result.FinishReason = *c.Finish
		}
		return nil
	}
	if out == nil {
		var e hfRemoteEnvelope
		if err = hfReadJSON(resp.Body, 8<<20, &e); err != nil {
			if ctx.Err() != nil {
				return result, ctx.Err()
			}
			return result, errors.New("invalid or oversized HF chat response")
		}
		if err = apply(e, false); err != nil {
			return result, err
		}
		if result.FinishReason == "" {
			return result, errors.New("HF chat response lacks a finish reason")
		}
		return result, nil
	}
	if !strings.HasPrefix(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		return result, errors.New("HF streaming response is not SSE")
	}
	scan := bufio.NewScanner(io.LimitReader(resp.Body, (8<<20)+1))
	scan.Buffer(make([]byte, 4096), 256<<10)
	var data strings.Builder
	total := 0
	dispatch := func() (bool, error) {
		value := strings.TrimSuffix(data.String(), "\n")
		data.Reset()
		if value == "" {
			return false, nil
		}
		if value == "[DONE]" {
			if result.FinishReason == "" {
				return false, errors.New("HF stream ended without a finish reason")
			}
			return true, nil
		}
		var e hfRemoteEnvelope
		if err := uniqueJSON([]byte(value), &e); err != nil {
			return false, errors.New("invalid HF SSE JSON")
		}
		return false, apply(e, true)
	}
	for scan.Scan() {
		line := scan.Text()
		total += len(line) + 1
		if total > 8<<20 {
			return result, errors.New("HF stream exceeds 8 MiB wire limit")
		}
		if line == "" {
			done, err := dispatch()
			if err != nil {
				return result, err
			}
			if done {
				return result, nil
			}
		} else if strings.HasPrefix(line, "data:") {
			part := strings.TrimPrefix(line, "data:")
			part = strings.TrimPrefix(part, " ")
			if data.Len()+len(part) > 256<<10 {
				return result, errors.New("HF SSE event exceeds limit")
			}
			data.WriteString(part)
			data.WriteByte('\n')
		}
	}
	if ctx.Err() != nil {
		return result, ctx.Err()
	}
	if scan.Err() != nil {
		return result, errors.New("HF stream interrupted or event too large")
	}
	if data.Len() > 0 {
		done, err := dispatch()
		if err != nil {
			return result, err
		}
		if done {
			return result, nil
		}
	}
	return result, errors.New("HF stream truncated before [DONE]; partial output was not committed to chat history")
}

type hfRemoteModel struct {
	ID        string `json:"id"`
	Providers []struct {
		Name   string `json:"provider"`
		Status string `json:"status"`
	} `json:"providers"`
}

func listHFRemote(ctx context.Context, client *http.Client, base, token string) ([]hfRemoteModel, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/models", nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := hfDo(ctx, client, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var envelope struct {
		Data []hfRemoteModel `json:"data"`
	}
	if err = hfReadJSON(resp.Body, 4<<20, &envelope); err != nil {
		return nil, err
	}
	if len(envelope.Data) > 4096 {
		return nil, errors.New("HF remote model list exceeds limit")
	}
	return envelope.Data, nil
}
func runRemoteChat(ctx context.Context, r hfRemote, system string, in io.Reader, out, status io.Writer) error {
	status, closeStatus := cliOutput(ctx, status)
	defer closeStatus()
	var initial []ChatMessage
	if system != "" {
		initial = []ChatMessage{{Role: "system", Content: system}}
	}
	history := append([]ChatMessage(nil), initial...)
	scan := bufio.NewScanner(in)
	scan.Buffer(make([]byte, 4096), (1<<20)+2)
	if _, err := fmt.Fprintln(status, "Remote chat ready. Prompts go to the selected HF endpoint. /reset clears history; /exit quits; Ctrl-C cancels."); err != nil {
		return err
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if _, err := fmt.Fprint(status, "you> "); err != nil {
			return err
		}
		if !scan.Scan() {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return scan.Err()
		}
		line := scan.Text()
		switch strings.TrimSpace(line) {
		case "":
			continue
		case "/exit", "/quit":
			return nil
		case "/reset":
			history = append([]ChatMessage(nil), initial...)
			if _, err := fmt.Fprintln(status, "History cleared."); err != nil {
				return err
			}
			continue
		}
		trial := append(append([]ChatMessage(nil), history...), ChatMessage{Role: "user", Content: line})
		if err := validateRemoteMessages(trial); err != nil {
			if _, err := fmt.Fprintf(status, "Cannot add turn: %v. Use /reset.\n", err); err != nil {
				return err
			}
			continue
		}
		if _, err := fmt.Fprint(status, "assistant> "); err != nil {
			return err
		}
		turnCtx, cancel := r.requestContext(ctx)
		turnOut, closeOutput := cliOutput(turnCtx, out)
		result, err := r.complete(turnCtx, trial, turnOut)
		if err == nil {
			_, err = fmt.Fprintln(turnOut)
		}
		closeOutput()
		cancel()
		if err != nil {
			return err
		}
		history = append(trial, ChatMessage{Role: "assistant", Content: result.Text})
	}
}
func runInferRemote(ctx context.Context, args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		fmt.Fprintln(os.Stdout, "usage: monolith infer remote <models|generate|chat> [flags]\nRemote inference uses HF APIs and never downloads weights. Set HF_TOKEN for generation.\nUse models -family gpt-oss (or mistral, llama, gemma) to inspect current router availability.\nA dedicated HF endpoint can be supplied with -endpoint; -api text accepts raw prompts.")
		return nil
	}
	f := flags("infer remote " + args[0])
	base := f.String("endpoint", "https://router.huggingface.co/v1", "HF API base URL, or full dedicated endpoint URL for -api text")
	asJSON := f.Bool("json", false, "emit structured result")
	timeout := f.Duration("timeout", 2*time.Minute, "request timeout (1s..15m)")
	family := f.String("family", "", "model list filter, such as gpt-oss, mistral, llama or gemma")
	model := f.String("model", "", "HF model ID, optionally with :provider")
	api := f.String("api", "chat", "chat or text (text requires a dedicated HF endpoint)")
	prompt := f.String("prompt", "", "user prompt")
	source := f.String("prompt-file", "", "UTF-8 prompt file, or - for stdin")
	messages := f.String("messages", "", "JSON messages file, or - for stdin")
	system := f.String("system", "", "optional system message")
	stream := f.Bool("stream", false, "stream assistant text over SSE")
	count := f.Int("max-tokens", 64, "maximum new tokens (1..4096)")
	temp := f.Float64("temperature", .7, "remote sampling temperature (0..2)")
	topP := f.Float64("top-p", 1, "remote nucleus sampling threshold")
	if err := parse(f, args[1:]); err != nil {
		return err
	}
	if args[0] != "models" && args[0] != "generate" && args[0] != "chat" {
		return errors.New("unknown remote action")
	}
	// Reject before reading prompts/stdin, starting output pumps or opening a
	// client. The transport boundary also guards non-CLI HF callers.
	if hfOffline() {
		return errHFOffline
	}
	if err := validateHFEndpoint(*base); err != nil {
		return err
	}
	if *timeout < time.Second || *timeout > 15*time.Minute {
		return errors.New("remote timeout must be 1s..15m")
	}
	if args[0] != "chat" {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}
	stdout, closeOutput := cliOutput(ctx, os.Stdout)
	defer closeOutput()
	stderr, closeStatus := cliOutput(ctx, os.Stderr)
	defer closeStatus()
	client := newHFClient(*timeout)
	// Remote API redirects are rejected, even to another HF host. A POST must
	// never be replayed or have prompts/credentials silently redirected.
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return errors.New("remote inference redirects are disabled")
	}
	defer client.CloseIdleConnections()
	token := os.Getenv("HF_TOKEN")
	if args[0] == "models" {
		for _, name := range []string{"model", "prompt", "prompt-file", "messages", "system", "stream", "max-tokens", "temperature", "top-p"} {
			if flagSet(f, name) {
				return fmt.Errorf("-%s does not apply to remote models", name)
			}
		}
		if *api != "chat" {
			return errors.New("models lists the chat router; text endpoints do not expose this catalog")
		}
		rows, err := listHFRemote(ctx, client, *base, token)
		if err != nil {
			return err
		}
		filtered := []hfRemoteModel{}
		for _, row := range rows {
			if strings.Contains(strings.ToLower(row.ID), strings.ToLower(*family)) {
				filtered = append(filtered, row)
			}
		}
		if *asJSON {
			return json.NewEncoder(stdout).Encode(filtered)
		}
		for _, row := range filtered {
			providers := []string{}
			for _, p := range row.Providers {
				if p.Status == "live" {
					providers = append(providers, p.Name)
				}
			}
			if _, err := fmt.Fprintf(stdout, "%s\t%s\n", row.ID, strings.Join(providers, ",")); err != nil {
				return err
			}
		}
		if len(filtered) == 0 {
			_, err = fmt.Fprintln(stderr, "No matching models in this HF router listing. A dedicated endpoint may still serve the model.")
			return err
		}
		return nil
	}
	if flagSet(f, "family") {
		return errors.New("-family is only for remote models")
	}
	if token == "" {
		return errors.New("set HF_TOKEN for remote inference; no request was sent")
	}
	if *api == "text" && *base == "https://router.huggingface.co/v1" {
		return errors.New("GPT-2/raw text generation requires an explicit dedicated -endpoint with -api text")
	}
	r := hfRemote{Base: *base, Token: token, Model: *model, API: *api, Client: client, MaxTokens: *count, Temperature: *temp, TopP: *topP}
	if args[0] == "chat" {
		if *api != "chat" || *asJSON || flagSet(f, "prompt") || *source != "" || *messages != "" {
			return errors.New("remote chat requires chat API and interactive text input")
		}
		if *model == "" {
			return errors.New("remote chat requires -model")
		}
		if err := validateRemoteMessages([]ChatMessage{{Role: "system", Content: *system}}); err != nil {
			return err
		}
		in, closeInput := cliInput(ctx, os.Stdin)
		defer closeInput()
		return runRemoteChat(ctx, r, *system, in, os.Stdout, os.Stderr)
	}
	if *stream && *asJSON {
		return errors.New("-stream and -json cannot be combined")
	}
	msg, err := remoteInput(ctx, f, *prompt, *source, *messages, *system)
	if err != nil {
		return err
	}
	var out io.Writer
	if *stream {
		out = os.Stdout
	}
	result, err := r.complete(ctx, msg, out)
	if err != nil {
		return err
	}
	if *asJSON {
		return json.NewEncoder(stdout).Encode(result)
	}
	if *stream {
		_, err = fmt.Fprintln(stdout)
	} else {
		_, err = fmt.Fprintln(stdout, result.Text)
	}
	return err
}
func runInfer(ctx context.Context, args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		fmt.Fprintln(os.Stdout, "infer: choose where inference executes\n  local   Go CPU execution; -repo downloads and caches pinned HF files\n  remote  HF API execution; no local weights (HF_TOKEN required to generate)\nusage: monolith infer <local|remote> -h\nTraining is available under learn. Backends never switch automatically.")
		return nil
	}
	switch args[0] {
	case "models":
		f := flags("infer models")
		asJSON := f.Bool("json", false, "emit the offline capability guide")
		if err := parse(f, args[1:]); err != nil {
			return err
		}
		profiles := []struct {
			Family string `json:"family"`
			Local  string `json:"local"`
			Remote string `json:"remote"`
		}{
			{"SmolLM2 / Llama", "restricted Llama profile; tested SmolLM2 135M and 360M", "HF chat API; provider or dedicated endpoint required"},
			{"Qwen2 / Qwen2.5", "restricted Qwen2 profile; tested Qwen2.5 0.5B and 1.5B; optional Q8 weights", "HF chat API; provider or dedicated endpoint required"},
			{"GPT-2", "tied GPT2LMHeadModel, absolute positions and gelu_new; raw text", "dedicated HF text-generation endpoint"},
			{"gpt-oss", "not supported (MoE)", "HF chat API; provider or dedicated endpoint required"},
			{"Mistral", "not supported", "HF chat API; provider or dedicated endpoint required"},
			{"Gemma", "not supported", "HF chat API; provider or dedicated endpoint required"},
		}
		if *asJSON {
			return json.NewEncoder(os.Stdout).Encode(profiles)
		}
		for _, p := range profiles {
			fmt.Printf("%s\n  local:  %s\n  remote: %s\n", p.Family, p.Local, p.Remote)
		}
		fmt.Println("This is an offline capability guide. Check current hosted availability: infer remote models")
		return nil
	case "local":
		return runInferLocal(ctx, args[1:])
	case "remote":
		return runInferRemote(ctx, args[1:])
	default:
		return errors.New("unknown inference backend; choose local or remote")
	}
}
func run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		fmt.Fprintln(os.Stdout, "monolith: Go Transformer training and explicit local/remote inference\n\n  learn   train, evaluate and prepare native models\n  infer   local Go execution or remote HF inference\n\nusage: monolith learn -h | monolith infer -h\nExisting direct commands remain available: demo, tokenizer, prepare, train, eval, generate, inspect, inspect-hf, check-hf, infer-hf, tokenize-hf, chat, serve")
		return nil
	}
	switch args[0] {
	case "learn":
		return runLearn(ctx, args[1:])
	case "infer":
		return runInfer(ctx, args[1:])
	case "prepare":
		return runPrepare(ctx, args[1:])
	case "demo":
		return runDemo(ctx, args[1:])
	case "train":
		return runTrain(ctx, args[1:])
	case "tokenizer":
		return runTokenizer(args[1:])
	case "eval":
		return runEval(ctx, args[1:])
	case "generate", "inspect-hf", "infer-hf", "tokenize-hf", "check-hf", "chat", "serve":
		return runInferenceAction(ctx, args[0], args[1:], false)
	case "inspect":
		return runInspect(args[1:])
	case "help", "-h", "--help":
		return run(ctx, nil)
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}
func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:]); err != nil && !errors.Is(err, flag.ErrHelp) {
		// A stalled diagnostic pipe must not prevent exit after cleanup.
		reportCtx, cancel := context.WithTimeout(context.Background(), time.Second)
		out, closeOutput := cliOutput(reportCtx, os.Stderr)
		fmt.Fprintln(out, "error:", err)
		closeOutput()
		cancel()
		os.Exit(1)
	}
}
