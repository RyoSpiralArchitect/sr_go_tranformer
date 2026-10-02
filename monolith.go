// Monolith is a self-contained, CPU-only decoder Transformer. No cgo or ML runtime.
// Layout: configuration/RNG -> tokenizer -> kernels -> model/backprop -> cache ->
// optimizer -> checkpoint -> training -> sampling/HTTP -> CLI.
package main

import (
	"bufio"
	"bytes"
	"context"
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
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode/utf8"
)

const (
	BOS                   = 256
	EOS                   = 257
	baseVocab             = 258
	maxParameters         = 250_000_000
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
	if c.Vocab < baseVocab || c.Vocab > 65536 || c.Dim < 2 || c.Dim > 8192 || c.Hidden < 1 || c.Hidden > 32768 || c.Layers < 1 || c.Layers > 128 || c.Heads < 1 || c.Heads > c.Dim || c.KVHeads < 1 || c.KVHeads > c.Heads || c.Context < 1 || c.Context > 32768 {
		return errors.New("invalid model dimensions (see README limits)")
	}
	if c.Dim%c.Heads != 0 || c.Heads%c.KVHeads != 0 || (c.Dim/c.Heads)%2 != 0 {
		return errors.New("dim must be divisible by heads with an even head dimension; heads must be divisible by kv_heads")
	}
	if !finite(c.RopeTheta) || c.RopeTheta <= 1 || !finite(c.NormEps) || c.NormEps < 1e-12 || c.NormEps > 1 {
		return errors.New("invalid RoPE theta or RMS epsilon")
	}
	if c.ParameterCount() > maxParameters {
		return fmt.Errorf("model exceeds %d parameter allocation limit", maxParameters)
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

type Param struct {
	Name             string
	Rows, Cols       int
	Decay            bool
	Data, Grad, M, V []float32
}

// linear computes X W^T; W is contiguous [out,in].
func linear(x []float32, w *Param, n int) []float32 {
	y := make([]float32, n*w.Rows)
	linearInto(y, x, w, n)
	return y
}
func linearInto(y, x []float32, w *Param, n int) {
	in, out := w.Cols, w.Rows
	// Four sequence rows share each weight load. Disjoint 4x32 output tiles can
	// run on different goroutines, including for a single long hidden vector.
	outTiles := (out + 31) / 32
	tasks := ((n + 3) / 4) * outTiles
	parallel(tasks, n*in*out, func(lo, hi int) {
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

// Gradients accumulate, which is necessary for tied weights and microbatches.
func linearBackward(x []float32, w *Param, dy []float32, n int) []float32 {
	in, out := w.Cols, w.Rows
	dx := make([]float32, n*in)
	parallel(n, n*in*out, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for o := 0; o < out; o++ {
				g := dy[r*out+o]
				wr := w.Data[o*in : (o+1)*in]
				for i, v := range wr {
					dx[r*in+i] += g * v
				}
			}
		}
	})
	parallel(out, n*in*out, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			gw := w.Grad[o*in : (o+1)*in]
			for r := 0; r < n; r++ {
				g := dy[r*out+o]
				xr := x[r*in : (r+1)*in]
				for i, v := range xr {
					gw[i] += g * v
				}
			}
		}
	})
	return dx
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
	d := w.Cols
	dx := make([]float32, len(x))
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
	return dx
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
	for i, v := range g {
		s := 1 / (1 + math.Exp(-float64(v)))
		y[i] = float32(float64(v)*s) * u[i]
	}
	return y
}
func swigluBackward(g, u, dy []float32) ([]float32, []float32) {
	dg, du := make([]float32, len(g)), make([]float32, len(g))
	for i, v := range g {
		x := float64(v)
		s := 1 / (1 + math.Exp(-x))
		dg[i] = dy[i] * u[i] * float32(s+x*s*(1-s))
		du[i] = dy[i] * float32(x*s)
	}
	return dg, du
}

// Model owns parameters; each Forward owns its tape and each decoder its cache.
// Immutable models are safe to share between inference requests.
type Block struct{ N1, Q, K, V, O, N2, Gate, Up, Down *Param }
type Model struct {
	Config     Config
	Emb, Final *Param
	Blocks     []Block
	Params     []*Param
	Cos, Sin   []float32
	Revision   uint64
	decodePool sync.Pool
}

func NewModel(c Config, seed uint64) (*Model, error) { r := &RNG{seed}; return allocateModel(c, r) }
func allocateModel(c Config, rng *RNG) (*Model, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	m := &Model{Config: c}
	makeParam := func(name string, rows, cols int, norm bool, scale float64) *Param {
		p := &Param{Name: name, Rows: rows, Cols: cols, Decay: !norm, Data: make([]float32, rows*cols)}
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
	return m, nil
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
					ri := (offset+t)*hd/2 + j
					c, s := m.Cos[ri], m.Sin[ri]
					if inverse {
						s = -s
					}
					a, z := x[i], x[i+1]
					x[i] = a*c - z*s
					x[i+1] = a*s + z*c
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
	c := m.Config
	d, kd, hd := c.Dim, c.KVDim(), c.Dim/c.Heads
	group := c.Heads / c.KVHeads
	scale := float32(1 / math.Sqrt(float64(hd)))
	dq, dk, dv := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v))
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
	return dq, dk, dv
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

func (m *Model) Forward(ids []int, batch, seq int) (*Tape, error) {
	c := m.Config
	if batch < 1 || batch > 4096 || seq < 1 || seq > c.Context || len(ids) != batch*seq {
		return nil, errors.New("invalid input shape or context length")
	}
	n := batch * seq
	// Conservative bound includes retained tapes plus transient backward arrays.
	elements := int64(n)*(int64(c.Layers)*(12*int64(c.Dim)+4*int64(c.KVDim())+6*int64(c.Hidden))+4*int64(c.Vocab)) + 2*int64(batch)*int64(c.Heads)*int64(seq)*int64(c.Layers)
	if elements > maxActivationElements {
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
	var loss float64
	for r, target := range targets {
		if target < 0 || target >= vocab {
			return 0, nil, errors.New("target outside vocabulary")
		}
		row := logits[r*vocab : (r+1)*vocab]
		maxLogit := math.Inf(-1)
		for _, v := range row {
			x := float64(v)
			if !finite(x) {
				return 0, nil, errors.New("nonfinite logit")
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
		if gradient {
			for j, v := range row {
				g := math.Exp(float64(v)-maxLogit) / sum
				if j == target {
					g--
				}
				grad[r*vocab+j] = float32(g * scale / float64(len(targets)))
			}
		}
	}
	return loss / float64(len(targets)), grad, nil
}
func (m *Model) Backward(t *Tape, targets []int, scale float64) (float64, error) {
	if t == nil || t.Model != m || t.Revision != m.Revision || t.Used {
		return 0, errors.New("tape is foreign, stale, or already consumed")
	}
	for _, p := range m.Params {
		if len(p.Grad) != len(p.Data) {
			return 0, errors.New("call ZeroGrad before Backward")
		}
	}
	loss, dy, err := CrossEntropy(t.Logits, targets, m.Config.Vocab, scale, true)
	if err != nil {
		return 0, err
	}
	t.Used = true
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

type decodeWorkspace struct{ X, N, Q, K, V, A, G, U, H, Tmp, Logits, Scores []float32 }

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
	type poolUse struct{ pages, reserved int }
	pools := make(map[*pagePool]poolUse)
	seen := make(map[*KVCache]bool, sessions)
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
	// Five dim-sized, two KV-sized and three hidden-sized row arrays, scores,
	// and both the pooled logits and caller-owned output must fit the budget.
	elements := int64(n)*(5*int64(d)+2*int64(kd)+3*int64(c.Hidden)+int64(maxPos)) + 2*int64(sessions)*int64(c.Vocab)
	if elements > maxActivationElements {
		return nil, errors.New("inference workspace budget exceeded; reduce prefill tokens or batch size")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	group := c.Heads / c.KVHeads
	raw := m.decodePool.Get()
	var w *decodeWorkspace
	if raw == nil {
		w = &decodeWorkspace{}
	} else {
		w = raw.(*decodeWorkspace)
	}
	defer m.decodePool.Put(w)
	// Mixed prior shapes must not let retained capacities exceed the budget.
	retained := int64(sessions * c.Vocab)
	arrays := []struct {
		x *[]float32
		n int
	}{
		{&w.X, n * d}, {&w.N, n * d}, {&w.Q, n * d},
		{&w.K, n * kd}, {&w.V, n * kd}, {&w.A, n * d},
		{&w.Tmp, n * d}, {&w.G, n * c.Hidden},
		{&w.U, n * c.Hidden}, {&w.H, n * c.Hidden},
		{&w.Logits, sessions * c.Vocab}, {&w.Scores, n * maxPos},
	}
	for _, a := range arrays {
		retained += int64(max(cap(*a.x), a.n))
	}
	if retained > maxActivationElements {
		*w = decodeWorkspace{}
	}
	for _, a := range arrays {
		*a.x = resize(*a.x, a.n)
	}
	// Every session owns consecutive rows, but positions include its prefix.
	rowCaches := make([]*KVCache, n)
	positions := make([]int, n)
	lastRows := make([]int, sessions)
	r := 0
	for i, cache := range caches {
		cache.ensurePages(cache.Pos + len(chunks[i]))
		for j, token := range chunks[i] {
			rowCaches[r], positions[r] = cache, cache.Pos+j
			copy(w.X[r*d:(r+1)*d], m.Emb.Data[token*d:(token+1)*d])
			r++
		}
		lastRows[i] = r - 1
	}
	for l, b := range m.Blocks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		rmsInto(w.N, w.X, b.N1, n, c.NormEps)
		linearInto(w.Q, w.N, b.Q, n)
		linearInto(w.K, w.N, b.K, n)
		linearInto(w.V, w.N, b.V, n)
		for i, cache := range rowCaches {
			m.rope(w.Q[i*d:(i+1)*d], 1, 1, c.Heads, positions[i], false)
			m.rope(w.K[i*kd:(i+1)*kd], 1, 1, c.KVHeads, positions[i], false)
			k, v := cache.row(l, positions[i])
			copy(k, w.K[i*kd:(i+1)*kd])
			copy(v, w.V[i*kd:(i+1)*kd])
		}
		parallel(n, n*c.Heads*maxPos*hd, func(lo, hi int) {
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
		linearInto(w.Tmp, w.A, b.O, n)
		addInPlace(w.X, w.Tmp)
		rmsInto(w.N, w.X, b.N2, n, c.NormEps)
		linearInto(w.G, w.N, b.Gate, n)
		linearInto(w.U, w.N, b.Up, n)
		for i, v := range w.G {
			w.H[i] = float32(float64(v)/(1+math.Exp(-float64(v)))) * w.U[i]
		}
		linearInto(w.Tmp, w.H, b.Down, n)
		addInPlace(w.X, w.Tmp)
	}
	// Gather only each session's final hidden row before the vocabulary head.
	for i, row := range lastRows {
		copy(w.Tmp[i*d:(i+1)*d], w.X[row*d:(row+1)*d])
	}
	rmsInto(w.N[:sessions*d], w.Tmp[:sessions*d], m.Final, sessions, c.NormEps)
	linearInto(w.Logits, w.N[:sessions*d], m.Emb, sessions)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for i, cache := range caches {
		cache.Pos += len(chunks[i])
	}
	return append([]float32(nil), w.Logits...), nil
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
	Spec         TrainSpec `json:"spec"`
	Step         int       `json:"step"`
	TokensSeen   int64     `json:"tokens_seen"`
	RNG          RNG       `json:"rng"`
	CorpusSHA256 string    `json:"corpus_sha256"`
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
	return nil
}

// AdamW applies global norm clipping, bias correction and decoupled decay.
// Preflight rejects nonfinite gradients before modifying any parameter/state.
func (m *Model) AdamW(s TrainSpec, step int) (float64, error) {
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
	fail := func(err error) (*Model, Tokenizer, *TrainState, error) { return nil, Tokenizer{}, nil, err }
	f, err := os.Open(path)
	if err != nil {
		return fail(err)
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	var prefix [12]byte
	if _, err = io.ReadFull(f, prefix[:]); err != nil {
		return fail(err)
	}
	if string(prefix[:8]) != checkpointMagic {
		return fail(errors.New("unknown checkpoint magic/version"))
	}
	headerLen := binary.LittleEndian.Uint32(prefix[8:])
	if headerLen == 0 || headerLen > 4*1024*1024 {
		return fail(errors.New("checkpoint header outside allocation limit"))
	}
	data := make([]byte, int(headerLen))
	if _, err = io.ReadFull(f, data); err != nil {
		return fail(err)
	}
	var meta checkpointMeta
	if err = strictJSON(data, &meta); err != nil {
		return fail(err)
	}
	if meta.Architecture != architecture {
		return fail(errors.New("unsupported checkpoint architecture"))
	}
	if err = meta.Config.Validate(); err != nil {
		return fail(err)
	}
	if err = meta.Tokenizer.Validate(); err != nil {
		return fail(err)
	}
	if meta.Tokenizer.Vocab() != meta.Config.Vocab {
		return fail(errors.New("checkpoint vocabulary mismatch"))
	}
	arrays := int64(1)
	if meta.Training != nil {
		arrays = 3
		if err = meta.Training.Validate(meta.Config); err != nil {
			return fail(err)
		}
	}
	expected := 12 + int64(headerLen) + 4*arrays*meta.Config.ParameterCount() + sha256.Size
	if stat.Size() != expected {
		return fail(fmt.Errorf("checkpoint size mismatch: got %d, expected %d", stat.Size(), expected))
	}
	hash := sha256.New()
	if _, err = io.Copy(hash, io.NewSectionReader(f, 0, expected-sha256.Size)); err != nil {
		return fail(err)
	}
	var sum [sha256.Size]byte
	if _, err = f.ReadAt(sum[:], expected-sha256.Size); err != nil {
		return fail(err)
	}
	if !bytes.Equal(sum[:], hash.Sum(nil)) {
		return fail(errors.New("checkpoint checksum mismatch"))
	}
	m, err := allocateModel(meta.Config, nil)
	if err != nil {
		return fail(err)
	}
	if _, err = f.Seek(12+int64(headerLen), io.SeekStart); err != nil {
		return fail(err)
	}
	r := bufio.NewReaderSize(f, 64*1024)
	for _, p := range m.Params {
		if err = readFloats(r, p.Data, false); err != nil {
			return fail(err)
		}
	}
	if meta.Training != nil {
		for _, p := range m.Params {
			p.M = make([]float32, len(p.Data))
			p.V = make([]float32, len(p.Data))
			if err = readFloats(r, p.M, false); err != nil {
				return fail(err)
			}
			if err = readFloats(r, p.V, true); err != nil {
				return fail(err)
			}
		}
	}
	return m, meta.Tokenizer, meta.Training, nil
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
	spec := s.Spec
	m.ZeroGrad()
	var loss float64
	for micro := 0; micro < spec.Accum; micro++ {
		x, y := sampleBatch(data, spec.Batch, spec.Seq, &s.RNG)
		t, err := m.Forward(x, spec.Batch, spec.Seq)
		if err != nil {
			return 0, 0, err
		}
		l, err := m.Backward(t, y, 1/float64(spec.Accum))
		if err != nil {
			return 0, 0, err
		}
		loss += l / float64(spec.Accum)
	}
	norm, err := m.AdamW(spec, s.Step+1)
	if err != nil {
		return 0, 0, err
	}
	s.Step++
	s.TokensSeen += int64(spec.Batch) * int64(spec.Seq) * int64(spec.Accum)
	return loss, norm, nil
}

// Evaluation scores every post-BOS target exactly once in independent, BOS-
// framed windows. A positive maxBatches selects a deterministic leading subset.
func evaluate(ctx context.Context, m *Model, data []int, seq, maxBatches int) (float64, int, error) {
	if len(data) < 2 || data[0] != BOS || seq < 1 || seq > m.Config.Context || maxBatches < 0 {
		return 0, 0, errors.New("invalid evaluation data/settings (BOS-framed input required)")
	}
	var total float64
	tokens, batches := 0, 0
	for start := 1; start < len(data) && (maxBatches == 0 || batches < maxBatches); start += seq {
		if err := ctx.Err(); err != nil {
			return 0, tokens, err
		}
		end := min(start+seq, len(data))
		x := make([]int, end-start)
		x[0] = BOS
		copy(x[1:], data[start:end-1])
		t, err := m.Forward(x, 1, len(x))
		if err != nil {
			return 0, tokens, err
		}
		loss, _, err := CrossEntropy(t.Logits, data[start:end], m.Config.Vocab, 1, false)
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
	seen := make(map[int]bool, len(history))
	for _, id := range history {
		seen[id] = true
	}
	cs := make([]candidate, 0, len(logits)-1)
	for id, v := range logits {
		if !finite(float64(v)) {
			return 0, errors.New("nonfinite sampling logit")
		}
		if id == BOS {
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
	if s.Temperature == 0 {
		return cs[0].ID, nil
	}
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

func preparePrompt(m *Model, tok Tokenizer, prompt string, s Sampling) ([]int, error) {
	if tok.Vocab() != m.Config.Vocab {
		return nil, errors.New("tokenizer/model mismatch")
	}
	if err := s.Validate(m.Config); err != nil {
		return nil, err
	}
	if len(prompt) > 1024*1024 {
		return nil, errors.New("prompt exceeds 1 MiB")
	}
	ids := tok.Encode(prompt, true, false)
	if len(ids)+s.MaxTokens > m.Config.Context {
		return nil, fmt.Errorf("prompt (%d) + generation (%d) exceeds context (%d)", len(ids), s.MaxTokens, m.Config.Context)
	}
	return ids, nil
}
func completion(tok Tokenizer, output []int, prompt int, reason string) Completion {
	return Completion{strings.ToValidUTF8(tok.Decode(output), "\ufffd"), output, prompt, len(output), reason}
}
func Generate(ctx context.Context, m *Model, tok Tokenizer, prompt string, s Sampling) (Completion, error) {
	ids, err := preparePrompt(m, tok, prompt, s)
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
		id, e := sample(logits, history, s, rng)
		if e != nil {
			return Completion{}, e
		}
		out = append(out, id)
		history = append(history, id)
		if id == EOS {
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
}

func (d *incrementalText) Push(piece []byte, final bool) string {
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

type engineRequest struct {
	Ctx               context.Context
	IDs               []int
	Sampling          Sampling
	Reply             chan engineReply
	Events            chan TokenEvent
	Cancel            context.CancelFunc
	release           func()
	Started, Enqueued time.Time
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

// Wait returns the terminal result. Call once, after draining Events, unless
// abandoning the stream. Cancellation never requires draining its event buffer.
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
}
type EngineStats struct {
	Submitted        int64 `json:"submitted"`
	Completed        int64 `json:"completed"`
	Canceled         int64 `json:"canceled"`
	Failed           int64 `json:"failed"`
	Rejected         int64 `json:"rejected"`
	Active           int64 `json:"active"`
	Queue            int   `json:"queue"`
	KVReservedBytes  int64 `json:"kv_reserved_bytes"`
	KVAllocatedBytes int64 `json:"kv_allocated_bytes"`
	InputTokens      int64 `json:"input_tokens"`
	GeneratedTokens  int64 `json:"generated_tokens"`
	Ticks            int64 `json:"ticks"`
	SlowConsumers    int64 `json:"slow_consumers"`
	PeakTickTokens   int64 `json:"peak_tick_tokens"`
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
	Model                                     *Model
	Tokenizer                                 Tokenizer
	Config                                    EngineConfig
	queue                                     chan *engineRequest
	admission                                 chan struct{}
	httpSlots                                 chan struct{}
	ctx                                       context.Context
	cancel                                    context.CancelFunc
	done                                      chan struct{}
	pool                                      pagePool
	stats                                     engineCounters
	pieces                                    [][]byte
	submitMu                                  sync.Mutex
	queueLatency, firstTokenLatency, tokenGap latencyHistogram
}

func NewEngine(m *Model, tok Tokenizer, c EngineConfig) (*Engine, error) {
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
	if c.MaxBatch < 1 || c.MaxBatch > 256 || c.Queue < 1 || c.Queue > 65536 || c.CacheBytes < 1 || c.CacheBytes > 1<<40 || c.StreamBuffer < 1 || c.StreamBuffer > 4096 || c.PrefillChunk < 1 || c.PrefillChunk > 1024 || c.MixedPrefillChunk < 1 || c.MixedPrefillChunk > 1024 || c.TokenBudget < c.MaxBatch || c.TokenBudget > 1024 {
		return nil, errors.New("invalid engine capacity or token budget")
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
	go e.run()
	return e, nil
}
func (e *Engine) Stats() EngineStats {
	s := &e.stats
	return EngineStats{Submitted: s.Submitted.Load(), Completed: s.Completed.Load(), Canceled: s.Canceled.Load(), Failed: s.Failed.Load(), Rejected: s.Rejected.Load(), Active: s.Active.Load(), Queue: len(e.queue), KVReservedBytes: s.Reserved.Load(), KVAllocatedBytes: s.Allocated.Load(), InputTokens: s.InputTokens.Load(), GeneratedTokens: s.GeneratedTokens.Load(), Ticks: s.Ticks.Load(), SlowConsumers: s.SlowConsumers.Load(), PeakTickTokens: s.PeakTickTokens.Load()}
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
	ids, err := preparePrompt(e.Model, e.Tokenizer, prompt, s)
	if err != nil {
		return nil, err
	}
	pages := (len(ids) + s.MaxTokens + pageTokens - 1) / pageTokens
	if pages > e.pool.Limit {
		return nil, errors.New("request exceeds total KV cache budget")
	}
	ctx, cancel := context.WithCancel(ctx)
	r := &engineRequest{Ctx: ctx, IDs: ids, Sampling: s, Reply: make(chan engineReply, 1), Cancel: cancel, release: func() { <-e.admission }, Started: started, Enqueued: time.Now()}
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
		// The terminal mailbox is independent of the bounded token mailbox.
		// Neither delivery nor closing it can block the actor on a network reader.
		r.Reply <- engineReply{out, err}
		if r.Events != nil {
			close(r.Events)
		}
		r.release()
		r.Cancel()
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
			sessions = append(sessions, &engineSession{Request: r, Cache: cache, History: append([]int(nil), r.IDs...), RNG: RNG{r.Sampling.Seed}})
			e.stats.Active.Add(1)
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
		logits, err := e.Model.PrefillBatch(e.ctx, caches[:n], chunks[:n])
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
			id, err := sample(logits[i*e.Model.Config.Vocab:(i+1)*e.Model.Config.Vocab], s.History, s.Request.Sampling, &s.RNG)
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
			if id == EOS {
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
			writeJSON(w, 200, map[string]any{"status": "ok", "parameters": e.Model.Config.ParameterCount(), "stats": e.Stats()})
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
		e.queueLatency.write(w, "monolith_queue_wait_seconds")
		e.firstTokenLatency.write(w, "monolith_first_token_seconds")
		e.tokenGap.write(w, "monolith_token_gap_seconds")
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
			Prompt string `json:"prompt"`
			Stream bool   `json:"stream"`
			Sampling
		}{Sampling: defaultSampling()}
		if err = strictJSON(body, &req); err != nil {
			writeInferenceError(w, err)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
		defer cancel()
		if req.Stream {
			e.serveStream(w, r, ctx, req.Prompt, req.Sampling)
			return
		}
		out, err := e.Generate(ctx, req.Prompt, req.Sampling)
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
func freshModel(presetName, configPath, tokenizerPath string, seed uint64) (*Model, Tokenizer, error) {
	tok := Tokenizer{}
	if tokenizerPath != "" {
		if err := readJSONFile(tokenizerPath, &tok); err != nil {
			return nil, tok, err
		}
		if err := tok.Validate(); err != nil {
			return nil, tok, err
		}
	}
	c, err := preset(presetName)
	if err != nil {
		return nil, tok, err
	}
	if configPath != "" {
		if err = readJSONFile(configPath, &c); err != nil {
			return nil, tok, err
		}
	}
	c.Vocab = tok.Vocab()
	m, err := NewModel(c, seed)
	return m, tok, err
}
func ensureMoments(m *Model) {
	for _, p := range m.Params {
		if len(p.M) == 0 {
			p.M = make([]float32, len(p.Data))
			p.V = make([]float32, len(p.Data))
		}
	}
}

type loopOptions struct {
	Out                                                    string
	StopAfter, SaveEvery, EvalEvery, EvalBatches, LogEvery int
}

func trainLoop(ctx context.Context, m *Model, tok Tokenizer, s *TrainState, train, val []int, o loopOptions) error {
	if err := s.Validate(m.Config); err != nil {
		return err
	}
	if o.StopAfter < 0 || o.SaveEvery < 0 || o.EvalEvery < 0 || o.EvalBatches < 0 || o.LogEvery < 1 {
		return errors.New("invalid logging/save/evaluation interval")
	}
	ensureMoments(m)
	initial, nt, err := evaluate(ctx, m, val, s.Spec.Seq, o.EvalBatches)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "parameters=%d fp32_weights_mib=%.2f train_tokens=%d val_tokens=%d workers=%d\n", m.Config.ParameterCount(), float64(m.Config.ParameterCount()*4)/(1<<20), len(train), len(val), runtime.GOMAXPROCS(0))
	fmt.Fprintf(os.Stderr, "step=%d val_loss=%.6f val_targets=%d\n", s.Step, initial, nt)
	start := time.Now()
	initialTokens := s.TokensSeen
	target := s.Spec.Steps
	if o.StopAfter > 0 {
		target = min(target, s.Step+o.StopAfter)
	}
	for s.Step < target {
		if ctx.Err() != nil {
			break
		}
		loss, norm, err := trainUpdate(m, s, train)
		if err != nil {
			return err
		}
		if s.Step%o.LogEvery == 0 || s.Step == target {
			fmt.Fprintf(os.Stderr, "step=%d/%d loss=%.6f lr=%.7f grad_norm=%.4f tokens_per_sec=%.1f\n", s.Step, s.Spec.Steps, loss, s.Spec.LearningRate(s.Step), norm, float64(s.TokensSeen-initialTokens)/time.Since(start).Seconds())
		}
		if o.EvalEvery > 0 && s.Step%o.EvalEvery == 0 {
			value, count, e := evaluate(ctx, m, val, s.Spec.Seq, o.EvalBatches)
			if e != nil {
				if ctx.Err() != nil {
					break
				}
				return e
			}
			fmt.Fprintf(os.Stderr, "step=%d val_loss=%.6f val_targets=%d\n", s.Step, value, count)
		}
		if o.SaveEvery > 0 && s.Step%o.SaveEvery == 0 {
			if err = SaveCheckpoint(o.Out, m, tok, s); err != nil {
				return err
			}
		}
	}
	if err = SaveCheckpoint(o.Out, m, tok, s); err != nil {
		return err
	}
	if ctx.Err() == nil {
		value, count, e := evaluate(ctx, m, val, s.Spec.Seq, o.EvalBatches)
		if e != nil {
			return e
		}
		fmt.Fprintf(os.Stderr, "final_step=%d val_loss=%.6f val_targets=%d checkpoint=%s\n", s.Step, value, count, o.Out)
	} else {
		fmt.Fprintf(os.Stderr, "interrupted; checkpoint saved at completed update %d: %s\n", s.Step, o.Out)
	}
	return nil
}
func runTrain(ctx context.Context, args []string) error {
	f := flags("train")
	spec := defaultTrainSpec()
	data := f.String("data", "", "UTF-8 or arbitrary byte text file (max 64 MiB)")
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
	f.IntVar(&o.StopAfter, "stop-after", 0, "stop after this many additional updates, retaining full schedule")
	f.IntVar(&o.SaveEvery, "save-every", 50, "checkpoint interval; 0 disables periodic saves")
	f.IntVar(&o.EvalEvery, "eval-every", 25, "validation interval; 0 disables periodic validation")
	f.IntVar(&o.EvalBatches, "eval-batches", 8, "deterministic validation windows; 0 means all")
	f.IntVar(&o.LogEvery, "log-every", 10, "training log interval")
	if err := parse(f, args); err != nil {
		return err
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	text, err := readText(*data, 64<<20)
	if err != nil {
		return err
	}
	o.Out = *out
	var m *Model
	var tok Tokenizer
	var state *TrainState
	if *resume != "" {
		allowed := map[string]bool{"resume": true, "data": true, "out": true, "threads": true, "stop-after": true, "save-every": true, "eval-every": true, "eval-batches": true, "log-every": true}
		var bad string
		f.Visit(func(v *flag.Flag) {
			if !allowed[v.Name] {
				bad = v.Name
			}
		})
		if bad != "" {
			return fmt.Errorf("-%s cannot override a resumed run; its schedule/tokenizer/model are saved", bad)
		}
		m, tok, state, err = LoadCheckpoint(*resume)
		if err != nil {
			return err
		}
		if state == nil {
			return errors.New("checkpoint has no training state")
		}
		if state.CorpusSHA256 != corpusHash(text) {
			return errors.New("resume corpus SHA-256 differs from original")
		}
		if state.Step >= state.Spec.Steps {
			return errors.New("saved training schedule is already complete")
		}
	} else {
		m, tok, err = freshModel(*pre, *config, *tokenizer, *seed)
		if err != nil {
			return err
		}
		if err = spec.Validate(m.Config); err != nil {
			return err
		}
		state = &TrainState{Spec: spec, RNG: RNG{*seed ^ 0xd1b54a32d192ed03}, CorpusSHA256: corpusHash(text)}
	}
	train, val, err := splitCorpus(tok, text, state.Spec.ValFraction, state.Spec.Seq)
	if err != nil {
		return err
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
func runGenerate(ctx context.Context, args []string) error {
	f := flags("generate")
	model := f.String("model", "runs/model.mglm", "checkpoint")
	prompt := f.String("prompt", "", "text prompt")
	asJSON := f.Bool("json", false, "emit structured completion")
	threads := threadFlag(f)
	s := defaultSampling()
	samplingFlags(f, &s)
	if err := parse(f, args); err != nil {
		return err
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	m, tok, _, err := LoadCheckpoint(*model)
	if err != nil {
		return err
	}
	m.DropOptimizer()
	out, err := Generate(ctx, m, tok, *prompt, s)
	if err != nil {
		return err
	}
	if *asJSON {
		return json.NewEncoder(os.Stdout).Encode(out)
	}
	fmt.Print(*prompt + out.Text + "\n")
	return nil
}
func runEval(ctx context.Context, args []string) error {
	f := flags("eval")
	model := f.String("model", "runs/model.mglm", "checkpoint")
	data := f.String("data", "", "evaluation text")
	seq := f.Int("seq", 0, "context window (default min(context,128))")
	batches := f.Int("batches", 0, "maximum windows; 0 evaluates all targets")
	threads := threadFlag(f)
	if err := parse(f, args); err != nil {
		return err
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	m, tok, _, err := LoadCheckpoint(*model)
	if err != nil {
		return err
	}
	m.DropOptimizer()
	text, err := readText(*data, 64<<20)
	if err != nil {
		return err
	}
	if *seq == 0 {
		*seq = min(128, m.Config.Context)
	}
	loss, count, err := evaluate(ctx, m, tok.Encode(text, true, true), *seq, *batches)
	if err != nil {
		return err
	}
	ppl := math.Exp(loss)
	var perplexity any = ppl
	if !finite(ppl) {
		perplexity = "overflow"
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"loss": loss, "perplexity": perplexity, "targets": count, "sequence_length": *seq})
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

func runServe(ctx context.Context, args []string) error {
	f := flags("serve")
	model := f.String("model", "runs/model.mglm", "checkpoint")
	addr := f.String("addr", "127.0.0.1:8080", "HTTP listen address")
	batch := f.Int("max-batch", 8, "maximum simultaneous sessions per scheduling tick")
	chunk := f.Int("prefill-chunk", 16, "maximum prompt tokens per session per tick; 1 is the sequential baseline")
	mixedChunk := f.Int("mixed-prefill-chunk", 1, "maximum prompt tokens per session when another session is generating")
	budget := f.Int("token-budget", 0, "total tokens per tick; 0 selects max(max-batch,64)")
	buffer := f.Int("stream-buffer", 32, "buffered token events per stream before slow-consumer termination")
	queue := f.Int("queue", 32, "bounded request queue capacity")
	memory := f.Int64("kv-mib", 256, "hard KV page budget in MiB")
	threads := threadFlag(f)
	if err := parse(f, args); err != nil {
		return err
	}
	if err := setThreads(*threads); err != nil {
		return err
	}
	if *memory < 1 || *memory > 1<<20 {
		return errors.New("invalid -kv-mib")
	}
	m, tok, _, err := LoadCheckpoint(*model)
	if err != nil {
		return err
	}
	m.DropOptimizer()
	e, err := NewEngine(m, tok, EngineConfig{MaxBatch: *batch, Queue: *queue, CacheBytes: *memory << 20, PrefillChunk: *chunk, MixedPrefillChunk: *mixedChunk, TokenBudget: *budget, StreamBuffer: *buffer})
	if err != nil {
		return err
	}
	defer e.cancel() // A deferred blocking Close would bypass the shutdown deadline.
	server := &http.Server{Addr: *addr, Handler: e.Handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 125 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	defer server.Close()
	result := make(chan error, 1)
	go func() { result <- server.ListenAndServe() }()
	fmt.Fprintf(os.Stderr, "serving %s parameters=%d max_batch=%d queue=%d kv_mib=%d workers=%d prefill_chunk=%d mixed_prefill_chunk=%d token_budget=%d stream_buffer=%d\n", *addr, m.Config.ParameterCount(), *batch, *queue, *memory, runtime.GOMAXPROCS(0), e.Config.PrefillChunk, e.Config.MixedPrefillChunk, e.Config.TokenBudget, e.Config.StreamBuffer)
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
func run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		fmt.Fprintln(os.Stdout, "monolith: one Go binary for Transformer training and bounded continuous-batch inference\n\ncommands: demo, tokenizer, train, eval, generate, inspect, serve\nusage: monolith <command> -h")
		return nil
	}
	switch args[0] {
	case "demo":
		return runDemo(ctx, args[1:])
	case "train":
		return runTrain(ctx, args[1:])
	case "tokenizer":
		return runTokenizer(args[1:])
	case "eval":
		return runEval(ctx, args[1:])
	case "generate":
		return runGenerate(ctx, args[1:])
	case "inspect":
		return runInspect(args[1:])
	case "serve":
		return runServe(ctx, args[1:])
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
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
