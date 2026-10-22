package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func hfCLIFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	h := hfTestConfig()
	h.Vocab, h.Context = tok.Vocab(), 8192
	hfWriteJSON(t, filepath.Join(dir, "config.json"), h)
	_, c, err := readHFConfig(dir, 64)
	if err != nil {
		t.Fatal(err)
	}
	header := map[string]any{}
	var payload []byte
	for _, spec := range hfTensorSpecs(c) {
		n := 1
		for _, d := range spec.Shape {
			n *= d
		}
		start := len(payload)
		payload = append(payload, make([]byte, n*4)...)
		header[spec.Name] = map[string]any{"dtype": "F32", "shape": spec.Shape, "data_offsets": []int{start, len(payload)}}
	}
	hfWriteShard(t, filepath.Join(dir, "model.safetensors"), header, payload)
	for _, name := range []string{"tokenizer.json", "tokenizer_config.json"} {
		b, err := os.ReadFile(filepath.Join("testdata/hf-tokenizer", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestHFCheckDefaultsAndPayloadBoundary(t *testing.T) {
	dir := hfCLIFixture(t)
	r, err := checkHF(dir, 0, 2048, true)
	if err != nil {
		t.Fatal(err)
	}
	if r.Weights.Config.Context != 2048 || r.ModelContext != 8192 || !r.ChatSupported || r.WeightValuesChecked {
		t.Fatal(r)
	}
	explicit, err := checkHF(dir, 33, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if explicit.KVPageAlignedBytes != 64*explicit.Weights.KVBytesPerToken {
		t.Fatal("KV page rounding", explicit)
	}
	// Preflight deliberately checks metadata, not payload values or model quality.
	path := filepath.Join(dir, "model.safetensors")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	offset := 8 + binary.LittleEndian.Uint64(b[:8])
	binary.LittleEndian.PutUint32(b[offset:], math.Float32bits(float32(math.NaN())))
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := checkHF(dir, 64, 1, true); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadHFModel(context.Background(), dir, 64, 1); err == nil {
		t.Fatal("load accepted nonfinite payload")
	}
}

func TestCLIInputSources(t *testing.T) {
	text := "猫\nGo\n"
	b, err := readCLIInput(context.Background(), "-", strings.NewReader(text))
	if err != nil || string(b) != text {
		t.Fatal(string(b), err)
	}
	if _, err := readCLIInput(context.Background(), "-", strings.NewReader(strings.Repeat("x", (1<<20)+1))); err == nil {
		t.Fatal("unbounded stdin")
	}
	path := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(path, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	for _, chat := range []bool{false, true} {
		f := flags("test")
		f.String("prompt-file", path, "")
		got, err := inferencePrompt(context.Background(), f, tok, "", chat, "")
		if err != nil {
			t.Fatal(err)
		}
		want := text
		if chat {
			want, _ = tok.formatChat([]ChatMessage{{Role: "user", Content: text}})
		}
		if got != want {
			t.Fatal(got, want)
		}
	}
	f := flags("test")
	f.String("prompt-file", path, "")
	f.String("prompt", "", "")
	if err := f.Set("prompt", "literal"); err != nil {
		t.Fatal(err)
	}
	if _, err := inferencePrompt(context.Background(), f, tok, "literal", false, ""); err == nil {
		t.Fatal("ambiguous prompt inputs")
	}
	if _, err := inferencePrompt(context.Background(), f, tok, "", false, path); err == nil {
		t.Fatal("file plus messages")
	}
}

func TestCLIInputCancellation(t *testing.T) {
	in, producer := io.Pipe()
	defer in.Close()
	defer producer.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := readCLIInput(ctx, "-", in)
		done <- err
	}()
	// Deliver a partial input without EOF, then cancel while the source stays open.
	if _, err := producer.Write([]byte("unfinished")); err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("stdin cancellation waited for input or EOF")
	}
}

// Exercise an inherited descriptor, which need not support Go's file poller.
func TestChatIdleStdinInterrupt(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("os.Process.Signal(os.Interrupt) is not implemented on Windows")
	}
	if dir := os.Getenv("MONOLITH_TEST_CHAT_CHILD"); dir != "" {
		os.Args = []string{"monolith", "chat", "-hf-model", dir, "-max-tokens", "1", "-kernel-workers", "2"}
		main()
		os.Exit(0)
	}
	dir := hfCLIFixture(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestChatIdleStdinInterrupt$")
	cmd.Env = append(os.Environ(), "MONOLITH_TEST_CHAT_CHILD="+dir)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer stdin.Close()
	logPath := filepath.Join(t.TempDir(), "stderr")
	log, err := os.Create(logPath)
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()
	cmd.Stderr = log
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	waited := false
	defer func() {
		cancel()
		if !waited {
			<-done
		}
	}()
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case err := <-done:
			waited = true
			t.Fatal("chat exited before input wait", err)
		case <-tick.C:
			b, err := os.ReadFile(logPath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(b, []byte("Chat ready.")) {
				continue
			}
			if err := cmd.Process.Signal(os.Interrupt); err != nil {
				t.Fatal(err)
			}
			err = <-done
			waited = true
			b, _ = os.ReadFile(logPath)
			if cmd.ProcessState.ExitCode() != 1 || ctx.Err() != nil || !bytes.Contains(b, []byte("context canceled")) {
				t.Fatalf("idle stdin interrupt: %v %s", err, b)
			}
			return
		}
	}
}

type chatBrokenWriter struct{}

func (chatBrokenWriter) Write(b []byte) (int, error) { return 0, io.ErrClosedPipe }

func TestChatHistoryResetAndRelease(t *testing.T) {
	tok := testHFTokenizer(t, "testdata/hf-tokenizer")
	m := hfServingModel(t, tok, -1)
	s := defaultSampling()
	s.MaxTokens, s.Temperature, s.RepeatPenalty = 1, 0, 1
	e, err := NewEngine(m, tok, EngineConfig{MaxBatch: 1, Queue: 1, CacheBytes: 2 << 20, KernelWorkers: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	var out, status bytes.Buffer
	first, _ := tok.formatChat([]ChatMessage{{Role: "user", Content: "Hello"}})
	want, err := Generate(context.Background(), m, tok, first, s)
	if err != nil {
		t.Fatal(err)
	}
	second, _ := tok.formatChat([]ChatMessage{{Role: "user", Content: "Hello"}, {Role: "assistant", Content: want.Text}, {Role: "user", Content: "Again"}})
	want2, err := Generate(context.Background(), m, tok, second, s)
	if err != nil {
		t.Fatal(err)
	}
	input := "Hello\nAgain\n/reset\nHello\n/exit\nignored\n"
	if err := chatLoop(context.Background(), e, tok, "", s, strings.NewReader(input), &out, &status); err != nil {
		t.Fatal(err)
	}
	if out.String() != want.Text+"\n"+want2.Text+"\n"+want.Text+"\n" {
		t.Fatal("conversation mismatch", out.String())
	}
	stats := e.Stats()
	if stats.Released != 3 || stats.AdmissionUsed != 0 || stats.KVReservedBytes != 0 {
		t.Fatal(stats)
	}
	out.Reset()
	status.Reset()
	input = strings.Repeat("x", 10000) + "\n/reset\nHello\n/exit\n"
	if err := chatLoop(context.Background(), e, tok, "", s, strings.NewReader(input), &out, &status); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(status.String(), "Cannot add turn") || out.String() != want.Text+"\n" {
		t.Fatal(status.String(), out.String())
	}
	if err := chatLoop(context.Background(), e, tok, "", s, strings.NewReader("Hello\n"), chatBrokenWriter{}, io.Discard); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
	stats = e.Stats()
	if stats.AdmissionUsed != 0 || stats.KVReservedBytes != 0 {
		t.Fatal("write failure leaked stream", stats)
	}
}
