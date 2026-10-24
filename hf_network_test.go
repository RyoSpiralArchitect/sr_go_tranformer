package main

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func fakeHFHub(t *testing.T, corrupt bool) (hfHub, hfSource, func() map[string]int) {
	t.Helper()
	dir := hfCLIFixture(t)
	files := map[string][]byte{}
	var rows []hfHubFile
	for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		hash := sha1.New()
		fmt.Fprintf(hash, "blob %d%c", len(b), 0)
		hash.Write(b)
		rows = append(rows, hfHubFile{Name: name, Size: int64(len(b)), Blob: hex.EncodeToString(hash.Sum(nil))})
		files[name] = b
	}
	if corrupt {
		files["model.safetensors"][len(files["model.safetensors"])-1] ^= 1
	}
	sha := strings.Repeat("a", 40)
	var mu sync.Mutex
	requests := map[string]int{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests[r.URL.Path]++
		mu.Unlock()
		if strings.HasPrefix(r.URL.Path, "/api/models/") {
			json.NewEncoder(w).Encode(hfHubInfo{SHA: sha, Files: rows})
			return
		}
		prefix := "/author/toy/resolve/" + sha + "/"
		if !strings.HasPrefix(r.URL.Path, prefix) {
			http.Error(w, "unpinned path", 400)
			return
		}
		data, ok := files[strings.TrimPrefix(r.URL.Path, prefix)]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.Write(data)
	}))
	t.Cleanup(server.Close)
	h := hfHub{Base: server.URL, Client: server.Client()}
	s := hfSource{Repo: "author/toy", Revision: "main", Cache: t.TempDir(), Context: 64, WeightMiB: 1, DownloadBytes: 2 << 20}
	counts := func() map[string]int {
		mu.Lock()
		defer mu.Unlock()
		out := map[string]int{}
		for k, v := range requests {
			out[k] = v
		}
		return out
	}
	return h, s, counts
}
func TestHFHubCacheOfflineAndIntegrity(t *testing.T) {
	h, s, counts := fakeHFHub(t, false)
	dir, manifest, err := h.resolve(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Revision != strings.Repeat("a", 40) || len(manifest.Files) != 4 {
		t.Fatal(manifest)
	}
	before := fmt.Sprint(counts())
	s.Offline = true
	cached, _, err := h.resolve(context.Background(), s)
	if err != nil || cached != dir {
		t.Fatal(cached, err)
	}
	if fmt.Sprint(counts()) != before {
		t.Fatal("offline made network requests")
	}
	path := filepath.Join(dir, "model.safetensors")
	b, _ := os.ReadFile(path)
	b[len(b)-1] ^= 1
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err = h.resolve(context.Background(), s); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatal("cache corruption accepted", err)
	}
	s.Revision = strings.Repeat("b", 40)
	if _, _, err = h.resolve(context.Background(), s); err == nil {
		t.Fatal("offline cache miss accepted")
	}
	if fmt.Sprint(counts()) != before {
		t.Fatal("offline miss made network requests")
	}
}
func TestHFHubBoundsAndAtomicPublication(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		t.Run(fmt.Sprint(corrupt), func(t *testing.T) {
			h, s, counts := fakeHFHub(t, corrupt)
			if !corrupt {
				s.DownloadBytes = 1024
			}
			if _, _, err := h.resolve(context.Background(), s); err == nil {
				t.Fatal("invalid download accepted")
			}
			entries, _ := os.ReadDir(filepath.Join(s.Cache, hfKey(s.Repo)))
			for _, e := range entries {
				if strings.HasPrefix(e.Name(), ".download-") || validHFCommit(e.Name()) {
					t.Fatal("partial snapshot left behind", e.Name())
				}
			}
			if !corrupt {
				for path := range counts() {
					if strings.HasSuffix(path, "model.safetensors") {
						t.Fatal("over-budget weights were requested")
					}
				}
			}
		})
	}
	h, s, _ := fakeHFHub(t, false)
	results := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() { _, _, err := h.resolve(context.Background(), s); results <- err }()
	}
	for i := 0; i < 2; i++ {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
}
func TestHFHubPinnedCacheCanRememberNewRef(t *testing.T) {
	h, s, counts := fakeHFHub(t, false)
	s.Revision = strings.Repeat("a", 40)
	if _, _, err := h.resolve(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	s.Revision = "main"
	if _, _, err := h.resolve(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	before := fmt.Sprint(counts())
	s.Offline = true
	if _, _, err := h.resolve(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(counts()) != before {
		t.Fatal("offline ref used network")
	}
}
func TestHFRedirectCredentialBoundary(t *testing.T) {
	target := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("token forwarded across origin")
		}
		io.WriteString(w, "ok")
	}))
	defer target.Close()
	source := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer private-test-token" {
			t.Error("source missing auth")
		}
		http.Redirect(w, r, target.URL+"/signed?signature=private", http.StatusFound)
	}))
	defer source.Close()
	client := source.Client()
	client.CheckRedirect = newHFClient(time.Second).CheckRedirect
	req, _ := http.NewRequest(http.MethodGet, source.URL, nil)
	req.Header.Set("Authorization", "Bearer private-test-token")
	resp, err := hfDo(context.Background(), client, req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	redirect, _ := url.Parse("http://huggingface.co/file")
	if err := client.CheckRedirect(&http.Request{URL: redirect, Header: make(http.Header)}, []*http.Request{req}); err == nil {
		t.Fatal("HTTP downgrade accepted")
	}
}

func testRemote(server *httptest.Server) hfRemote {
	return hfRemote{Base: server.URL, Token: "private-test-token", Model: "openai/gpt-oss-20b", API: "chat", Client: server.Client(), MaxTokens: 8, Temperature: 0, TopP: 1}
}
func TestHFRemoteChatAndStreamProtocol(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprint(streaming), func(t *testing.T) {
			count := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				count++
				if r.URL.Path != "/chat/completions" || r.Header.Get("Authorization") != "Bearer private-test-token" {
					t.Error("incorrect request")
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if body["model"] != "openai/gpt-oss-20b" || body["stream"] != streaming || body["max_tokens"] != float64(8) {
					t.Error(body)
				}
				if _, ok := body["top_k"]; ok {
					t.Error("local sampling leaked into API")
				}
				if !streaming {
					io.WriteString(w, `{"model":"served-model","choices":[{"index":0,"message":{"content":"猫🐈‍⬛"},"finish_reason":"stop"}],"usage":{"total_tokens":8}}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				data := ": keepalive\n\n" + "data: " + `{"model":"served-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n" +
					"data: " + `{"choices":[{"index":0,"delta":{"content":"猫🐈‍⬛"},"finish_reason":null}]}` + "\n\n" +
					"data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
					"data: " + `{"choices":[],"usage":{"total_tokens":8}}` + "\n\n" + "data: [DONE]\n\n"
				for i := 0; i < len(data); i += 3 {
					w.Write([]byte(data[i:min(i+3, len(data))]))
					w.(http.Flusher).Flush()
				}
			}))
			defer server.Close()
			var out bytes.Buffer
			var writer io.Writer
			if streaming {
				writer = &out
			}
			got, err := testRemote(server).complete(context.Background(), []ChatMessage{{Role: "user", Content: "test"}}, writer)
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != "猫🐈‍⬛" || got.Model != "served-model" || got.FinishReason != "stop" || got.Backend != "hf-remote" {
				t.Fatal(got)
			}
			if streaming && out.String() != got.Text {
				t.Fatal(out.String())
			}
			if count != 1 {
				t.Fatal("request was retried", count)
			}
		})
	}
}
func TestHFRemoteFailureBoundaries(t *testing.T) {
	cases := []struct {
		name, body string
		status     int
		stream     bool
	}{
		{"rate limit", "private-test-token", 429, false},
		{"auth", "private-test-token", 401, false},
		{"truncated", "data: " + `{"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}` + "\n\n", 200, true},
		{"provider error", "data: " + `{"error":{"message":"private-test-token"}}` + "\n\n", 200, true},
		{"premature done", "data: [DONE]\n\n", 200, true},
		{"oversized event", "data: " + strings.Repeat("x", 300<<10) + "\n\n", 200, true},
		{"bad finish", `{"choices":[{"index":0,"message":{"content":"incomplete"}}]}`, 200, false},
		{"tool call", `{"choices":[{"index":0,"message":{"tool_calls":[{}]},"finish_reason":"tool_calls"}]}`, 200, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			count := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				count++
				if tc.stream {
					w.Header().Set("Content-Type", "text/event-stream")
				}
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			}))
			defer server.Close()
			var writer io.Writer
			if tc.stream {
				writer = io.Discard
			}
			_, err := testRemote(server).complete(context.Background(), []ChatMessage{{Role: "user", Content: "test"}}, writer)
			if err == nil || strings.Contains(err.Error(), "private-test-token") {
				t.Fatal("missing/sensitive error", err)
			}
			if count != 1 {
				t.Fatal("implicit retry", count)
			}
		})
	}
}
func TestHFRemoteCancellationAndBrokenOutput(t *testing.T) {
	for _, broken := range []bool{false, true} {
		t.Run(fmt.Sprint(broken), func(t *testing.T) {
			started, released := make(chan struct{}), make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, "data: "+`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`+"\n\n")
				w.(http.Flusher).Flush()
				close(started)
				<-r.Context().Done()
				close(released)
			}))
			defer server.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			var out io.Writer = io.Discard
			if broken {
				out = chatBrokenWriter{}
			}
			go func() {
				_, err := testRemote(server).complete(ctx, []ChatMessage{{Role: "user", Content: "test"}}, out)
				done <- err
			}()
			<-started
			if !broken {
				cancel()
			}
			select {
			case err := <-done:
				if broken && !errors.Is(err, io.ErrClosedPipe) || !broken && !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("request did not cancel")
			}
			select {
			case <-released:
			case <-time.After(3 * time.Second):
				t.Fatal("HTTP body/connection retained")
			}
		})
	}
}
func TestHFRemoteTextEndpoint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["inputs"] != "The cat" {
			t.Error(body)
		}
		if _, ok := body["messages"]; ok {
			t.Error("chat was sent to text endpoint")
		}
		io.WriteString(w, `[{"generated_text":" sat down."}]`)
	}))
	defer server.Close()
	r := testRemote(server)
	r.API = "text"
	r.Model = "openai-community/gpt2"
	result, err := r.complete(context.Background(), []ChatMessage{{Role: "user", Content: "The cat"}}, nil)
	if err != nil || result.Text != " sat down." {
		t.Fatal(result, err)
	}
}

func TestHFRemoteBlockedOutput(t *testing.T) {
	for _, timeout := range []bool{false, true} {
		t.Run(fmt.Sprintf("timeout=%t", timeout), func(t *testing.T) {
			released := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, "data: "+`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`+"\n\n")
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				close(released)
			}))
			defer server.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			reader, writer := io.Pipe()
			defer reader.Close()
			defer writer.Close()
			blocked := &stalledCLIWriter{writer: writer, started: make(chan struct{})}
			r := testRemote(server)
			if timeout {
				r.Client.Timeout = 500 * time.Millisecond
			}
			done := make(chan error, 1)
			go func() {
				_, err := r.complete(ctx, []ChatMessage{{Role: "user", Content: "test"}}, blocked)
				done <- err
			}()
			select {
			case <-blocked.started:
			case <-time.After(3 * time.Second):
				t.Fatal("did not attempt output")
			}
			want := context.Canceled
			if timeout {
				want = context.DeadlineExceeded
			} else {
				cancel()
			}
			select {
			case err := <-done:
				if !errors.Is(err, want) {
					t.Fatal(err)
				}
			case <-time.After(2 * time.Second):
				_ = reader.Close()
				<-done
				t.Fatal("remote cancellation/timeout waited for the output reader")
			}
			select {
			case <-released:
			case <-time.After(3 * time.Second):
				t.Fatal("HTTP body retained")
			}
		})
	}
}

func TestHFRemoteChatBlockedStatus(t *testing.T) {
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	blocked := &stalledCLIWriter{writer: writer, started: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- runRemoteChat(ctx, hfRemote{}, "", strings.NewReader(""), io.Discard, blocked) }()
	select {
	case <-blocked.started:
	case <-time.After(3 * time.Second):
		t.Fatal("did not attempt status")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		_ = reader.Close()
		<-done
		t.Fatal("remote status did not cancel")
	}
}

func TestHFRemoteChatDeadlineIncludesFinalNewline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: "+`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	blocked := &stalledCLIWriter{writer: writer, started: make(chan struct{}), match: "\n"}
	r := testRemote(server)
	r.Client.Timeout = 500 * time.Millisecond
	done := make(chan error, 1)
	go func() {
		done <- runRemoteChat(context.Background(), r, "", strings.NewReader("hi\n"), blocked, io.Discard)
	}()
	select {
	case <-blocked.started:
	case <-time.After(3 * time.Second):
		t.Fatal("did not attempt newline")
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		_ = reader.Close()
		<-done
		t.Fatal("turn timeout did not include final output")
	}
}

func TestHFLocalInvalidActionDoesNotFetch(t *testing.T) {
	// Intercept dialing rather than making any external requests. The client
	// clones DefaultTransport, so this also observes accidental source fetches.
	original := http.DefaultTransport
	transport := original.(*http.Transport).Clone()
	var dials atomic.Int32
	transport.DialContext = func(context.Context, string, string) (net.Conn, error) {
		dials.Add(1)
		return nil, errors.New("unexpected source download")
	}
	http.DefaultTransport = transport
	defer func() { http.DefaultTransport = original; transport.CloseIdleConnections() }()
	t.Setenv("HF_TOKEN", "")
	for _, action := range []string{"check", "inspect", "tokens", "ids", "generate", "chat", "serve"} {
		t.Run(action, func(t *testing.T) {
			cache := filepath.Join(t.TempDir(), "cache")
			before := dials.Load()
			err := run(context.Background(), []string{"infer", "local", "-repo", "author/toy", "-cache-dir", cache, action, "-temperatur", "0"})
			if err == nil || !strings.Contains(err.Error(), "flag provided but not defined") || dials.Load() != before {
				t.Fatalf("invalid action triggered source resolution: dials=%d err=%v", dials.Load()-before, err)
			}
			if _, err := os.Stat(cache); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("invalid action created a cache: %v", err)
			}
		})
	}
	for _, args := range [][]string{
		{"generate", "-max-tokens", "no"}, {"generate", "-temperature"},
		{"generate", "extra"}, {"generate", "-temperature", "NaN"},
		{"generate", "-max-tokens", "0"}, {"generate", "-top-p", "0"},
		{"generate", "-threads", "0"}, {"generate", "-repeat-penalty", ".5"},
		{"generate", "-prompt", "hi", "-prompt-file", "-"},
		{"tokens", "-messages", "-", "-chat"},
		{"ids", "-ids", "1,no"}, {"ids", "-ids", "-1"},
		{"ids", "-ids", "1", "-max-tokens", "257"},
		{"ids", "-ids", "1", "-prefill-chunk", "0"},
		{"chat", "-kernel-workers", "257"}, {"chat", "-kv-mib", "0"},
		{"chat", "-system", string([]byte{0xff})},
		{"serve", "-max-batch", "0"}, {"serve", "-queue", "-1"},
		{"serve", "-token-budget", "1", "-max-batch", "2"},
		{"serve", "-stream-buffer", "-1"}, {"serve", "-prefill-chunk", "1025"},
		{"generate", "-hf-model", "override"}, {"check", "-context", "64"},
		{"-context", "-1", "check"}, {"-weight-mib", "0", "inspect"},
	} {
		t.Run(strings.Join(args, "/"), func(t *testing.T) {
			cache := filepath.Join(t.TempDir(), "cache")
			before := dials.Load()
			err := run(context.Background(), append([]string{"infer", "local", "-repo", "author/toy", "-cache-dir", cache}, args...))
			if err == nil || dials.Load() != before {
				t.Fatalf("invalid action triggered source resolution: dials=%d err=%v", dials.Load()-before, err)
			}
			if _, err := os.Stat(cache); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("invalid action created a cache: %v", err)
			}
		})
	}
}

func TestHFLocalPreflightHasNoExecutionEffects(t *testing.T) {
	trace := filepath.Join(t.TempDir(), "trace.out")
	threads := runtime.GOMAXPROCS(0)
	for _, args := range [][]string{
		{"check-hf", "-json"}, {"inspect-hf"},
		{"tokenize-hf", "-prompt-file", "-"}, // Must not read stdin.
		{"infer-hf", "-ids", "1,2", "-threads", "1"},
		{"generate", "-prompt", "-context", "-threads", "1"}, // A flag-like value is text.
		{"chat", "-threads", "1", "-kernel-workers", "2"},
		{"serve", "-threads", "1", "-trace", trace},
	} {
		if err := runInferenceAction(context.Background(), args[0], args[1:], true); err != nil {
			t.Fatal(args, err)
		}
		if runtime.GOMAXPROCS(0) != threads {
			t.Fatal("preflight changed worker configuration")
		}
	}
	if _, err := os.Stat(trace); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("preflight opened a trace", err)
	}
	if err := run(context.Background(), []string{"infer", "local", "-dir", hfCLIFixture(t), "tokens", "-prompt", "-context"}); err != nil {
		t.Fatal("flag-like prompt was mistaken for a source option", err)
	}
}
func TestHFCLIBackendBoundaries(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	for _, args := range [][]string{
		{"learn", "chat"},
		{"infer", "local", "-repo", "author/toy", "-dir", "other", "check"},
		{"infer", "local", "-dir", "other", "generate", "-hf-model", "override"},
		{"infer", "remote", "generate", "-model", "openai/gpt-oss-20b", "-prompt", "test"},
	} {
		if err := run(context.Background(), args); err == nil {
			t.Fatal("invalid backend selection accepted", args)
		}
	}
	for _, endpoint := range []string{"http://router.huggingface.co/v1", "https://attacker.example/v1", "https://router.huggingface.co.attacker.example/v1", "https://user:secret@router.huggingface.co/v1", "https://router.huggingface.co/v1?token=secret"} {
		if err := validateHFEndpoint(endpoint); err == nil {
			t.Fatal("unsafe endpoint", endpoint)
		}
	}
	for _, endpoint := range []string{"https://router.huggingface.co/v1", "https://toy.us-east-1.aws.endpoints.huggingface.cloud"} {
		if err := validateHFEndpoint(endpoint); err != nil {
			t.Fatal(err)
		}
	}
}
