// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func resumeFile(data []byte, lfs bool) hfHubFile {
	f := hfHubFile{Name: "model.safetensors", Size: int64(len(data))}
	if lfs {
		b := sha256.Sum256(data)
		f.LFS = &struct {
			SHA  string `json:"sha256"`
			Size int64  `json:"size"`
		}{hex.EncodeToString(b[:]), f.Size}
	} else {
		h := sha1.New()
		fmt.Fprintf(h, "blob %d%c", len(data), 0)
		h.Write(data)
		f.Blob = hex.EncodeToString(h.Sum(nil))
	}
	return f
}

func TestHFDownloadResumeAndIgnoredRange(t *testing.T) {
	for _, lfs := range []bool{false, true} {
		for _, ignoreRange := range []bool{false, true} {
			t.Run(fmt.Sprintf("lfs=%v/ignore=%v", lfs, ignoreRange), func(t *testing.T) {
				t.Setenv("HF_HUB_OFFLINE", "0")
				data := bytes.Repeat([]byte("abcdef0123456789"), 10000)
				f := resumeFile(data, lfs)
				dir := t.TempDir()
				calls := 0
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls++
					if r.Header.Get("Accept-Encoding") != "identity" {
						t.Error("encoded transfer requested")
					}
					if calls == 1 {
						if r.Header.Get("Range") != "" {
							t.Error("unexpected initial range")
						}
						w.Header().Set("Content-Length", fmt.Sprint(len(data)))
						w.Write(data[:70001]) // A real short HTTP response, not a synthetic error.
						return
					}
					if r.Header.Get("Range") != "bytes=70001-" {
						t.Errorf("range=%q", r.Header.Get("Range"))
					}
					if ignoreRange {
						w.Header().Set("Content-Length", fmt.Sprint(len(data)))
						w.Write(data)
					} else {
						w.Header().Set("Content-Length", fmt.Sprint(len(data)-70001))
						w.Header().Set("Content-Range", fmt.Sprintf("bytes 70001-%d/%d", len(data)-1, len(data)))
						w.WriteHeader(http.StatusPartialContent)
						w.Write(data[70001:])
					}
				}))
				defer server.Close()
				h := hfHub{Base: server.URL, Client: server.Client()}
				if _, err := h.download(context.Background(), dir, "author/toy", strings.Repeat("a", 40), f); err == nil {
					t.Fatal("short download accepted")
				}
				if calls != 1 {
					t.Fatal("automatic retry", calls)
				}
				got, err := h.download(context.Background(), dir, "author/toy", strings.Repeat("a", 40), f)
				if err != nil {
					t.Fatal(err)
				}
				expected := sha256.Sum256(data)
				if got.SHA != hex.EncodeToString(expected[:]) {
					t.Fatal("incorrect resumed checksum")
				}
				actual, _ := os.ReadFile(filepath.Join(dir, f.Name))
				if !bytes.Equal(actual, data) {
					t.Fatal("prefix duplicated or lost")
				}
				// A completed shard is verified again without a request.
				if _, err = h.download(context.Background(), dir, "author/toy", strings.Repeat("a", 40), f); err != nil {
					t.Fatal(err)
				}
				if calls != 2 {
					t.Fatal("complete file downloaded again")
				}
			})
		}
	}
}

func TestHFDownloadRejectsUnsafeResume(t *testing.T) {
	for _, kind := range []string{"wrong-start", "wrong-total", "missing-range", "duplicate-range", "multipart", "encoded", "wrong-length", "oversize", "corrupt", "corrupt-prefix", "rate-limit", "range-unsatisfied"} {
		t.Run(kind, func(t *testing.T) {
			t.Setenv("HF_HUB_OFFLINE", "0")
			data := []byte("0123456789abcdefghij")
			f := resumeFile(data, true)
			dir := t.TempDir()
			prefix := append([]byte(nil), data[:5]...)
			if kind == "corrupt-prefix" {
				prefix[0] ^= 1
			}
			path := filepath.Join(dir, f.Name)
			if err := os.WriteFile(path, prefix, 0600); err != nil {
				t.Fatal(err)
			}
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				cr, body := "bytes 5-19/20", append([]byte(nil), data[5:]...)
				switch kind {
				case "wrong-start":
					cr = "bytes 4-18/20"
				case "wrong-total":
					cr = "bytes 5-19/21"
				case "missing-range", "multipart":
					cr = ""
				case "duplicate-range":
					w.Header().Add("Content-Range", cr)
				case "encoded":
					w.Header().Set("Content-Encoding", "gzip")
				case "wrong-length":
					w.Header().Set("Content-Length", "14")
				case "oversize":
					body = append(body, '!')
				case "corrupt":
					body[0] ^= 1
				case "rate-limit":
					w.WriteHeader(429)
					return
				case "range-unsatisfied":
					w.WriteHeader(416)
					return
				}
				if cr != "" {
					w.Header().Add("Content-Range", cr)
				}
				if kind == "multipart" {
					w.Header().Set("Content-Type", "multipart/byteranges; boundary=x")
				}
				w.WriteHeader(http.StatusPartialContent)
				w.(http.Flusher).Flush() // Unknown Content-Length exercises the byte bound.
				w.Write(body)
			}))
			defer server.Close()
			h := hfHub{Base: server.URL, Client: server.Client()}
			if _, err := h.download(context.Background(), dir, "author/toy", strings.Repeat("a", 40), f); err == nil {
				t.Fatal("unsafe resume accepted")
			}
			if calls != 1 {
				t.Fatal("automatic retry", calls)
			}
			actual, err := os.ReadFile(path)
			if kind == "corrupt" || kind == "corrupt-prefix" || kind == "oversize" {
				if !os.IsNotExist(err) {
					t.Fatal("corrupt file retained", err)
				}
			} else if err != nil || !bytes.Equal(actual, prefix) {
				t.Fatal("invalid headers modified prefix", err)
			}
		})
	}
}

type resumeTransport func(*http.Request) (*http.Response, error)

func (f resumeTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestHFHubResumePublicationAndOffline(t *testing.T) {
	h, s, counts := fakeHFHub(t, false)
	base := h.Client.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	first := true
	h.Client.Transport = resumeTransport(func(req *http.Request) (*http.Response, error) {
		resp, err := base.RoundTrip(req)
		if err != nil || !strings.HasSuffix(req.URL.Path, "model.safetensors") {
			return resp, err
		}
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		if first {
			first = false
			resp.Body = io.NopCloser(bytes.NewReader(data[:123]))
		} else {
			if req.Header.Get("Range") != "bytes=123-" {
				t.Errorf("range=%q", req.Header.Get("Range"))
			}
			resp.StatusCode = http.StatusPartialContent
			resp.ContentLength = int64(len(data) - 123)
			resp.Header.Set("Content-Range", fmt.Sprintf("bytes 123-%d/%d", len(data)-1, len(data)))
			resp.Body = io.NopCloser(bytes.NewReader(data[123:]))
		}
		return resp, nil
	})
	if _, _, err := h.resolve(context.Background(), s); err == nil {
		t.Fatal("incomplete snapshot published")
	}
	sha := strings.Repeat("a", 40)
	root := filepath.Join(s.Cache, hfKey(s.Repo))
	if _, err := os.Stat(filepath.Join(root, sha)); !os.IsNotExist(err) {
		t.Fatal("published before complete")
	}
	before := fmt.Sprint(counts())
	offline := s
	offline.Revision, offline.Offline = sha, true
	if _, _, err := h.resolve(context.Background(), offline); err == nil {
		t.Fatal("offline accepted partial")
	}
	if fmt.Sprint(counts()) != before {
		t.Fatal("offline partial used network")
	}
	path, _, err := h.resolve(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	for p, n := range counts() {
		if strings.HasSuffix(p, ".json") && n != 1 {
			t.Fatal("complete metadata fetched again", p, n)
		}
	}
	if _, err = os.Stat(filepath.Join(root, ".partial-"+sha)); !os.IsNotExist(err) {
		t.Fatal("partial not consumed", err)
	}
	if _, err = validateHFSnapshot(context.Background(), path, s, sha); err != nil {
		t.Fatal(err)
	}
	s.Offline = true
	before = fmt.Sprint(counts())
	if _, _, err = h.resolve(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(counts()) != before {
		t.Fatal("offline snapshot used network")
	}
}

func TestHFDownloadCancellationKeepsPrefix(t *testing.T) {
	t.Setenv("HF_HUB_OFFLINE", "0")
	data := bytes.Repeat([]byte("abcdef"), 30000)
	f := resumeFile(data, true)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.Write(data[:12345])
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()
	dir := t.TempDir()
	// Cancel only after the prefix is actually written; no scheduler timing guess.
	done := make(chan error, 1)
	h := hfHub{Base: server.URL, Client: server.Client()}
	go func() { _, err := h.download(ctx, dir, "author/toy", strings.Repeat("a", 40), f); done <- err }()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
wait:
	for {
		select {
		case <-deadline.C:
			cancel()
			t.Fatal("prefix was not written")
		case <-ticker.C:
			if st, err := os.Stat(filepath.Join(dir, f.Name)); err == nil && st.Size() == 12345 {
				break wait
			}
		}
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, f.Name))
	if err != nil || !bytes.Equal(b, data[:12345]) {
		t.Fatal("cancel lost prefix", err)
	}
}

func TestHFDownloadStageExclusiveOwnership(t *testing.T) {
	root, sha := t.TempDir(), strings.Repeat("b", 40)
	parked := filepath.Join(root, ".partial-"+sha)
	if err := os.Mkdir(parked, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(parked, "prefix"), []byte("saved"), 0600); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	stages := make(chan string, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			stage, err := hfDownloadStage(root, sha)
			if err != nil {
				t.Error(err)
				return
			}
			stages <- stage
		}()
	}
	wg.Wait()
	close(stages)
	seen, owners := map[string]bool{}, 0
	for stage := range stages {
		if seen[stage] {
			t.Fatal("shared active stage")
		}
		seen[stage] = true
		if _, err := os.Stat(filepath.Join(stage, "prefix")); err == nil {
			owners++
		}
		if err := os.WriteFile(filepath.Join(stage, "private"), []byte("x"), 0600); err != nil {
			t.Fatal(err)
		}
		parkHFDownload(stage, root, sha)
	}
	if len(seen) != 8 || owners != 1 {
		t.Fatal("claim not exclusive", len(seen), owners)
	}
	entries, _ := os.ReadDir(root)
	if len(entries) != 1 || entries[0].Name() != ".partial-"+sha {
		t.Fatal("private stages leaked", entries)
	}
}

func TestHFDownloadRejectsPartialSymlinks(t *testing.T) {
	root, sha := t.TempDir(), strings.Repeat("b", 40)
	if err := os.Symlink(t.TempDir(), filepath.Join(root, ".partial-"+sha)); err != nil {
		t.Fatal(err)
	}
	if _, err := hfDownloadStage(root, sha); err == nil {
		t.Fatal("symlink stage accepted")
	}
	dir, external := t.TempDir(), filepath.Join(t.TempDir(), "external")
	if err := os.WriteFile(external, []byte("saved"), 0600); err != nil {
		t.Fatal(err)
	}
	f := resumeFile([]byte("0123456789"), true)
	if err := os.Symlink(external, filepath.Join(dir, f.Name)); err != nil {
		t.Fatal(err)
	}
	if _, err := (hfHub{}).download(context.Background(), dir, "author/toy", sha, f); err == nil {
		t.Fatal("symlink file accepted")
	}
	b, _ := os.ReadFile(external)
	if string(b) != "saved" {
		t.Fatal("external target modified")
	}
}

func TestHFDownloadStageConcurrentHandoffs(t *testing.T) {
	root, sha := t.TempDir(), strings.Repeat("c", 40)
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				stage, err := hfDownloadStage(root, sha)
				if err != nil {
					t.Error(err)
					return
				}
				if err = os.WriteFile(filepath.Join(stage, "progress"), []byte("saved"), 0600); err != nil {
					t.Error(err)
					return
				}
				if err = parkHFDownload(stage, root, sha); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 1 || entries[0].Name() != ".partial-"+sha {
		t.Fatalf("returned stages stranded: %v %v", entries, err)
	}
}
