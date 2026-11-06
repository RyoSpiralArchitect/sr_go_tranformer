package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func preparedDataset(t *testing.T, tok Tokenizer, train, val string, shardTokens int) (string, *TokenDataset) {
	t.Helper()
	dir := t.TempDir()
	for name, text := range map[string]string{"train.txt": train, "val.txt": val} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	out := filepath.Join(dir, "dataset")
	err := PrepareTokenDataset(context.Background(), prepareOptions{Train: filepath.Join(dir, "train.txt"), Validation: filepath.Join(dir, "val.txt"), Out: out, Tokenizer: tok, ShardTokens: shardTokens, DocumentBytes: maxDocumentBytes})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(out, "manifest.json")
	d, err := OpenTokenDataset(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	return path, d
}

func TestTokenShardsMatchDocumentEncoding(t *testing.T) {
	tok := Tokenizer{Merges: []Pair{{'a', 'b'}, {258, 258}, {259, 259}}}
	train := "abababab\r\n日本語🐈\n\n" + strings.Repeat("ab", 9000) + "\nlast\xff"
	val := "held out\nsecond\n"
	for _, tokenizer := range []Tokenizer{{}, tok} {
		_, d := preparedDataset(t, tokenizer, train, val, 20000)
		for _, input := range []struct{ split, text string }{{"train", train}, {"validation", val}} {
			var want, got []int
			// Independent document enumeration; keep newline bytes.
			for _, doc := range strings.SplitAfter(input.text, "\n") {
				if doc != "" {
					want = append(want, tokenizer.Encode(doc, true, true)...)
				}
			}
			for i, s := range d.manifest.Shards {
				if s.Split != input.split {
					continue
				}
				for offset := int64(0); offset < s.Tokens; {
					buf := make([]int, min(int64(7), s.Tokens-offset))
					if err := d.ReadTokens(context.Background(), i, offset, buf); err != nil {
						t.Fatal(err)
					}
					got = append(got, buf...)
					offset += int64(len(buf))
				}
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s tokenization changed across shard/read boundaries", input.split)
			}
		}
	}
}

func TestTokenDatasetRejectsCorruption(t *testing.T) {
	for _, kind := range []string{"checksum", "truncated", "extra", "header", "vocab", "bos", "eos", "documents", "tokenizer", "duplicate", "path", "split", "oversize", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			path, d := preparedDataset(t, Tokenizer{}, "abc\ndef\n", "validation\n", 32)
			m := d.manifest
			s := &m.Shards[0]
			file := filepath.Join(d.root, s.File)
			b, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			rehash := false
			switch kind {
			case "checksum":
				b[shardHeaderBytes+4] ^= 1
			case "truncated":
				b = b[:len(b)-1]
			case "extra":
				b = append(b, 0)
			case "header":
				b[0] ^= 1
			case "vocab":
				binary.LittleEndian.PutUint32(b[shardHeaderBytes+4:], uint32(m.Vocab))
				rehash = true
			case "bos":
				binary.LittleEndian.PutUint32(b[shardHeaderBytes:], 97)
				rehash = true
			case "eos":
				binary.LittleEndian.PutUint32(b[len(b)-4:], 97)
				rehash = true
			case "documents":
				s.Documents++
				binary.LittleEndian.PutUint64(b[24:], uint64(s.Documents))
				rehash = true
			case "tokenizer":
				m.Tokenizer.Merges = []Pair{{'a', 'b'}}
			case "duplicate":
				m.Shards = append(m.Shards, *s)
			case "path":
				s.File = "../train.txt"
			case "split":
				s.Split = "testing"
			case "oversize":
				s.Tokens = 1 << 60
			case "symlink":
				if err = os.Rename(file, file+".real"); err != nil {
					t.Fatal(err)
				}
				if err = os.Symlink(file+".real", file); err != nil {
					t.Fatal(err)
				}
			}
			if kind != "symlink" {
				if err = os.WriteFile(file, b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if rehash {
				s.SHA256 = corpusHash(string(b))
			}
			if err = writeJSONFile(path, m); err != nil {
				t.Fatal(err)
			}
			if _, err = OpenTokenDataset(context.Background(), path); err == nil {
				t.Fatal("invalid dataset accepted")
			}
		})
	}
}

func TestDatasetReadBoundsAndCancellation(t *testing.T) {
	path, d := preparedDataset(t, Tokenizer{}, "abc\n", "xyz\n", 10)
	for _, request := range []struct {
		shard  int
		offset int64
		n      int
	}{{-1, 0, 1}, {2, 0, 1}, {0, -1, 1}, {0, 5, 2}, {0, 0, maxReadTokens + 1}} {
		if err := d.ReadTokens(context.Background(), request.shard, request.offset, make([]int, request.n)); err == nil {
			t.Fatal("invalid read accepted", request)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := d.ReadTokens(ctx, 0, 0, make([]int, 1)); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := OpenTokenDataset(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	// Reopening a modified shard checks its header/size again.
	f, err := os.OpenFile(filepath.Join(d.root, d.manifest.Shards[0].File), os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	if err = d.ReadTokens(context.Background(), 0, 0, make([]int, 1)); err == nil {
		t.Fatal("changed shard accepted")
	}
}

func TestPrepareDoesNotPublishPartialDataset(t *testing.T) {
	for _, kind := range []string{"long_document", "small_shard", "empty_validation", "canceled", "exists"} {
		t.Run(kind, func(t *testing.T) {
			dir := t.TempDir()
			train, val := filepath.Join(dir, "train"), filepath.Join(dir, "val")
			if err := os.WriteFile(train, []byte("abc\n"), 0600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(val, []byte("xyz\n"), 0600); err != nil {
				t.Fatal(err)
			}
			o := prepareOptions{Train: train, Validation: val, Out: filepath.Join(dir, "out"), ShardTokens: 32, DocumentBytes: 32}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch kind {
			case "long_document":
				o.DocumentBytes = 3
			case "small_shard":
				o.ShardTokens = 3
			case "empty_validation":
				if err := os.WriteFile(val, nil, 0600); err != nil {
					t.Fatal(err)
				}
			case "canceled":
				cancel()
			case "exists":
				if err := os.WriteFile(o.Out, []byte("keep"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err := PrepareTokenDataset(ctx, o); err == nil {
				t.Fatal("invalid preparation accepted")
			}
			if kind == "exists" {
				b, _ := os.ReadFile(o.Out)
				if string(b) != "keep" {
					t.Fatal("existing output changed")
				}
			} else if _, err := os.Stat(o.Out); !os.IsNotExist(err) {
				t.Fatal("partial output published", err)
			}
			staging, err := filepath.Glob(filepath.Join(dir, ".monolith-dataset-*"))
			if err != nil || len(staging) != 0 {
				t.Fatal("staging directory leaked", staging, err)
			}
		})
	}
}

func TestBoundedDocumentReader(t *testing.T) {
	text := bytes.Repeat([]byte("a"), maxDocumentBytes+1)
	r := bufio.NewReaderSize(bytes.NewReader(text), 16)
	if _, err := readDocument(r, 64); err == nil {
		t.Fatal("oversized document accepted")
	}
	if remaining := r.Buffered(); remaining > 16 {
		t.Fatal("unbounded read buffer", remaining)
	}
	r = bufio.NewReaderSize(strings.NewReader("abc\nlast"), 16)
	for _, want := range []string{"abc\n", "last"} {
		got, err := readDocument(r, 4)
		if err != nil || string(got) != want {
			t.Fatal(string(got), err)
		}
	}
	if _, err := readDocument(r, 4); err != io.EOF {
		t.Fatal(err)
	}
}
