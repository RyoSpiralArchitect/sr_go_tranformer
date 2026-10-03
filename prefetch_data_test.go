package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Run each size in a fresh test process to record process peak RSS separately.
// Both corpora have one train shard, isolating payload size from manifest size.
func BenchmarkTokenPrefetchCorpusSize(b *testing.B) {
	for _, documents := range []int{4096, 40960} {
		b.Run(fmt.Sprintf("documents_%d", documents), func(b *testing.B) {
			dir := b.TempDir()
			train, val := filepath.Join(dir, "train"), filepath.Join(dir, "val")
			f, err := os.Create(train)
			if err != nil {
				b.Fatal(err)
			}
			w := bufio.NewWriter(f)
			for i := 0; i < documents; i++ {
				if _, err = w.WriteString(strings.Repeat("a", 255) + "\n"); err != nil {
					b.Fatal(err)
				}
			}
			if err = w.Flush(); err != nil {
				b.Fatal(err)
			}
			if err = f.Close(); err != nil {
				b.Fatal(err)
			}
			if err = os.WriteFile(val, []byte("validation\n"), 0600); err != nil {
				b.Fatal(err)
			}
			out := filepath.Join(dir, "dataset")
			if err = PrepareTokenDataset(context.Background(), prepareOptions{Train: train, Validation: val, Out: out, DocumentBytes: 256, ShardTokens: 16 << 20}); err != nil {
				b.Fatal(err)
			}
			d, err := OpenTokenDataset(context.Background(), filepath.Join(out, "manifest.json"))
			if err != nil {
				b.Fatal(err)
			}
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			p, err := NewTokenPrefetch(context.Background(), d, "train", PrefetchConfig{Batch: 2, Seq: 128, Workers: 2, Depth: 4}, DatasetCursor{Offset: 1})
			if err != nil {
				b.Fatal(err)
			}
			defer p.Close()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				batch, err := p.Next(context.Background())
				if err != nil {
					b.Fatal(err)
				}
				batch.Release()
			}
			b.StopTimer()
			runtime.GC()
			runtime.ReadMemStats(&after)
			b.ReportMetric(float64(max(int64(0), int64(after.HeapAlloc)-int64(before.HeapAlloc))), "live_heap_B")
			b.ReportMetric(float64(documents*256), "input_B")
		})
	}
}

func nextTokens(t *testing.T, p *TokenPrefetch) *TokenBatch {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	b, err := p.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func TestPrefetchPackedTargetsAndCursor(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "abc\ndef\nghi\n", "xyz\n", 12)
	c := PrefetchConfig{Batch: 2, Seq: 5, Workers: 2, Depth: 3}
	p, err := NewTokenPrefetch(context.Background(), d, "train", c, DatasetCursor{Offset: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	var epoch []int
	for i, s := range d.manifest.Shards {
		if s.Split == "train" {
			ids := make([]int, s.Tokens-1)
			if err := d.ReadTokens(context.Background(), i, 1, ids); err != nil {
				t.Fatal(err)
			}
			epoch = append(epoch, ids...)
		}
	}
	var last DatasetCursor
	for batch := 0; batch < 9; batch++ {
		b := nextTokens(t, p)
		for i, v := range b.Y {
			if v != epoch[(batch*10+i)%len(epoch)] {
				t.Fatalf("target %d: got %d", batch*10+i, v)
			}
		}
		for row := 0; row < 2; row++ {
			if b.X[row*5] != BOS || !reflect.DeepEqual(b.X[row*5+1:(row+1)*5], b.Y[row*5:(row+1)*5-1]) {
				t.Fatal("BOS/shift framing changed")
			}
		}
		last = b.After
		b.Release()
	}
	if last.Epoch != uint64(90/len(epoch)) {
		t.Fatal("incorrect epoch", last)
	}
	// Restart from consumed position while the original still has speculative IO.
	q, err := NewTokenPrefetch(context.Background(), d, "train", c, last)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	a, b := nextTokens(t, p), nextTokens(t, q)
	defer a.Release()
	defer b.Release()
	if !reflect.DeepEqual(a.X, b.X) || !reflect.DeepEqual(a.Y, b.Y) || a.After != b.After {
		t.Fatal("cursor replay changed next batch")
	}
}

func TestPrefetchWorkerAndDepthInvariance(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "first\nsecond\nthird\nfourth\nfifth\n", "held\n", 12)
	type snapshot struct {
		x, y  []int
		after DatasetCursor
	}
	var want []snapshot
	for _, settings := range [][2]int{{1, 1}, {2, 3}, {4, 7}} {
		p, err := NewTokenPrefetch(context.Background(), d, "train", PrefetchConfig{Batch: 2, Seq: 7, Workers: settings[0], Depth: settings[1], Seed: 913, Shuffle: true}, DatasetCursor{Offset: 1})
		if err != nil {
			t.Fatal(err)
		}
		var got []snapshot
		for i := 0; i < 25; i++ {
			b := nextTokens(t, p)
			got = append(got, snapshot{append([]int(nil), b.X...), append([]int(nil), b.Y...), b.After})
			b.Release()
		}
		p.Close()
		if want == nil {
			want = got
		} else if !reflect.DeepEqual(got, want) {
			t.Fatal("worker/depth settings changed ordered batches", settings)
		}
	}
}

func TestPrefetchReordersAndCreditsFollowConsumer(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "abcdefghijklmnopqrstuvwxyz\n", "xyz\n", 40)
	first := make(chan struct{})
	later := make(chan struct{})
	var signal sync.Once
	var reads atomic.Int64
	read := func(ctx context.Context, shard int, offset int64, dst []int) error {
		reads.Add(1)
		if shard == 0 && offset == 1 {
			select {
			case <-first:
			case <-ctx.Done():
				return ctx.Err()
			}
		} else {
			signal.Do(func() { close(later) })
		}
		return d.ReadTokens(ctx, shard, offset, dst)
	}
	p, err := newTokenPrefetch(context.Background(), d, "train", PrefetchConfig{Batch: 1, Seq: 3, Workers: 2, Depth: 2}, DatasetCursor{Offset: 1}, read)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	select {
	case <-later:
	case <-time.After(3 * time.Second):
		t.Fatal("later job did not overtake first")
	}
	select {
	case <-p.out:
		t.Fatal("out-of-order batch escaped")
	default:
	}
	close(first)
	a, b := nextTokens(t, p), nextTokens(t, p)
	if !reflect.DeepEqual(a.Y, []int{'a', 'b', 'c'}) || !reflect.DeepEqual(b.Y, []int{'d', 'e', 'f'}) {
		t.Fatal("reorder failed")
	}
	// Every credit is now held by this consumer: no third read is permitted.
	if reads.Load() != 2 {
		t.Fatal("prefetch exceeded depth", reads.Load())
	}
	select {
	case <-p.out:
		t.Fatal("pipeline emitted beyond leased capacity")
	case <-time.After(25 * time.Millisecond):
	}
	a.Release()
	a.Release() // Stale/double releases must not add a second credit.
	c := nextTokens(t, p)
	if reads.Load() != 3 {
		t.Fatal("release did not return exactly one credit", reads.Load())
	}
	if !reflect.DeepEqual(b.Y, []int{'d', 'e', 'f'}) {
		t.Fatal("leased memory reused before release")
	}
	b.Release()
	c.Release()
}

func TestPrefetchCancellationAndIOFailure(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "abcdefghijklmnop\n", "xyz\n", 32)
	for _, blocked := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		entered := make(chan struct{})
		var once sync.Once
		boom := errors.New("read failed")
		read := func(ctx context.Context, _ int, _ int64, _ []int) error {
			once.Do(func() { close(entered) })
			if blocked {
				<-ctx.Done()
				return ctx.Err()
			}
			return boom
		}
		p, err := newTokenPrefetch(ctx, d, "train", PrefetchConfig{Batch: 1, Seq: 3, Workers: 2, Depth: 3}, DatasetCursor{Offset: 1}, read)
		if err != nil {
			t.Fatal(err)
		}
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			t.Fatal("reader did not start")
		}
		if blocked {
			cancel()
		}
		_, err = p.Next(context.Background())
		if blocked && !errors.Is(err, context.Canceled) || !blocked && !errors.Is(err, boom) {
			t.Fatal("lost read error", err)
		}
		done := make(chan struct{})
		go func() { p.Close(); close(done) }()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatal("pipeline goroutines did not exit")
		}
		cancel()
	}
	// Closing a pipeline whose downstream consumer never reads must also join.
	p, err := NewTokenPrefetch(context.Background(), d, "train", PrefetchConfig{Batch: 1, Seq: 3, Workers: 2, Depth: 2}, DatasetCursor{Offset: 1})
	if err != nil {
		t.Fatal(err)
	}
	p.Close()
	if _, err = p.Next(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestPrefetchRejectsUnboundedSettingsAndBadCursor(t *testing.T) {
	_, d := preparedDataset(t, Tokenizer{}, "abc\n", "xyz\n", 12)
	for _, c := range []PrefetchConfig{{Batch: 1, Seq: 1, Workers: 0, Depth: 1}, {Batch: 1, Seq: 1, Workers: 2, Depth: 1}, {Batch: 1, Seq: 1, Workers: 1, Depth: 65}, {Batch: 4096, Seq: 32768, Workers: 1, Depth: 64}} {
		if p, err := NewTokenPrefetch(context.Background(), d, "train", c, DatasetCursor{Offset: 1}); err == nil {
			p.Close()
			t.Fatal("unbounded settings accepted")
		}
	}
	c := PrefetchConfig{Batch: 1, Seq: 1, Workers: 1, Depth: 1}
	for _, cursor := range []DatasetCursor{{}, {Shard: -1, Offset: 1}, {Shard: 1, Offset: 1}, {Offset: 6}} {
		if p, err := NewTokenPrefetch(context.Background(), d, "train", c, cursor); err == nil {
			p.Close()
			t.Fatal("invalid cursor accepted", cursor)
		}
	}
}
