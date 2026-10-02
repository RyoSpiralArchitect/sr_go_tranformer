// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func prefillTokens(n, salt int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*17 + salt) % 256
	}
	return ids
}

func TestPrefillMixedChunksMatchSequentialAndForward(t *testing.T) {
	m := testModel(t)
	prefixes := []int{0, 31, 63, 2}
	lengths := []int{37, 4, 5, 9}
	caches := make([]*KVCache, len(prefixes))
	refs := make([]*KVCache, len(prefixes))
	chunks := make([][]int, len(prefixes))
	want := make([][]float32, len(prefixes))
	for i, prefix := range prefixes {
		caches[i], refs[i] = m.NewCache(), m.NewCache()
		defer caches[i].Close()
		defer refs[i].Close()
		ids := prefillTokens(prefix+lengths[i], 3+i)
		for j, id := range ids {
			var err error
			want[i], err = m.Step(refs[i], id)
			if err != nil {
				t.Fatal(err)
			}
			if j < prefix {
				if _, err := m.Step(caches[i], id); err != nil {
					t.Fatal(err)
				}
			}
		}
		full, err := m.Forward(ids, 1, len(ids))
		if err != nil {
			t.Fatal(err)
		}
		closeSlices(t, want[i], full.Logits[(len(ids)-1)*m.Config.Vocab:], 2e-6)
		chunks[i] = ids[prefix:]
	}
	got, err := m.PrefillBatch(context.Background(), caches, chunks)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(caches)*m.Config.Vocab {
		t.Fatalf("prefill returned %d logits", len(got))
	}
	for i, cache := range caches {
		closeSlices(t, got[i*m.Config.Vocab:(i+1)*m.Config.Vocab], want[i], 0)
		if cache.Pos != prefixes[i]+lengths[i] {
			t.Fatalf("session %d position %d", i, cache.Pos)
		}
		for layer := range m.Blocks {
			for pos := 0; pos < cache.Pos; pos++ {
				k, v := cache.row(layer, pos)
				rk, rv := refs[i].row(layer, pos)
				closeSlices(t, k, rk, 0)
				closeSlices(t, v, rv, 0)
			}
		}
		actual, err := m.Step(cache, 117)
		if err != nil {
			t.Fatal(err)
		}
		expected, err := m.Step(refs[i], 117)
		if err != nil {
			t.Fatal(err)
		}
		closeSlices(t, actual, expected, 0)
	}
	// A caller-owned result cannot alias a workspace returned to sync.Pool.
	saved := append([]float32(nil), got...)
	if _, err := m.PrefillBatch(context.Background(), caches[:1], [][]int{{1, 2}}); err != nil {
		t.Fatal(err)
	}
	closeSlices(t, got, saved, 0)
}

func TestPrefillInvalidInputDoesNotMutateCaches(t *testing.T) {
	cases := []struct {
		name  string
		setup func(*Model, *KVCache) (context.Context, []*KVCache, [][]int)
	}{
		{"empty", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), nil, nil
		}},
		{"shape", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c}, nil
		}},
		{"empty_chunk", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c}, [][]int{{}}
		}},
		{"nil_context", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return nil, []*KVCache{c}, [][]int{{1}}
		}},
		{"nil_cache", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c, nil}, [][]int{{1}, {2}}
		}},
		{"duplicate", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c, c}, [][]int{{1}, {2}}
		}},
		{"foreign", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c, testModel(t).NewCache()}, [][]int{{1}, {2}}
		}},
		{"released", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			other := m.NewCache()
			other.Close()
			return context.Background(), []*KVCache{c, other}, [][]int{{1}, {2}}
		}},
		{"stale", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			other := m.NewCache()
			other.Revision++
			return context.Background(), []*KVCache{c, other}, [][]int{{1}, {2}}
		}},
		{"negative_token", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c}, [][]int{{1, -1}}
		}},
		{"large_token", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c}, [][]int{{1, m.Config.Vocab}}
		}},
		{"context_overflow", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c}, [][]int{make([]int, m.Config.Context)}
		}},
		{"negative_position", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			other := m.NewCache()
			other.Pos = -1
			return context.Background(), []*KVCache{c, other}, [][]int{{1}, {2}}
		}},
		{"missing_prefix_pages", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			other := m.NewCache()
			other.Pos = 32
			return context.Background(), []*KVCache{c, other}, [][]int{{1}, {2}}
		}},
		{"oversize", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), []*KVCache{c}, [][]int{make([]int, 1025)}
		}},
		{"too_many_sessions", func(m *Model, c *KVCache) (context.Context, []*KVCache, [][]int) {
			return context.Background(), make([]*KVCache, 257), make([][]int, 257)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := testModel(t)
			cache := m.NewCache()
			defer cache.Close()
			if _, err := m.Step(cache, BOS); err != nil {
				t.Fatal(err)
			}
			before := append([]float32(nil), cache.Pages[0].Data...)
			ctx, caches, chunks := tc.setup(m, cache)
			if _, err := m.PrefillBatch(ctx, caches, chunks); err == nil {
				t.Fatal("accepted invalid prefill input")
			}
			if cache.Pos != 1 || len(cache.Pages) != 1 || !reflect.DeepEqual(before, cache.Pages[0].Data) {
				t.Fatal("invalid input changed an earlier valid session")
			}
		})
	}
}

// This context cancels at a deterministic kernel checkpoint, including after
// some future rows have been populated, without timing-dependent goroutines.
type prefillCancelContext struct {
	context.Context
	calls, cancelAt int
}

func (c *prefillCancelContext) Err() error {
	c.calls++
	if c.calls >= c.cancelAt {
		return context.Canceled
	}
	return nil
}

func TestPrefillCancellationPreservesPrefixAndRetry(t *testing.T) {
	// Start, post-validation, first layer, second layer, and pre-commit.
	for _, cancelAt := range []int{1, 2, 3, 4, 5} {
		m := testModel(t)
		cache, ref := m.NewCache(), m.NewCache()
		for _, id := range prefillTokens(31, 2) {
			if _, err := m.Step(cache, id); err != nil {
				t.Fatal(err)
			}
			if _, err := m.Step(ref, id); err != nil {
				t.Fatal(err)
			}
		}
		ctx := &prefillCancelContext{Context: context.Background(), cancelAt: cancelAt}
		if _, err := m.PrefillBatch(ctx, []*KVCache{cache}, [][]int{{3, 4, 5}}); !errors.Is(err, context.Canceled) {
			t.Fatalf("checkpoint %d: %v", cancelAt, err)
		}
		if cache.Pos != 31 {
			t.Fatalf("checkpoint %d committed canceled prefill", cancelAt)
		}
		for layer := range m.Blocks {
			for pos := 0; pos < cache.Pos; pos++ {
				k, v := cache.row(layer, pos)
				rk, rv := ref.row(layer, pos)
				closeSlices(t, k, rk, 0)
				closeSlices(t, v, rv, 0)
			}
		}
		// Retry with different tokens and a shorter chunk so canceled future KV
		// cannot accidentally leak into the new causal prefix.
		got, err := m.PrefillBatch(context.Background(), []*KVCache{cache}, [][]int{{8, 9}})
		if err != nil {
			t.Fatal(err)
		}
		var want []float32
		for _, id := range []int{8, 9} {
			want, err = m.Step(ref, id)
			if err != nil {
				t.Fatal(err)
			}
		}
		closeSlices(t, got, want, 0)
		cache.Close()
		ref.Close()
	}
}

func TestPrefillReservationValidationAndRelease(t *testing.T) {
	m := testModel(t)
	pool := &pagePool{PageElements: 2 * pageTokens * m.Config.KVDim() * m.Config.Layers, Limit: 4, Reserved: 3}
	a, b := m.NewCache(), m.NewCache()
	a.pool, a.reserved = pool, 2
	b.pool, b.reserved = pool, 1
	chunks := [][]int{prefillTokens(35, 1), prefillTokens(33, 2)}
	if _, err := m.PrefillBatch(context.Background(), []*KVCache{a, b}, chunks); err == nil {
		t.Fatal("accepted under-reserved session")
	}
	if pool.Allocated != 0 || a.Pos != 0 || b.Pos != 0 || len(a.Pages) != 0 {
		t.Fatal("invalid reservation mutated cache or pool")
	}
	chunks[1] = chunks[1][:31]
	if _, err := m.PrefillBatch(context.Background(), []*KVCache{a, b}, chunks); err != nil {
		t.Fatal(err)
	}
	if pool.Allocated != 3 || pool.Reserved != 3 {
		t.Fatalf("incorrect pool accounting: %+v", pool)
	}
	a.Close()
	b.Close()
	if pool.Reserved != 0 || len(pool.Free) != 3 {
		t.Fatalf("KV pages not returned: %+v", pool)
	}
	// Reused pages contain old rows; causality must keep them out of the result.
	c, ref := m.NewCache(), m.NewCache()
	defer c.Close()
	defer ref.Close()
	c.pool, c.reserved, pool.Reserved = pool, 1, 1
	got, err := m.PrefillBatch(context.Background(), []*KVCache{c}, [][]int{{9, 8, 7}})
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.PrefillBatch(context.Background(), []*KVCache{ref}, [][]int{{9, 8, 7}})
	if err != nil {
		t.Fatal(err)
	}
	closeSlices(t, got, want, 0)
}

func TestPrefillWorkspaceBoundBeforeAllocation(t *testing.T) {
	// Valid model geometry but no weights are needed: workspace rejection must
	// happen before allocation, model reads, or any change to the cache.
	c := Config{Vocab: baseVocab, Dim: 512, Hidden: 32768, Layers: 1, Heads: 256, KVHeads: 1, Context: 32768, RopeTheta: 10000, NormEps: 1e-5}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	m := &Model{Config: c}
	cache := m.NewCache()
	defer cache.Close()
	cache.Pos = c.Context - 1024
	cache.ensurePages(cache.Pos)
	pages := len(cache.Pages)
	if _, err := m.PrefillBatch(context.Background(), []*KVCache{cache}, [][]int{make([]int, 1024)}); err == nil {
		t.Fatal("accepted workspace beyond allocation budget")
	}
	if cache.Pos != c.Context-1024 || len(cache.Pages) != pages {
		t.Fatal("workspace rejection changed cache")
	}
}

func TestPrefillTotalTokenLimit(t *testing.T) {
	m := testModel(t)
	caches := make([]*KVCache, 9)
	chunks := make([][]int, len(caches))
	for i := range caches {
		caches[i] = m.NewCache()
		defer caches[i].Close()
		chunks[i] = prefillTokens(128, i)
	}
	if _, err := m.PrefillBatch(context.Background(), caches, chunks); err == nil {
		t.Fatal("accepted more than 1024 tokens across multiple sessions")
	}
	for _, cache := range caches {
		if cache.Pos != 0 || len(cache.Pages) != 0 {
			t.Fatal("oversize aggregate changed a cache")
		}
	}
	got, err := m.PrefillBatch(context.Background(), caches[:8], chunks[:8])
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 8*m.Config.Vocab {
		t.Fatalf("wrong output shape at token limit: %d", len(got))
	}
	for _, cache := range caches[:8] {
		if cache.Pos != m.Config.Context {
			t.Fatal("exact context boundary was not committed")
		}
	}
}

func BenchmarkPrefillSequentialVsChunk(b *testing.B) {
	for _, presetName := range []string{"demo", "tiny"} {
		b.Run(presetName, func(b *testing.B) {
			c, _ := preset(presetName)
			m, err := NewModel(c, 19)
			if err != nil {
				b.Fatal(err)
			}
			ids := prefillTokens(128, 3)
			for _, chunkSize := range []int{1, 16} {
				name := "sequential"
				if chunkSize > 1 {
					name = "chunk16"
				}
				b.Run(name, func(b *testing.B) {
					cache := m.NewCache()
					defer cache.Close()
					// Match retained page allocation in both cases; include workspace
					// and returned-logit costs, which are part of chunking's effect.
					cache.ensurePages(len(ids))
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						cache.Pos = 0
						for offset := 0; offset < len(ids); offset += chunkSize {
							chunk := ids[offset:min(offset+chunkSize, len(ids))]
							if _, err := m.PrefillBatch(context.Background(), []*KVCache{cache}, [][]int{chunk}); err != nil {
								b.Fatal(err)
							}
						}
					}
					b.ReportMetric(float64(b.N*len(ids))/b.Elapsed().Seconds(), "tokens/s")
				})
			}
		})
	}
}
