// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestPrefixTailDivergenceAndPrivatePage(t *testing.T) {
	m := promptPrefillTestModel(t, "q4")
	ctx := context.Background()
	ids := prefillTokens(79, 5)
	source := m.NewCache()
	defer source.Close()
	if _, err := m.PrefillBatch(ctx, []*KVCache{source}, [][]int{ids[:75]}); err != nil {
		t.Fatal(err)
	}
	p := prefixSnapshot{includeTail: true}
	p.capture(m, source, ids, 3)
	if !slices.Equal(p.ids, ids[:75]) || len(p.pages) != 3 {
		t.Fatal("snapshot must stop at the committed KV position")
	}
	kd := m.Config.KVDim()
	for _, n := range []int{0, 1, 17, 31, 32, 33, 63, 64, 74, 75} {
		query := slices.Clone(ids[:n+2])
		query[n] = (query[n] + 1) % m.Config.Vocab
		a, ref := m.NewCache(), m.NewCache()
		// Recycle a marked page. An over-copy into unmatched rows will be visible.
		private := &kvPage{Data: make([]float32, len(p.pages[0].Data))}
		for i := range private.Data {
			private.Data[i] = -12345
		}
		pool := &pagePool{PageElements: len(private.Data), Limit: 4, Reserved: 4, Allocated: 1, Free: []*kvPage{private}}
		a.pool, a.reserved = pool, 4
		p.restore(a, query)
		if a.Pos != n || a.sharedPages != n/pageTokens || (a.shared != nil) != (n >= pageTokens) {
			t.Fatalf("bad match or ownership at %d", n)
		}
		for i := 0; i < a.sharedPages; i++ {
			if a.Pages[i] != p.pages[i] {
				t.Fatalf("complete page %d was copied at %d", i, n)
			}
		}
		if tail := n % pageTokens; tail > 0 {
			if a.Pages[n/pageTokens] != private {
				t.Fatalf("tail at %d did not come from the reservation", n)
			}
			for plane := 0; plane < len(private.Data); plane += pageTokens * kd {
				for _, v := range private.Data[plane+tail*kd : plane+pageTokens*kd] {
					if v != -12345 {
						t.Fatalf("copied an unmatched row at %d", n)
					}
				}
			}
		}
		want, err := m.PrefillBatch(ctx, []*KVCache{ref}, [][]int{query})
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.PrefillBatch(ctx, []*KVCache{a}, [][]int{query[n:]})
		if err != nil {
			t.Fatal(err)
		}
		exactWorkerFloats(t, got, want)
		for pos := range query {
			for layer := 0; layer < m.Config.Layers; layer++ {
				ak, av := a.row(layer, pos)
				bk, bv := ref.row(layer, pos)
				exactWorkerFloats(t, ak, bk)
				exactWorkerFloats(t, av, bv)
			}
		}
		a.Close()
		ref.Close()
		if pool.Reserved != 0 || p.borrowers != 0 || len(pool.Free) != pool.Allocated {
			t.Fatal("reservation or reader survived release", n)
		}
		for _, page := range pool.Free {
			if slices.Contains(p.pages, page) {
				t.Fatalf("shared page in mutable pool at %d", n)
			}
		}
	}
	// The cap also applies when the last committed page is partial.
	p.capture(m, source, ids, 2)
	if len(p.ids) != 64 || len(p.pages) != 3 {
		t.Fatal("cap must limit searchable IDs and keep spare storage accounted")
	}
}

func TestPrefixTailEngineSubpageHitAndReset(t *testing.T) {
	m := promptPrefillTestModel(t, "q4")
	cfg := prefixEngineConfig(m)
	cfg.PrefixCacheTail = true
	e, err := NewEngine(m, fixedStepGenerateTokenizer{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ctx := streamTestContext(t)
	s := prefixSampling()
	s.MaxTokens = 1 // No output token has a KV row.
	for i, prompt := range []string{strings.Repeat("a", 30), strings.Repeat("a", 30), strings.Repeat("a", 31)} {
		want, err := Generate(ctx, m, fixedStepGenerateTokenizer{}, prompt, s)
		if err != nil {
			t.Fatal(err)
		}
		got, err := e.Generate(ctx, prompt, s)
		if err != nil || !reflect.DeepEqual(want, got) {
			t.Fatal("tail changed completion", got, want, err)
		}
		stats := e.Stats()
		if stats.PrefixTokens != []int64{0, 30, 61}[i] || stats.PrefixHits != int64(i) ||
			stats.PrefixBytes != cfg.PrefixCacheBytes/4 || stats.KVReservedBytes != 0 || stats.KVAllocatedBytes > cfg.CacheBytes {
			t.Fatal("unexpected sub-page accounting", i, stats)
		}
	}
	if err := e.ClearPrefix(ctx); err != nil {
		t.Fatal(err)
	}
	if e.Stats().PrefixBytes != 0 {
		t.Fatal("reset left a partial page")
	}
}
