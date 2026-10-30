// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestPrefixGrowthFiveTurnsAndBranch(t *testing.T) {
	for _, format := range []string{"f32", "q8", "q4"} {
		t.Run(format, func(t *testing.T) {
			m := promptPrefillTestModel(t, format)
			cfg := prefixEngineConfig(m)
			pageBytes := int64(2 * pageTokens * m.Config.KVDim() * m.Config.Layers * 4)
			cfg.PrefixCacheBytes = 6 * pageBytes
			e, err := NewEngine(m, fixedStepGenerateTokenizer{}, cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			ctx := streamTestContext(t)
			s := prefixSampling()
			var reused int64
			for turn := 0; turn < 5; turn++ {
				prompt := strings.Repeat("a", 65+32*turn)
				want, err := Generate(ctx, m, fixedStepGenerateTokenizer{}, prompt, s)
				if err != nil {
					t.Fatal(err)
				}
				got, err := e.Generate(ctx, prompt, s)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatal("growing completion mismatch", turn, err)
				}
				if turn > 0 {
					reused += int64((turn + 1) * pageTokens)
				}
				stats := e.Stats()
				t.Logf("format=%s turn=%d input=%d reused=%d prefix_bytes=%d kv_allocated_bytes=%d", format, turn+1, stats.InputTokens, stats.PrefixTokens, stats.PrefixBytes, stats.KVAllocatedBytes)
				if stats.PrefixTokens != reused || stats.PrefixBytes != int64(turn+2)*pageBytes || stats.PrefixHits != int64(turn) {
					t.Fatal("prefix did not grow", turn, stats)
				}
				if stats.KVAllocatedBytes > cfg.CacheBytes || stats.PrefixBytes > cfg.PrefixCacheBytes || stats.KVReservedBytes != 0 {
					t.Fatal("budget/release mismatch", stats)
				}
			}
			// Divergence inside page three reuses exactly two pages; replacing a
			// larger snapshot keeps its spare storage charged to the allowance.
			branch := strings.Repeat("a", 80) + "b" + strings.Repeat("c", 80)
			want, err := Generate(ctx, m, fixedStepGenerateTokenizer{}, branch, s)
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Generate(ctx, branch, s)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatal("branch mismatch", err)
			}
			stats := e.Stats()
			if stats.PrefixTokens != reused+64 || stats.PrefixBytes != 6*pageBytes {
				t.Fatal("branch reuse/retention mismatch", stats)
			}
			if err := e.ClearPrefix(ctx); err != nil {
				t.Fatal(err)
			}
			if e.Stats().PrefixBytes != 0 {
				t.Fatal("reset retained pages")
			}
			e.Close()
			if e.Stats().KVReservedBytes != 0 || e.Stats().PrefixBytes != 0 {
				t.Fatal("close retained ownership")
			}
		})
	}
}

func TestPrefixGrowthSoleBorrowerAndPoolOwnership(t *testing.T) {
	m := promptPrefillTestModel(t, "q4")
	ctx := context.Background()
	ids := prefillTokens(129, 5)
	source := m.NewCache()
	defer source.Close()
	if _, err := m.PrefillBatch(ctx, []*KVCache{source}, [][]int{ids[:64]}); err != nil {
		t.Fatal(err)
	}
	var p prefixSnapshot
	p.capture(m, source, ids[:64], 6)
	pool := pagePool{PageElements: len(p.pages[0].Data), Limit: 8, Reserved: 8}
	borrow := func() *KVCache {
		c := m.NewCache()
		c.shared = &p
		c.sharedPages = 2
		c.Pos = 64
		c.Pages = append(c.Pages, p.pages[:2]...)
		p.borrowers++
		return c
	}
	a, other := borrow(), borrow()
	defer a.Close()
	defer other.Close()
	a.pool, a.reserved = &pool, 8
	if _, err := m.PrefillBatch(ctx, []*KVCache{a}, [][]int{ids[64:127]}); err != nil {
		t.Fatal(err)
	}
	originalIDs := append([]int(nil), p.ids...)
	firstPage := p.pages[0]
	p.capture(m, a, ids[:128], 6)
	if !reflect.DeepEqual(p.ids, originalIDs) || len(p.pages) != 2 {
		t.Fatal("replaced while another reader remained")
	}
	other.Close()
	p.capture(m, a, ids[:128], 6)
	if len(p.ids) != 96 || len(p.pages) != 3 || p.pages[0] != firstPage {
		t.Fatal("captured uncommitted final token or replaced shared prefix")
	}
	if _, err := m.PrefillBatch(ctx, []*KVCache{a}, [][]int{ids[127:128]}); err != nil {
		t.Fatal(err)
	}
	p.capture(m, a, ids, 6)
	if len(p.ids) != 128 || len(p.pages) != 4 {
		t.Fatal("new full page not captured")
	}
	for i := range p.pages {
		exactWorkerFloats(t, p.pages[i].Data, a.Pages[i].Data)
		if i >= a.sharedPages && p.pages[i] == a.Pages[i] {
			t.Fatal("mutable pool page transferred into snapshot")
		}
	}
	kept := append([]*kvPage(nil), p.pages...)
	a.Close()
	if pool.Reserved != 0 || len(pool.Free) != 2 || p.borrowers != 0 {
		t.Fatal("suffix ownership not released", pool, p.borrowers)
	}
	for _, free := range pool.Free {
		for _, page := range kept {
			if free == page {
				t.Fatal("snapshot returned to mutable pool")
			}
		}
	}
	// Invalidation while a reader exists prevents that reader from reviving it.
	c := borrow()
	p.clear()
	p.capture(m, c, ids[:64], 6)
	if p.model != nil || len(p.ids) != 0 {
		t.Fatal("invalidated prefix revived")
	}
	c.Close()
	if len(p.pages) != 0 {
		t.Fatal("invalidated pages retained after last reader")
	}
}

func TestPrefixGrowthFailureDoesNotPublish(t *testing.T) {
	for _, cancel := range []bool{false, true} {
		t.Run(fmt.Sprint("cancel=", cancel), func(t *testing.T) {
			m := promptPrefillTestModel(t, "f32")
			cfg := prefixEngineConfig(m)
			cfg.StreamBuffer = 1
			e, err := NewEngine(m, fixedStepGenerateTokenizer{}, cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			ctx := streamTestContext(t)
			s := prefixSampling()
			s.MaxTokens = 1
			if _, err := e.Generate(ctx, strings.Repeat("a", 65), s); err != nil {
				t.Fatal(err)
			}
			before := e.Stats()
			s.MaxTokens = 32
			stream, err := e.Stream(ctx, strings.Repeat("a", 161), s)
			if err != nil {
				t.Fatal(err)
			}
			if cancel {
				<-stream.Events
				stream.Close()
			}
			_, err = stream.Wait()
			<-stream.Released()
			if !errors.Is(err, ErrSlowConsumer) && !errors.Is(err, context.Canceled) {
				t.Fatal("failed session unexpectedly succeeded", err)
			}
			stats := e.Stats()
			if stats.PrefixBytes != before.PrefixBytes || stats.PrefixTokens != 64 || stats.KVReservedBytes != 0 {
				t.Fatal("failed borrower published growth", stats)
			}
		})
	}
}

func TestPrefixGrowthResetDuringBorrow(t *testing.T) {
	m := promptPrefillTestModel(t, "f32")
	cfg := prefixEngineConfig(m)
	e, err := newEngine(m, fixedStepGenerateTokenizer{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Stage a clear command while compute is gated. Buffering only this test's
	// actor mailbox makes delivery deterministic; the reset operation is real.
	e.clearPrefix = make(chan chan struct{}, 1)
	prompt := strings.Repeat("a", 161)
	ids := (Tokenizer{}).Encode(prompt, true, false)
	source := m.NewCache()
	defer source.Close()
	if _, err := m.PrefillBatch(context.Background(), []*KVCache{source}, [][]int{ids[:64]}); err != nil {
		t.Fatal(err)
	}
	e.prefix.capture(m, source, ids[:64], 4)
	s := prefixSampling()
	stream, err := e.Stream(streamTestContext(t), prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	gate := startGatedEngine(t, e)
	if e.Stats().PrefixHits != 1 {
		t.Fatal("fixture did not borrow")
	}
	done := make(chan struct{})
	e.clearPrefix <- done
	gate.open()
	waitLifecycle(t, done)
	got, err := stream.Wait()
	<-stream.Released()
	if err != nil {
		t.Fatal(err)
	}
	want, err := Generate(context.Background(), m, fixedStepGenerateTokenizer{}, prompt, s)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("reset changed completion", err)
	}
	if e.Stats().PrefixBytes != 0 || e.Stats().KVReservedBytes != 0 {
		t.Fatal("old epoch republished prefix", e.Stats())
	}
}

func TestPrefixGrowthCancelDuringBorrow(t *testing.T) {
	m := promptPrefillTestModel(t, "q4")
	e, err := newEngine(m, fixedStepGenerateTokenizer{}, prefixEngineConfig(m))
	if err != nil {
		t.Fatal(err)
	}
	prompt := strings.Repeat("a", 161)
	ids := (Tokenizer{}).Encode(prompt, true, false)
	source := m.NewCache()
	defer source.Close()
	if _, err := m.PrefillBatch(context.Background(), []*KVCache{source}, [][]int{ids[:64]}); err != nil {
		t.Fatal(err)
	}
	e.prefix.capture(m, source, ids[:64], 4)
	ctx, cancel := context.WithCancel(streamTestContext(t))
	defer cancel()
	stream, err := e.Stream(ctx, prompt, prefixSampling())
	if err != nil {
		t.Fatal(err)
	}
	gate := startGatedEngine(t, e)
	before := e.Stats()
	if before.PrefixHits != 1 {
		t.Fatal("fixture did not borrow")
	}
	cancel()
	gate.open()
	_, err = stream.Wait()
	<-stream.Released()
	if !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation was not exercised", err)
	}
	after := e.Stats()
	if after.PrefixBytes != before.PrefixBytes || after.KVReservedBytes != 0 {
		t.Fatal("canceled borrower published prefix", after)
	}
}
