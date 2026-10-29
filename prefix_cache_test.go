// Copyright 2026 RyoSpiralArchitect
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func prefixEngineConfig(m *Model) EngineConfig {
	page := int64(2 * pageTokens * m.Config.KVDim() * m.Config.Layers * 4)
	return EngineConfig{MaxBatch: 4, Queue: 8, CacheBytes: 20 * page, PrefixCacheBytes: 4 * page, StreamBuffer: 32, KernelWorkers: 4}
}
func prefixSampling() Sampling {
	s := defaultSampling()
	s.MaxTokens = 8
	s.Temperature = .7
	s.Seed = 19
	return s
}

func TestPrefixEngineParityAndReset(t *testing.T) {
	m := workerTestModel(t, 2)
	c := prefixEngineConfig(m)
	e, err := NewEngine(m, Tokenizer{}, c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	ctx := streamTestContext(t)
	prompt := strings.Repeat("a", 75)
	s := prefixSampling()
	first, err := e.Generate(ctx, prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	before := e.Stats()
	if before.PrefixBytes == 0 || before.PrefixHits != 0 {
		t.Fatal("first call did not populate prefix", before)
	}
	for _, tail := range []string{"", " tail", " other"} {
		p := prompt + tail
		want, err := Generate(ctx, m, Tokenizer{}, p, s)
		if err != nil {
			t.Fatal(err)
		}
		got, err := e.Generate(ctx, p, s)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatal("cached completion mismatch", got, want, err)
		}
		if tail == "" && !reflect.DeepEqual(got, first) {
			t.Fatal("repeat changed")
		}
	}
	stats := e.Stats()
	if stats.PrefixHits != 3 || stats.PrefixTokens != 3*64 {
		t.Fatal("prefix not reused", stats)
	}
	if stats.KVAllocatedBytes > c.CacheBytes || stats.PrefixBytes > c.PrefixCacheBytes {
		t.Fatal("budget exceeded", stats)
	}
	// Retain usage semantics: reused input is accounted separately from computed input.
	if stats.InputTokens >= int64(4*76+len(" tail")+len(" other")) {
		t.Fatal("prefix still computed")
	}
	if err = e.ClearPrefix(ctx); err != nil {
		t.Fatal(err)
	}
	if e.Stats().PrefixBytes != 0 {
		t.Fatal("reset retained unborrowed prefix")
	}
	if _, err = e.Generate(ctx, prompt, s); err != nil {
		t.Fatal(err)
	}
	if e.Stats().PrefixHits != 3 {
		t.Fatal("clear did not invalidate lookup")
	}
	recorder := httptest.NewRecorder()
	e.Handler().ServeHTTP(recorder, httptest.NewRequest("GET", "/metrics", nil))
	if !strings.Contains(recorder.Body.String(), "monolith_prefix_tokens_reused_total 192") {
		t.Fatal(recorder.Body.String())
	}
}

func TestPrefixSnapshotSharedPagesAndDivergentTails(t *testing.T) {
	m := workerTestModel(t, 2)
	ctx := context.Background()
	source := m.NewCache()
	defer source.Close()
	ids := prefillTokens(80, 2)
	if _, err := m.PrefillBatch(ctx, []*KVCache{source}, [][]int{ids}); err != nil {
		t.Fatal(err)
	}
	var snapshot prefixSnapshot
	snapshot.capture(m, source, ids, 3)
	if len(snapshot.pages) != 2 {
		t.Fatal("partial page retained")
	}
	frozen := make([][]float32, len(snapshot.pages))
	for i, p := range snapshot.pages {
		frozen[i] = append([]float32(nil), p.Data...)
	}
	query := append(append([]int(nil), ids[:64]...), 17, 18, 19)
	n := snapshot.match(m, query)
	if n != 2 {
		t.Fatalf("matched %d pages", n)
	}
	a, b := m.NewCache(), m.NewCache()
	defer a.Close()
	defer b.Close()
	a.shared, a.sharedPages = &snapshot, n
	snapshot.borrowers++
	a.Pages = append(a.Pages, snapshot.pages[:n]...)
	a.Pos = n * pageTokens
	if a.Pages[0] != snapshot.pages[0] {
		t.Fatal("snapshot pages copied per hit")
	}
	before, err := m.PrefillBatch(ctx, []*KVCache{b}, [][]int{query})
	if err != nil {
		t.Fatal(err)
	}
	after, err := m.PrefillBatch(ctx, []*KVCache{a}, [][]int{query[a.Pos:]})
	if err != nil {
		t.Fatal(err)
	}
	exactWorkerFloats(t, before, after)
	for i, p := range snapshot.pages {
		exactWorkerFloats(t, frozen[i], p.Data)
	}
	for pos := 0; pos < a.Pos; pos++ {
		for layer := 0; layer < m.Config.Layers; layer++ {
			ak, av := a.row(layer, pos)
			bk, bv := b.row(layer, pos)
			exactWorkerFloats(t, ak, bk)
			exactWorkerFloats(t, av, bv)
		}
	}
	// Exact IDs, model identity and revision all participate in lookup.
	query[40]++
	if snapshot.match(m, query) != 1 {
		t.Fatal("divergence inside second page reused")
	}
	if snapshot.match(m, ids[:64]) != 1 {
		t.Fatal("full prompt reused without a logits forward")
	}
	other := workerTestModel(t, 2)
	if snapshot.match(other, ids) != 0 {
		t.Fatal("foreign model matched")
	}
	m.Revision++
	if snapshot.match(m, ids) != 0 {
		t.Fatal("stale revision matched")
	}
	m.Revision--
	// A live reader prevents replacement; invalidation prevents new readers and
	// releases bytes only when the final existing reader closes.
	oldIDs := append([]int(nil), snapshot.ids...)
	snapshot.capture(m, b, query, 1)
	if !reflect.DeepEqual(oldIDs, snapshot.ids) {
		t.Fatal("borrowed entry replaced")
	}
	a.Pos = 1
	if _, err = m.PrefillBatch(ctx, []*KVCache{a}, [][]int{{1}}); err == nil {
		t.Fatal("rollback wrote shared prefix")
	}
	for i, p := range snapshot.pages {
		exactWorkerFloats(t, frozen[i], p.Data)
	}
	snapshot.clear()
	if len(snapshot.pages) == 0 || snapshot.match(m, ids) != 0 {
		t.Fatal("live clear broke ownership")
	}
	a.Close()
	if len(snapshot.pages) != 0 || snapshot.borrowers != 0 {
		t.Fatal("last reader did not release invalidated prefix")
	}
}

func TestPrefixPoolReturnsOnlyOwnedPages(t *testing.T) {
	m := workerTestModel(t, 2)
	source := m.NewCache()
	defer source.Close()
	ids := prefillTokens(64, 3)
	if _, err := m.PrefillBatch(context.Background(), []*KVCache{source}, [][]int{ids}); err != nil {
		t.Fatal(err)
	}
	var snapshot prefixSnapshot
	snapshot.capture(m, source, ids, 2)
	p := pagePool{PageElements: len(snapshot.pages[0].Data), Limit: 4, Reserved: 4}
	c := m.NewCache()
	c.pool = &p
	c.reserved = 4
	c.shared = &snapshot
	c.sharedPages = 2
	c.Pos = 64
	c.Pages = append(c.Pages, snapshot.pages...)
	snapshot.borrowers++
	if _, err := m.PrefillBatch(context.Background(), []*KVCache{c}, [][]int{{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	c.Close()
	c.Close()
	if p.Allocated != 1 || len(p.Free) != 1 || p.Reserved != 0 || snapshot.borrowers != 0 {
		t.Fatal("incorrect pool ownership", p, snapshot.borrowers)
	}
	for _, page := range snapshot.pages {
		if page == p.Free[0] {
			t.Fatal("immutable page returned to mutable pool")
		}
	}
}

func TestPrefixConcurrentCancelAndClose(t *testing.T) {
	m := workerTestModel(t, 2)
	c := prefixEngineConfig(m)
	e, err := NewEngine(m, Tokenizer{}, c)
	if err != nil {
		t.Fatal(err)
	}
	ctx := streamTestContext(t)
	prompt := strings.Repeat("cat", 25)
	s := prefixSampling()
	want, err := e.Generate(ctx, prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := e.Generate(ctx, prompt, s)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Errorf("shared prefix: %v", err)
			}
		}()
	}
	wg.Wait()
	stream, err := e.Stream(ctx, prompt, s)
	if err != nil {
		t.Fatal(err)
	}
	<-stream.Events
	stream.Close()
	<-stream.Released()
	_, err = stream.Wait()
	if err != nil && !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err = e.ClearPrefix(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err = e.Generate(ctx, prompt, s); err != nil {
		t.Fatal(err)
	}
	e.Close()
	if e.prefix.borrowers != 0 || len(e.prefix.pages) != 0 || e.Stats().PrefixBytes != 0 || e.Stats().KVReservedBytes != 0 {
		t.Fatal("shutdown retained prefix or reservation", e.Stats())
	}
	if !errors.Is(e.ClearPrefix(ctx), ErrClosed) {
		t.Fatal("clear accepted after close")
	}
}

func TestPrefixBudgetAndFlags(t *testing.T) {
	m := workerTestModel(t, 2)
	c := prefixEngineConfig(m)
	for _, bad := range []int64{-1, c.CacheBytes, c.CacheBytes + 1} {
		c.PrefixCacheBytes = bad
		if _, err := newEngine(m, Tokenizer{}, c); err == nil {
			t.Fatal("invalid allowance accepted", bad)
		}
	}
	c = prefixEngineConfig(m)
	c.PrefixCacheBytes = 1
	if _, err := newEngine(m, Tokenizer{}, c); err == nil {
		t.Fatal("sub-page prefix accepted")
	}
	for _, action := range []string{"chat", "serve"} {
		for _, args := range [][]string{{"-prefix-cache-mib", "-1"}, {"-prefix-cache-mib", "1099511627776"}, {"-kv-mib", "4", "-prefix-cache-mib", "4"}} {
			if err := runInferenceAction(context.Background(), action, args, true); err == nil {
				t.Fatal("bad prefix flags accepted", action, args)
			}
		}
		if err := runInferenceAction(context.Background(), action, []string{"-kv-mib", "4", "-prefix-cache-mib", "1"}, true); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPrefixFailedSessionDoesNotPublish(t *testing.T) {
	m := workerTestModel(t, 2)
	c := prefixEngineConfig(m)
	c.StreamBuffer = 1
	e, err := NewEngine(m, Tokenizer{}, c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	s := prefixSampling()
	s.MaxTokens = 24
	stream, err := e.Stream(streamTestContext(t), strings.Repeat("a", 70), s)
	if err != nil {
		t.Fatal(err)
	}
	// Deliberately leave the one-slot event mailbox undrained.
	_, err = stream.Wait()
	<-stream.Released()
	if !errors.Is(err, ErrSlowConsumer) {
		t.Fatal("expected bounded slow-consumer failure", err)
	}
	if stats := e.Stats(); stats.PrefixBytes != 0 || stats.PrefixHits != 0 || stats.KVReservedBytes != 0 {
		t.Fatal("failed request published prefix", stats)
	}
}
