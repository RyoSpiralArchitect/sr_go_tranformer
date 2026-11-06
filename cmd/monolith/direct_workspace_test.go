package main

import (
	"context"
	"testing"
)

func TestSingleDirectWorkspaceValidationMetadata(t *testing.T) {
	c := Config{Vocab: 258, Dim: 64, Hidden: 100, Heads: 4, KVHeads: 2, Layers: 1, Context: 128, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	defer cache.Close()
	var w decodeWorkspace
	if _, err := m.prefillBorrowed(context.Background(), []*KVCache{cache}, [][]int{{1, 2, 3}}, nil, &w); err != nil {
		t.Fatal(err)
	}
	if w.seen != nil || w.pools != nil {
		t.Fatal("single direct session allocated batch/pool validation maps")
	}
	assertWorkspaceReferencesCleared(t, &w)
}

func TestDirectWorkspaceSingleBatchTransitions(t *testing.T) {
	c := Config{Vocab: 258, Dim: 64, Hidden: 100, Heads: 4, KVHeads: 2, Layers: 1, Context: 128, RopeTheta: 10000, NormEps: 1e-5}
	m, err := NewModel(c, 19)
	if err != nil {
		t.Fatal(err)
	}
	a, b := m.NewCache(), m.NewCache()
	defer a.Close()
	defer b.Close()
	var w decodeWorkspace
	for _, caches := range [][]*KVCache{{a}, {a, b}, {a}} {
		chunks := make([][]int, len(caches))
		for i := range chunks {
			chunks[i] = []int{1, 2}
		}
		if _, err := m.prefillBorrowed(context.Background(), caches, chunks, nil, &w); err != nil {
			t.Fatal(err)
		}
		assertWorkspaceReferencesCleared(t, &w)
	}
	position, pages, bytes := a.Pos, len(a.Pages), w.floatBytes()
	if _, err := m.prefillBorrowed(context.Background(), []*KVCache{a, a}, [][]int{{1}, {2}}, nil, &w); err == nil {
		t.Fatal("duplicate cache accepted")
	}
	if a.Pos != position || len(a.Pages) != pages || w.floatBytes() != bytes {
		t.Fatal("rejected batch mutated cache or workspace capacity")
	}
	assertWorkspaceReferencesCleared(t, &w)
	if _, err := m.prefillBorrowed(context.Background(), []*KVCache{a}, [][]int{{3}}, nil, &w); err != nil {
		t.Fatal(err)
	}
	assertWorkspaceReferencesCleared(t, &w)
}

func TestKVDirectoryGrowthPreservesPages(t *testing.T) {
	m := &Model{Config: Config{Dim: 64, Heads: 4, KVHeads: 2, Layers: 1, Context: 1024}}
	cache := m.NewCache()
	defer cache.Close()
	for _, end := range []int{1, pageTokens, pageTokens + 1, 257, 513, 1024, 1023} {
		before := append([]*kvPage(nil), cache.Pages...)
		cache.ensurePages(end)
		if len(cache.Pages) != max(len(before), (end+pageTokens-1)/pageTokens) {
			t.Fatal("unexpected page coverage")
		}
		for i, p := range before {
			if cache.Pages[i] != p || p.Data[0] != float32(i+1) {
				t.Fatal("existing page changed during directory growth")
			}
		}
		for i, p := range cache.Pages {
			p.Data[0] = float32(i + 1)
		}
	}
}

func TestDirectKVPageDirectoryAllocation(t *testing.T) {
	m := &Model{Config: Config{Dim: 64, Heads: 4, KVHeads: 2, Layers: 1, Context: 1024}}
	pages := (m.Config.Context + pageTokens - 1) / pageTokens
	allocs := testing.AllocsPerRun(10, func() {
		cache := m.NewCache()
		cache.ensurePages(m.Config.Context)
		cache.Close()
	})
	t.Logf("bulk KV directory: %.1f allocations for %d pages", allocs, pages)
	if allocs > float64(2*pages+2) {
		t.Fatalf("bulk KV growth allocated %.1f objects for %d pages", allocs, pages)
	}
}
