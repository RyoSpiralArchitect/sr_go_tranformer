package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestMain(m *testing.M) {
	dir, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			fmt.Fprintln(os.Stderr, "repository root not found")
			os.Exit(1)
		}
		dir = parent
	}
	if err := os.Chdir(dir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}
