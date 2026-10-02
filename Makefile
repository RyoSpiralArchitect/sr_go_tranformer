GO ?= ./scripts/go.sh

.PHONY: build test race vet bench demo
build:
	CGO_ENABLED=0 $(GO) build -trimpath -o monolith .
test:
	$(GO) test ./...
race:
	$(GO) test -race ./...
vet:
	$(GO) vet ./...
bench:
	$(GO) test -run '^$$' -bench . -benchmem -cpu 4
demo:
	$(GO) run . demo
