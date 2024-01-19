PREFIX := $(if $(PREFIX),$(PREFIX),$(HOME)/local)
WHICH_GO := $(shell which go)
DEFAULT_GO := $(if $(WHICH_GO),$(WHICH_GO),$(HOME)/local/go/bin/go)
GO := $(if $(GO),$(GO),$(DEFAULT_GO))
CUDA := $(if $(CUDA),$(CUDA),$(shell [ -c /dev/nvidia0 ] && echo cuda))
# BIN_DIR := $(if $(BIN_DIR),$(BIN_DIR),$(HOME)/.tenplex/bin)
BIN_DIR := $(CURDIR)/bin

GO_MOD := $(shell ./show-go-mod.sh)
buildinfo := $(GO_MOD)/mlfs/buildinfo
LDFLAGS += -X $(buildinfo).BuildHost=$(shell hostname)
LDFLAGS += -X $(buildinfo).BuildTimestamp=$(shell date +%s)
LDFLAGS += -X $(buildinfo).GitCommit=$(shell git rev-list -1 HEAD)
LDFLAGS += -X $(buildinfo).GitBranch=$(shell git rev-parse --abbrev-ref HEAD)
LDFLAGS += -X $(buildinfo).GitRev=$(shell git rev-list --count HEAD)

default: binaries test

binaries: bin
	GOBIN=$(PWD)/bin $(GO) install -ldflags "$(LDFLAGS)"  -v ./...

install:
	$(GO) install -ldflags "$(LDFLAGS)" -v ./...

test:
	$(GO) test -v ./...

update:
	$(GO) get -u ./...

clean:
	$(GO) clean -v -cache ./...

tidy:
	$(GO) mod tidy

format:
	$(GO) fmt ./...

i: install


u: update tidy


t: test


bin:
	mkdir -p $(BIN_DIR)

deb: binaries
	./scripts/pack.sh

sys-install: deb
	sudo dpkg -i ./build/*.deb
