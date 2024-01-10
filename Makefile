PREFIX := $(if $(PREFIX),$(PREFIX),$(HOME)/local)
WHICH_GO := $(shell which go)
DEFAULT_GO := $(if $(WHICH_GO),$(WHICH_GO),$(HOME)/local/go/bin/go)
GO := $(if $(GO),$(GO),$(DEFAULT_GO))
CUDA := $(if $(CUDA),$(CUDA),$(shell [ -c /dev/nvidia0 ] && echo cuda))
TAGS := $(if $(TAGS),$(TAGS),)
# BIN_DIR := $(if $(BIN_DIR),$(BIN_DIR),$(HOME)/.tenplex/bin)
BIN_DIR := $(CURDIR)/bin

default: binaries test


binaries: bin
	GOBIN=$(CURDIR)/bin $(GO) install -v -tags "$(TAGS)" ./...

install:
	$(GO) install -v -tags "$(TAGS)" ./...

test:
	$(GO) test -v -tags "$(TAGS)" ./...

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
