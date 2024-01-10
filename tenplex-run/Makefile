WHICH_GO := $(shell which go)
DEFAULT_GO := $(if $(WHICH_GO),$(WHICH_GO),$(HOME)/local/go/bin/go)
GO := $(if $(GO),$(GO),$(DEFAULT_GO))

default: binaries test


binaries:
	# GOBIN=$(HOME)/.tenplex/bin $(GO) install -v ./...
	GOBIN=$(PWD)/bin $(GO) install -v ./...

install:
	$(GO) install -v ./...

test:
	$(GO) test -v ./...

update:
	$(GO) get -u ./...

tidy:
	$(GO) mod tidy

format:
	$(GO) fmt ./...

i: install


u: update


t: test
