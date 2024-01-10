GO := $(if $(GO),$(GO),$(HOME)/local/go/bin/go)
CUDA := $(if $(CUDA),$(CUDA),$(shell [ -c /dev/nvidia0 ] && echo cuda))
TAGS := $(if $(TAGS),$(TAGS),)

default: binaries test


binaries:
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


