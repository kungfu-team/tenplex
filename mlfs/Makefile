WHICH_GO := $(shell which go)
DEFAULT_GO := $(if $(WHICH_GO),$(WHICH_GO),$(HOME)/local/go/bin/go)
GO := $(if $(GO),$(GO),$(DEFAULT_GO))

buildinfo := github.com/kungfu-team/mlfs/buildinfo
LDFLAGS += -X $(buildinfo).BuildHost=$(shell hostname)
LDFLAGS += -X $(buildinfo).BuildTimestamp=$(shell date +%s)
LDFLAGS += -X $(buildinfo).GitCommit=$(shell git rev-list -1 HEAD)
LDFLAGS += -X $(buildinfo).GitBranch=$(shell git rev-parse --abbrev-ref HEAD)
LDFLAGS += -X $(buildinfo).GitRev=$(shell git rev-list --count HEAD)

binaries:
	GOBIN=$(PWD)/bin $(GO) install -ldflags "$(LDFLAGS)"  -v ./cmd/...

# V = -v

test:
	$(GO) test $(V) ./...

install:
	$(GO) install -ldflags "$(LDFLAGS)" -v ./cmd/...

i: install

deb: binaries
	./scripts/pack.sh

sys-install: deb
	sudo dpkg -i ./build/*.deb

reload:
	sudo systemctl daemon-reload
	sudo systemctl restart mlfs
	systemctl status mlfs

stop:
	sudo systemctl stop mlfs
