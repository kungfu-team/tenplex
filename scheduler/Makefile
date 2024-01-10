WHICH_GO := $(shell which go)
DEFAULT_GO := $(if $(WHICH_GO),$(WHICH_GO),$(HOME)/local/go/bin/go)
GO := $(if $(GO),$(GO),$(DEFAULT_GO))

TENPLEX_PREFIX := $(HOME)/.tenplex

binaries:
	GOBIN=$(TENPLEX_PREFIX)/scheduler/bin $(GO) install -v ./cmd/...

# state-migrator:
# 	GO=$(GO) make -C ./vendors/state-migrator/go

# all: binaries mlfs state-migrator

update:
	$(GO) get -u ./...

tidy:
	$(GO) mod tidy

u: update tidy

deb:
	./scripts/build-deb.sh

sys-install: deb
	sudo systemctl stop tenplex-scheduler
	sudo dpkg -i build/*.deb
	sudo systemctl start tenplex-scheduler
