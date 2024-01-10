package main

import (
	"context"
	"flag"

	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/mlfs/utils"
)

func main() {
	var d mlfs.Daemon
	d.RegisterFlags(flag.CommandLine)
	flag.Parse()
	utils.LogArgs()
	d.RunCtx(context.Background())
}
