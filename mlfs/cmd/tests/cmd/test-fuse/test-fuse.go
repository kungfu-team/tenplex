package main

import (
	"flag"

	"github.com/kungfu-team/tenplex/mlfs/fuse"
	"github.com/kungfu-team/tenplex/mlfs/utils"
)

var (
	mnt = flag.String("mnt", "", "")
)

func main() {
	flag.Parse()
	f, err := fuse.New(*mnt)
	if err != nil {
		utils.ExitErr(err)
	}
	f.Run()
}
