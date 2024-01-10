package main

import (
	"log"
	"os"
	"path"
	"time"

	"github.com/kungfu-team/mlfs/ds"
	"github.com/kungfu-team/mlfs/mlfs/t"
)

var pwd, _ = os.Getwd()

func main() {
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	dt := t.DistTest{
		HTTPPort: 30000,
		CtrlPort: 40000,
		// Mount:    path.Join(pwd, `mnt`),
		Tmp:   path.Join(pwd, `tmp`),
		JobID: `A`,
		DP:    4,
		DS:    ds.Imagenet,
	}
	dt.Run()
}
