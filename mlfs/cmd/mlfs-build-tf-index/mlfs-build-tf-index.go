package main

import (
	"flag"
	"log"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/tfrecord"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

var (
	m        = flag.Int("m", 2, "")
	filename = flag.String("output", "a.idx.txt", "")
)

func main() {
	flag.Parse()
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	idx, err := tfrecord.BuildIndex(flag.Args(), *m)
	if err != nil {
		log.Printf("%v", err)
		return
	}
	if err := vfile.SaveIdxFile(*filename, idx); err != nil {
		log.Printf("%v", err)
		return
	}
	log.Printf("generated %s", *filename)
}
