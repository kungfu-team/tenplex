package main

import (
	"flag"
	"log"

	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/tensor"
)

var (
	port = flag.Int("p", 8080, ``)
)

func main() {
	flag.Parse()
	c, err := mlfs.NewClient(*port)
	if err != nil {
		log.Panic(err)
	}
	x := tensor.New(`f32`, 2, 2, 2)
	if err := c.Upload(`/a/b/c`, x.Data); err != nil {
		log.Panic(err)
	}
	log.Printf("done")
}
