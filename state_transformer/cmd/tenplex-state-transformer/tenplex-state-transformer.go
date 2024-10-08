package main

import (
	"flag"
	"log"
	"time"

	"github.com/kungfu-team/tenplex/state_transformer/meta"
	"github.com/kungfu-team/tenplex/state_transformer/statetransform"
)

func main() {
	startTransform := time.Now()
	var conf meta.Config
	conf.RegisterFlags(flag.CommandLine)
	flag.Parse()
	conf.Complete()
	log.Printf("config %+v", conf)
	log.Printf("target device %v", conf.TargetRank)
	if err := statetransform.MigrateState(&conf, conf.TargetRank); err != nil {
		log.Panicf("Transformation for device %d failed with %v", conf.TargetRank, err)
	}
	log.Printf("State transformation took %s", time.Since(startTransform))
}
