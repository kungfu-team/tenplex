package main

import (
	"log"
	"time"

	"github.com/kungfu-team/tenplex/state_transformer/meta"
	"github.com/kungfu-team/tenplex/state_transformer/statetransform"
)

func main() {
	startTransform := time.Now()
	conf, targetRank := meta.ReadFlags()
	log.Printf("config %+v", conf)
	log.Printf("target device %v", targetRank)
	if err := statetransform.MigrateState(conf, targetRank); err != nil {
		log.Panicf("Transformation for device %d failed with %v", targetRank, err)
	}
	log.Printf("State transformation took %s", time.Since(startTransform))
}
