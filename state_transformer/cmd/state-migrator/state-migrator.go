package main

import (
	"log"

	"github.com/kungfu-team/tenplex/state_transformer/go/meta"
	"github.com/kungfu-team/tenplex/state_transformer/go/statetransform"
)

func main() {
	conf, targetRank := meta.ReadFlags()
	log.Printf("config %+v", conf)
	log.Printf("target device %v", targetRank)
	if err := statetransform.MigrateState(conf, targetRank); err != nil {
		log.Panicf("Transformation for device %d failed with %v", targetRank, err)
	}
}
