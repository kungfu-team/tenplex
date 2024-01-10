package main

import (
	"log"

	"github.com/kungfu-team/state-migrator/go/meta"
	"github.com/kungfu-team/state-migrator/go/statetransform"
)

func main() {
	conf, targetRank := meta.ReadFlags()
	log.Printf("config %+v", conf)
	log.Printf("target device %v", targetRank)
	if err := statetransform.MigrateState(conf, targetRank); err != nil {
		log.Panicf("Transformation for device %d failed with %v", targetRank, err)
	}
}
