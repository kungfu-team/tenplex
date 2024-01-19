package main

import (
	"log"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds"
)

func main() {
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	ds.ImagenetIndex.Check()
	ds.ImagenetMd5.Check()
	ds.Squad1Index.Check()
	ds.Squad1MD5.Check()

	ds.MnistTrainImages.Check()
	ds.MnistTrainLabels.Check()
	ds.MnistTestImages.Check()
	ds.MnistTestLabels.Check()
}
