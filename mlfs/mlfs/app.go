package mlfs

import (
	"flag"
	"os"
	"path"
	"time"
)

func Main(main func() error) {
	flag.Parse()
	t0 := time.Now()
	prog := path.Base(os.Args[0])
	defer func() { log.Printf("%s took %s", prog, time.Since(t0)) }()
	if err := main(); err != nil {
		log.Fatal(err)
	}
}
