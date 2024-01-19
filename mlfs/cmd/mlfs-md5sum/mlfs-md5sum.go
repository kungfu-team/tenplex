package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/hash"
	"github.com/kungfu-team/tenplex/mlfs/iotrace"
	"github.com/kungfu-team/tenplex/mlfs/utils"
)

var (
	m        = flag.Int("m", 2, "")
	filename = flag.String("output", "a.md5.txt", "")
)

func main() {
	flag.Parse()
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	names := flag.Args()
	hs, err := buildIndex(names, *m)
	if err != nil {
		utils.ExitErr(err)
	}
	f, err := os.Create(*filename)
	if err != nil {
		utils.ExitErr(err)
	}
	defer f.Close()
	for i, h := range hs {
		fmt.Fprintf(f, "%s %s\n", h, names[i])
	}
}

func buildIndex(names []string, m int) ([]string, error) {
	log.Printf("building md5sum for %d files", len(names))
	c := iotrace.NewCounter()
	defer iotrace.Reporter(c, ``).Stop()
	sums := make([]string, len(names))
	var failed int32
	ch := make(chan struct{}, m)
	var wg sync.WaitGroup
	for i, filename := range names {
		wg.Add(1)
		go func(i int, filename string) {
			defer wg.Done()
			ch <- struct{}{}
			defer func() { <-ch }()
			rs, err := hash.FileMD5(c, filename)
			if err != nil {
				log.Printf("failed to read %s: %v", filename, err)
				atomic.AddInt32(&failed, 1)
				return
			}
			log.Printf("got %s from %s", rs, filename)
			sums[i] = rs
		}(i, filename)
	}
	wg.Wait()
	if failed > 0 {
		return nil, fmt.Errorf("%d failed", failed)
	}
	return sums, nil
}
