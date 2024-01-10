package tfrecord

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"sync"
	"sync/atomic"

	"github.com/kungfu-team/mlfs/iotrace"
	"github.com/kungfu-team/mlfs/uri"
	"github.com/kungfu-team/mlfs/vfs/vfile"
)

func BuildIndex(names []string, m int) (vfile.IndexedFiles, error) {
	log.Printf("building index for %d files", len(names))
	c := iotrace.NewCounter()
	defer iotrace.Reporter(c, ``).Stop()
	idx := make(vfile.IndexedFiles, len(names))
	var failed int32
	ch := make(chan struct{}, m)
	var wg sync.WaitGroup
	for i, filename := range names {
		wg.Add(1)
		go func(i int, filename string) {
			defer wg.Done()
			ch <- struct{}{}
			defer func() { <-ch }()
			rs, err := readTFRecords(c, filename)
			if err != nil {
				log.Printf("failed to read %s: %v", filename, err)
				atomic.AddInt32(&failed, 1)
				return
			}
			log.Printf("got %d from %s", len(rs), filename)
			idx[i].Filepath = filename
			idx[i].Ranges = rs
		}(i, filename)
	}
	wg.Wait()
	if failed > 0 {
		return nil, fmt.Errorf("%d failed", failed)
	}
	return idx, nil
}

func readTFRecords(c *iotrace.Counter, filename string) (vfile.Ranges, error) {
	f, err := uri.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	f = io.NopCloser(bufio.NewReader(f))
	var rs vfile.Ranges
	var off uint64
	for i := 0; ; i++ {
		info, _, err := ReadTFRecord(iotrace.TraceReader(f, c))
		if err == io.EOF {
			break
		}
		if err != nil {
			return rs, err
		}
		r := vfile.Range{
			Begin: off,
			End:   off + uint64(info.Len) + 16,
		}
		off = r.End
		rs = append(rs, r)
	}
	return rs, nil
}
