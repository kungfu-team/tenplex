package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"net/url"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/mlfs/uri"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

// ./bin/mlfs-check-index $(cat tests/data/*.json | jq -r '."idx-url"')

func main() { mlfs.Main(Main) }

func Main() error {
	for _, f := range flag.Args() {
		if err := checkIndexFile(f); err != nil {
			return err
		}
	}
	return nil
}

func checkIndexFile(filename string) error {
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	fs, err := vfile.LoadIdxFile(filename)
	if err != nil {
		return err
	}
	domain := getDomain(filename)
	for i, f := range fs {
		log.Printf("checking %d/%d", i+1, len(fs))
		info, err := uri.Stat(f.Filepath)
		if err != nil {
			return err
		}
		if info.Size < 0 {
			return errCannotGetSize
		}
		if size := int64(f.IndexedBytes()); size != info.Size {
			return fmt.Errorf("%v: %d, expect %d", errUnexpectedSize, info.Size, size)
		}
		if getDomain(f.Filepath) != domain {
			log.Printf("%s file has different domain", f.Filepath)
		}
	}
	fmt.Printf("OK: %s\n", filename)
	return nil
}

var (
	errCannotGetSize  = errors.New(`can't get size`)
	errUnexpectedSize = errors.New(`unexpected get size`)
)

func getDomain(filepath string) string {
	u, err := url.Parse(filepath)
	if err != nil {
		return ""
	}
	return u.Host
}
