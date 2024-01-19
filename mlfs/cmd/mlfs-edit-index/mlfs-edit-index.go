/*
e.g.

	mlfs-edit-index -index-url imagenet.idx.txt -o sub-imagenet.idx.txt -take 128 -from https://minddata.blob.core.windows.net -to https://tenplex.blob.core.windows.net
*/
package main

import (
	"flag"
	"log"
	"strings"

	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

var (
	idxFile = flag.String("index-url", "", "")
	output  = flag.String("o", "", "")

	take        = flag.Int(`take`, 0, ``)
	localize    = flag.Bool(`localize`, false, ``)
	replaceFrom = flag.String(`from`, ``, ``)
	replaceTo   = flag.String(`to`, ``, ``)
)

func main() { mlfs.Main(Main) }

func Main() error {
	log.Printf("loading from %q", *idxFile)
	idx, err := vfile.LoadIdxFile(*idxFile)
	if err != nil {
		return err
	}
	if *take > 0 {
		idx = idx[:*take]
	}
	if *localize {
		idx.SetHost(``)
	} else if len(*replaceFrom) > 0 {
		replaceURL(idx, *replaceFrom, *replaceTo)
	}
	log.Printf("saving to %q", *output)
	return vfile.SaveIdxFile(*output, idx)
}

func replaceURL(fs vfile.IndexedFiles, from, to string) {
	for i := range fs {
		fs[i].Filepath = strings.Replace(fs[i].Filepath, from, to, 1)
	}
}
