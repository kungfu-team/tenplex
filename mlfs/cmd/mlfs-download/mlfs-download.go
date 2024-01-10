package main

import (
	"flag"
	"io"
	"log"
	"net/url"
	"os"
	"path"
	"strings"
	"time"

	"github.com/kungfu-team/mlfs/mlfs"
	"github.com/kungfu-team/mlfs/uri"
	"github.com/kungfu-team/mlfs/utils"
	"github.com/kungfu-team/mlfs/vfs/vfile"
)

var (
	localRoot = flag.String("o", ".", "")
	idxFile   = flag.String("index-url", "", "")
)

func main() { mlfs.Main(Main) }

func Main() error {
	t0 := time.Now()
	fs, err := vfile.LoadIdxFile(*idxFile)
	if err != nil {
		return err
	}
	for i, f := range fs {
		localFile := path.Join(*localRoot, getRelPath(f.Filepath))
		if err := downloadOne(f.Filepath, localFile); err != nil {
			return err
		}
		log.Printf("downloaded %d/%d %s -> %s", i+1, len(fs), f.Filepath, localFile)
		utils.LogETA(t0, i+1, len(fs))
	}
	return nil
}

func downloadOne(filepath, localFile string) error {
	r, err := uri.Open(filepath)
	if err != nil {
		return err
	}
	defer r.Close()
	if err := os.MkdirAll(path.Dir(localFile), os.ModePerm); err != nil {
		return err
	}
	w, err := os.Create(localFile)
	if err != nil {
		return err
	}
	defer w.Close()
	if _, err := io.Copy(w, r); err != nil {
		return err
	}
	return nil
}

func getRelPath(filepath string) string {
	u, err := url.Parse(filepath)
	if err != nil {
		return filepath
	}
	p := u.Path
	p = strings.TrimPrefix(p, `/`)
	return p
}
