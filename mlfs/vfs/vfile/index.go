package vfile

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/kungfu-team/mlfs/uri"
)

func SaveIdxFile(filename string, idx IndexedFiles) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	bw := bufio.NewWriter(f)
	defer bw.Flush()
	return SaveIdx(bw, idx)
}

func SaveIdx(f io.Writer, idx IndexedFiles) error {
	fmt.Fprintf(f, "%d\n", len(idx))
	var n int
	for _, i := range idx {
		fmt.Fprintf(f, "%s %d\n", i.Filepath, len(i.Ranges))
		for _, r := range i.Ranges {
			fmt.Fprintf(f, "%d %d\n", r.Begin, r.End)
		}
		n += len(i.Ranges)
	}
	log.Printf("%d files, %d Records", len(idx), n)
	return nil
}

func LoadIdxFile(filename string) (IndexedFiles, error) {
	t0 := time.Now()
	defer func() { log.Printf("%s took %s", "LoadIdxFile", time.Since(t0)) }()
	f, err := uri.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReader(f)
	return LoadIdx(br)
}

func LoadIdx(f io.Reader) (IndexedFiles, error) {
	idx, err := readIdx(f)
	if err != nil {
		return nil, fmt.Errorf("readIdx: %v", err)
	}
	return idx, nil
}

func readIdx(f io.Reader) (IndexedFiles, error) {
	var idx IndexedFiles
	var n int
	if _, err := fmt.Fscanf(f, "%d\n", &n); err != nil {
		return nil, err
	}
	for i := 0; i < n; i++ {
		var name string
		var m int
		if _, err := fmt.Fscanf(f, "%s %d\n", &name, &m); err != nil {
			return nil, err
		}
		var rs Ranges
		for j := 0; j < m; j++ {
			var r Range
			if _, err := fmt.Fscanf(f, "%d%d\n", &r.Begin, &r.End); err != nil {
				return nil, err
			}
			rs = append(rs, r)
		}
		idx = append(idx, IndexedFile{name, rs})
	}
	return idx, nil
}
