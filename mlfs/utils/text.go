package utils

import (
	"io"
	"strings"

	"github.com/kungfu-team/tenplex/mlfs/uri"
)

func Readlines(filename string) ([]string, error) {
	f, err := uri.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	bs, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	var ls []string
	for _, l := range strings.Split(string(bs), "\n") {
		l = strings.TrimSpace(l)
		if len(l) > 0 {
			ls = append(ls, l)
		}
	}
	return ls, nil
}
