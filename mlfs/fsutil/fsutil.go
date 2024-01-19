package fsutil

import (
	"errors"
	"io"
	"strconv"
	"strings"

	"github.com/kungfu-team/tenplex/mlfs/vfs"
)

var (
	errNodeNotExists = errors.New("node not exist")
	errNotFile       = errors.New("not a file")
)

func ReadTextLines(r *vfs.Tree, p string) ([]string, error) {
	n, _, ok := r.Get(p)
	if !ok {
		return nil, errNodeNotExists
	}
	if n.IsDir() {
		return nil, errNotFile
	}
	bs, err := io.ReadAll(n.AsFile().Open())
	if err != nil {
		return nil, err
	}
	var lines []string
	for _, line := range strings.Split(string(bs), "\n") {
		line = strings.TrimSpace(line)
		if len(line) > 0 {
			lines = append(lines, line)
		}
	}
	return lines, nil
}

func ReadIntLines(r *vfs.Tree, p string) ([]int, error) {
	n, _, ok := r.Get(p)
	if !ok {
		return nil, errNodeNotExists
	}
	if n.IsDir() {
		return nil, errNotFile
	}
	bs, err := io.ReadAll(n.AsFile().Open())
	if err != nil {
		return nil, err
	}
	var xs []int
	for _, line := range strings.Split(string(bs), "\n") {
		line = strings.TrimSpace(line)
		if len(line) > 0 {
			x, err := strconv.Atoi(line)
			if err != nil {
				return xs, err
			}
			xs = append(xs, x)
		}
	}
	return xs, nil
}
