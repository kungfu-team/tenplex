package vfs

import (
	"bytes"
	"io"
)

type file struct {
	bs []byte
}

func ToFile(bs []byte) *file { return &file{bs: bs} }

func (f *file) IsDir() bool { return false }

func (f *file) IsExecutable() bool { return bytes.HasPrefix(f.bs, []byte(`#!`)) }

func (f *file) AsFile() FileNode { return f }

func (f *file) AsDir() DirNode { return nil }

func (f *file) Open() io.ReadCloser {
	r := bytes.NewBuffer(f.bs)
	return io.NopCloser(r)
}

func (f *file) Size() int64 {
	return int64(len(f.bs))
}

func (f *file) ReadAt(buf []byte, pos int64) (int, error) {
	n := min(len(buf), len(f.bs)-int(pos))
	copy(buf[:n], f.bs[pos:(pos)+int64(n)])
	return n, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
