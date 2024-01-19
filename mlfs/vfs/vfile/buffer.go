package vfile

import (
	"bytes"
	"io"
)

type Buffer struct {
	bs []byte
}

func NewBuffer() *Buffer {
	return &Buffer{}
}

func (f *Buffer) Open() io.ReadCloser {
	r := bytes.NewBuffer(f.bs)
	return io.NopCloser(r)
}

func (f *Buffer) Size() int64 {
	return int64(len(f.bs))
}

func (f *Buffer) Truncate() {
	f.bs = nil
}

func (f *Buffer) ReadAt(buf []byte, pos int64) (int, error) {
	br := bytes.NewBuffer(f.bs[pos:])
	return br.Read(buf)
}

func (f *Buffer) WriteAt(buf []byte, pos int64) (int, error) {
	if n := len(buf) + int(pos); n > len(f.bs) {
		f.bs = append(f.bs, make([]byte, n-len(f.bs))...)
	}
	return copy(f.bs[pos:], buf), nil
}
