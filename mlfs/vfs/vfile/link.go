package vfile

import (
	"io"

	"github.com/kungfu-team/tenplex/mlfs/uri"
)

type link struct {
	string
	int64
}

func Link(url string, size int64) link { return link{url, size} }

func (f link) Size() int64 { return f.int64 }

func (f link) Open() io.ReadCloser {
	r := io.NewSectionReader(f, 0, f.Size())
	return io.NopCloser(r)
}

func (f link) ReadAt(buf []byte, pos int64) (int, error) {
	r, err := uri.OpenRange(f.string, pos, f.int64)
	if err != nil {
		return 0, err
	}
	defer r.Close()
	return r.Read(buf)
}
