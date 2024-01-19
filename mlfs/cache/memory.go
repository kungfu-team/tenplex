package cache

import (
	"bytes"
	"io"
	"sync"
	"sync/atomic"

	"github.com/kungfu-team/tenplex/mlfs/vfs"
)

type memcached struct {
	f vfs.FileNode

	cached int32
	bs     []byte
	mu     sync.Mutex
}

func Memcache(f vfs.FileNode) *memcached {
	return &memcached{f: f}
}

func (f *memcached) isCached() bool {
	return atomic.LoadInt32(&f.cached) > 0
}

func (f *memcached) Size() int64 { return f.f.Size() }

func (f *memcached) Open() io.ReadCloser {
	if f.isCached() {
		return io.NopCloser(bytes.NewReader(f.bs))
	}
	return f.f.Open()
}

func (f *memcached) ReadAt(buf []byte, pos int64) (int, error) {
	if f.isCached() {
		br := bytes.NewReader(f.bs)
		return br.ReadAt(buf, pos)
	}
	return f.f.ReadAt(buf, pos)
}

func (f *memcached) Cache() {
	if f.isCached() {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	r := f.f.Open()
	bs, err := io.ReadAll(r)
	r.Close()
	if err == nil {
		f.bs = bs
		atomic.StoreInt32(&f.cached, 1)
	}
}

func (f *memcached) Uncache() {
	atomic.StoreInt32(&f.cached, 0)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bs = nil
}
