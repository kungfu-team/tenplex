package hfs

import (
	"io"
	"io/fs"
	"net/http"
	"os"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/vfs"
)

type info struct {
	name    string
	size    int64
	mode    os.FileMode
	modtime time.Time
	isdir   bool
}

func (i *info) Name() string       { return i.name }
func (i *info) Size() int64        { return i.size }
func (i *info) Mode() os.FileMode  { return i.mode }
func (i *info) ModTime() time.Time { return i.modtime }
func (i *info) IsDir() bool        { return i.isdir }
func (i *info) Sys() interface{}   { return nil }

// httpFile implements http.File
type httpFile struct {
	name string
	size int64
	n    vfs.Node
	r    io.ReadSeeker
}

func (h *httpFile) Close() error { return nil }

func (h *httpFile) Read(buf []byte) (int, error) {
	return h.r.Read(buf)
}

func (h *httpFile) Readdir(n int) ([]os.FileInfo, error) {
	var items []os.FileInfo
	for _, i := range h.n.AsDir().Items() {
		items = append(items, &info{name: i.Name})
	}
	return items, nil
}

func (h *httpFile) Seek(offset int64, whence int) (int64, error) {
	return h.r.Seek(offset, whence)
}

func (h *httpFile) Stat() (os.FileInfo, error) {
	if h.n.IsDir() {
		return &info{
			name:  h.name,
			isdir: true,
		}, nil
	} else {
		return &info{
			name: h.name,
			size: h.size,
		}, nil
	}
}

// httpFS implements http.FileSystem
type httpFS struct {
	r *vfs.Tree
}

func (s *httpFS) Open(name string) (http.File, error) {
	n, _, ok := s.r.Get(name)
	if !ok {
		return nil, fs.ErrNotExist
	}
	p := vfs.ParseP(name)
	var basename string
	if len(p) > 0 {
		basename = p[len(p)-1]
	}
	var size int64
	var r io.ReadSeeker
	if f := n.AsFile(); f != nil {
		size = f.Size()
		r = io.NewSectionReader(f, 0, size)
	}
	h := &httpFile{
		size: size,
		name: basename,
		n:    n,
		r:    r,
	}
	return h, nil
}

type server struct {
	fileServer http.Handler
}

func HS(r *vfs.Tree) *server {
	fs := &httpFS{r: r}
	s := &server{
		fileServer: http.FileServer(fs),
	}
	return s
}

func (s *server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	s.fileServer.ServeHTTP(w, req)
}
