package mlfs

import (
	"flag"
	"fmt"
	golog "log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"

	"github.com/kungfu-team/mlfs/uri"
	"github.com/kungfu-team/mlfs/vfs"
	"github.com/kungfu-team/mlfs/vfs/hfs"
	"github.com/kungfu-team/mlfs/vfs/vfile"
)

const DefaultServerPort = DefaultCtrlPort + 100

type LocalServer struct {
	IdxFile string `json:"index-url"`
	Port    int    `json:"port"`
	SelfIP  string `json:"self"`

	SAS string `json:"sas"` //
	log *golog.Logger
}

func (s *LocalServer) RegisterFlags(flag *flag.FlagSet) {
	flag.IntVar(&s.Port, `port`, DefaultServerPort, ``)
	flag.StringVar(&s.IdxFile, `index-url`, ``, ``)
	flag.StringVar(&s.SelfIP, `self`, ``, ``)
	flag.StringVar(&s.SAS, `sas`, ``, ``)
}

func (s *LocalServer) Run() error {
	s.log = golog.New(os.Stderr, `[mlfs] serve % `, 0)
	fs, err := vfile.LoadIdxFile(s.IdxFile)
	if err != nil {
		s.log.Panic(err)
	}
	host := net.JoinHostPort(s.SelfIP, str(s.Port))
	idx, err := fromLocal(fs, host)
	if err != nil {
		s.log.Panic(err)
	}
	srv := localHandler(idx)
	s.log.Printf("http://%s/", host)
	return http.ListenAndServe(host, &srv) // TODO: ignore shutdown error
}

type localHandler vfile.IndexedFiles

func (l *localHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	golog.Printf("%s %s @ %s | %s %s", req.Method, req.RequestURI, req.Header.Get(`Range`), req.UserAgent(), req.RemoteAddr)
	if req.URL.Path == `/` {
		vfile.SaveIdx(w, vfile.IndexedFiles(*l))
		return
	}
	http.ServeFile(w, req, req.URL.Path)
}

func fromLocal(fs vfile.IndexedFiles, host string) (vfile.IndexedFiles, error) {
	var idx vfile.IndexedFiles
	for _, f := range fs {
		u, err := url.Parse(f.Filepath)
		if err != nil {
			return nil, err
		}
		if u.Scheme != `` && u.Scheme != `file` {
			return nil, fmt.Errorf("index contains non-local file: %s", u)
		}
		idx = append(idx, vfile.IndexedFile{
			Filepath: (&url.URL{
				Scheme: `http`,
				Host:   host,
				Path:   u.Path,
			}).String(),
			Ranges: f.Ranges,
		})
	}
	return idx, nil
}

type RelayServer struct {
	LocalServer
}

func (s *RelayServer) Run() error {
	s.log = golog.New(os.Stderr, `[mlfs] serve % `, 0)
	if len(s.SAS) > 0 {
		if parts := strings.SplitN(s.SAS, `:`, 2); len(parts) == 2 {
			uri.SetSAS(parts[0], parts[1])
		} else {
			s.log.Panicf("invalid -sas: %q", s.SAS)
		}
	}
	fs, err := vfile.LoadIdxFile(s.IdxFile)
	if err != nil {
		s.log.Panic(err)
	}
	host := net.JoinHostPort(s.SelfIP, str(s.Port))
	idx, h, err := newRelay(fs, host)
	if err != nil {
		s.log.Panic(err)
	}
	s.log.Printf("http://%s/", host)
	return http.ListenAndServe(host, &relayHandler{
		idx: idx,
		h:   h,
	})
}

type relayHandler struct {
	idx vfile.IndexedFiles
	h   http.Handler
}

func (l *relayHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	golog.Printf("%s %s @ %s | %s %s", req.Method, req.RequestURI, req.Header.Get(`Range`), req.UserAgent(), req.RemoteAddr)
	if req.URL.Path == `/` {
		vfile.SaveIdx(w, l.idx)
		return
	}
	l.h.ServeHTTP(w, req)
}

func newRelay(fs vfile.IndexedFiles, host string) (vfile.IndexedFiles, http.Handler, error) {
	var idx vfile.IndexedFiles
	tree := vfs.New()
	for _, f := range fs {
		u, err := url.Parse(f.Filepath)
		if err != nil {
			return nil, nil, err
		}
		tree.MkdirAll(path.Dir(u.Path))
		if _, err := tree.TouchFile(u.Path,
			vfile.Link(f.Filepath, int64(f.IndexedBytes()))); err != nil {
			return nil, nil, err
		}
		idx = append(idx, vfile.IndexedFile{
			Filepath: (&url.URL{
				Scheme: `http`,
				Host:   host,
				Path:   u.Path,
			}).String(),
			Ranges: f.Ranges,
		})
	}

	return idx, hfs.HS(tree), nil
}
