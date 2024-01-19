// model parallelism
package mp

import (
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Synchronizer struct {
	Port          int
	Rank          int
	WorkerPrefix  string
	GatherPrefix  string
	ScatterPrefix string
}

func (s Synchronizer) RunGather(rank int, hosts []string) {
	srv := s.newServer(s.WorkerPrefix)
	srv.Start()
	log.Printf("server started, addr=%s", srv.hs.Addr)
	if rank == 0 {
		if err := s.runRoot(hosts); err != nil {
			log.Panic(err)
		}
		log.Printf("all replicas gathered")
	}
	srv.Wait()
	log.Printf("server stopped, addr=%s", srv.hs.Addr)
}

func (s Synchronizer) RunScatter(rank int, hosts []string) {
	log.Printf("TODO: RunScatter")
}

func (s Synchronizer) syncWorker(i int, h string) error {
	c := client{
		addr: net.JoinHostPort(h, str(s.Port+i)),
	}
	poll(delayErr(200*time.Millisecond, c.ping))
	log.Printf("%s is up", h)
	defer c.shut()

	fs, err := c.list()
	if err != nil {
		return err
	}
	log.Printf("will sync %d files from %s", len(fs), c.addr)

	for _, f := range fs {
		log.Printf("[sync] %s:%s", c.addr, f)
		r, n, err := c.open(f)
		if err != nil {
			return err
		}
		defer r.Close()
		log.Printf("%d bytes to read", n)
		withOpenWrite(path.Join(s.GatherPrefix, f), func(w io.Writer) error {
			_, err := io.Copy(w, r)
			return err
		})
	}
	return nil
}

func (s Synchronizer) runRoot(hosts []string) error {
	errs := make([]error, len(hosts))
	var wg sync.WaitGroup
	for i, h := range hosts {
		wg.Add(1)
		go func(i int, h string) {
			defer wg.Done()
			log.Printf("running syncWorker [%d]=%s", i, h)
			if err := s.syncWorker(i, h); err != nil {
				log.Printf("sync worker [%d]=%s failed: %v", i, h, err)
				errs[i] = err
			}
			log.Printf("syncWorker [%d]=%s done", i, h)
		}(i, h)
	}
	wg.Wait()
	return mergeErr(errs)
}

func splitLines(s string) []string {
	var ss []string
	for _, s := range strings.Split(s, "\n") {
		s = strings.TrimSpace(s)
		if len(s) > 0 {
			ss = append(ss, s)
		}
	}
	return ss
}

var str = strconv.Itoa

type client struct {
	hc   http.Client
	addr string
}

func (c *client) shut() {
	u := url.URL{Scheme: `http`, Host: c.addr, Path: `/_shut`}
	resp, err := c.hc.Get(u.String())
	if err != nil {
		return
	}
	resp.Body.Close()
}

func (c *client) ping() error {
	u := url.URL{Scheme: `http`, Host: c.addr, Path: `/_ping`}
	resp, err := c.hc.Get(u.String())
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func (c *client) list() ([]string, error) {
	u := url.URL{Scheme: `http`, Host: c.addr, Path: `/_list`}
	resp, err := c.hc.Get(u.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	bs, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return splitLines(string(bs)), nil
}

func (c *client) open(f string) (io.ReadCloser, int, error) {
	u := url.URL{Scheme: `http`, Host: c.addr, Path: `/files` + f}
	log.Printf("open %s on %s with %s", f, c.addr, u.Path)
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("get: %s", resp.Status)
	}
	var n int
	if ct := resp.Header.Get("Content-Length"); len(ct) != 0 {
		var err error
		n, err = strconv.Atoi(ct)
		if err != nil {
			log.Printf("invalid Content-Length")
		}
	}
	return resp.Body, n, nil
}

type server struct {
	prefix string
	mux    http.ServeMux
	fs     http.Handler
	hs     http.Server
	wg     sync.WaitGroup
}

func (s Synchronizer) newServer(prefix string) *server {
	srv := &server{
		prefix: prefix,
		fs:     http.FileServer(http.Dir(prefix)),
	}
	srv.mux.HandleFunc(`/_list`, srv.list)
	srv.mux.HandleFunc(`/_ping`, srv.ping)
	srv.mux.HandleFunc(`/_shut`, srv.shut)
	srv.mux.HandleFunc(`/files/`, srv.files)
	srv.hs = http.Server{
		Addr:    net.JoinHostPort("", str(s.Port+s.Rank)),
		Handler: &srv.mux,
	}
	log.Printf("new fs with prefix: %s", prefix)
	return srv
}

func (s *server) Start() {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		if err := s.hs.ListenAndServe(); err != nil {
			if err != http.ErrServerClosed {
				log.Printf("unexpected server stop: %v", err)
			}
		}
	}()
}

func (s *server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	s.mux.ServeHTTP(w, req)
}

func (s *server) ping(w http.ResponseWriter, req *http.Request) {}

func (s *server) list(w http.ResponseWriter, req *http.Request) {
	fs, err := filepath.Glob(s.prefix + `/*`)
	if err != nil {
		return
	}
	for _, f := range fs {
		f = strings.TrimPrefix(f, s.prefix)
		fmt.Fprintf(w, "%s\n", f)
	}
}

func (s *server) shut(w http.ResponseWriter, req *http.Request) { s.hs.Close() }

func (s *server) files(w http.ResponseWriter, req *http.Request) {
	log.Printf("req files: %s", req.URL.Path)
	req.URL.Path = strings.TrimPrefix(req.URL.Path, `/files`)
	s.fs.ServeHTTP(w, req)
}

func (s *server) Wait() { s.wg.Wait() }

func poll(f func() error) {
	for {
		if err := f(); err == nil {
			break
		}
	}
}

func delayErr(d time.Duration, f func() error) func() error {
	return func() error {
		if err := f(); err != nil {
			time.Sleep(d)
			return err
		}
		return nil
	}
}

func mergeErr(errs []error) error {
	var msgs []string
	for _, e := range errs {
		if e != nil {
			msgs = append(msgs, e.Error())
		}
	}
	if len(msgs) > 0 {
		return fmt.Errorf("%d failed: %s", len(msgs), strings.Join(msgs, ","))
	}
	return nil
}

func withOpenWrite(name string, g func(io.Writer) error) error {
	if err := os.MkdirAll(path.Dir(name), os.ModePerm); err != nil {
		return err
	}
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	defer f.Close()
	return g(f)
}
