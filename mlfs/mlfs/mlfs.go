package mlfs

import (
	"bytes"
	"errors"
	"fmt"
	golog "log"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"path"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/buildinfo"
	"github.com/kungfu-team/tenplex/mlfs/cache"
	"github.com/kungfu-team/tenplex/mlfs/ds/trds"
	"github.com/kungfu-team/tenplex/mlfs/pid"
	"github.com/kungfu-team/tenplex/mlfs/uri"
	"github.com/kungfu-team/tenplex/mlfs/utils"
	"github.com/kungfu-team/tenplex/mlfs/vfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/hfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/ufs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

var log = golog.New(os.Stderr, fmt.Sprintf(`[mlfs:%d] `, os.Getpid()), 0)

type (
	Peer     = pid.PeerID
	PeerList = pid.PeerList
)

type MLFS struct {
	t0         time.Time
	proxy      *Proxy
	tree       *vfs.Tree
	redundency int // number of extra replicas to write, (i, i + 1, ..., i + r mod n)
	peers      PeerList
	rank       int
	indexes    map[string]*DSIDX
	cache      *cache.Cache
	paused     int32
	stop       []func()

	mu     sync.RWMutex
	root   string
	LogReq bool
}

func New() *MLFS {
	t0 := time.Now()
	tree := vfs.New()
	tree.Mkdir(`/`)
	tree.TouchText(`/t0.txt`, fmt.Sprintf("%d\n", t0.Unix()))
	e := &MLFS{
		t0:      t0,
		tree:    tree,
		indexes: make(map[string]*DSIDX),
	}
	e.proxy = &Proxy{
		AddIdxFn:        e.AddIndex,
		MountFn:         e.Mount,
		FetchFn:         e.Fetch,
		GetRootFn:       e.GetRoot,
		FetchAllFn:      e.FetchAll,
		FetchPartFn:     e.FetchPart,
		PauseFn:         e.Pause,
		SetSASFn:        e.SetSAS,
		SetPeersFn:      e.SetPeers,
		SetRedundencyFn: e.SetRedundency,
	}
	return e
}

var errNotDir = errors.New("not a dir")

func (e *MLFS) Tree() *vfs.Tree { return e.tree }

func (e *MLFS) SetCache(root string) error {
	info, err := os.Stat(root)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return errNotDir
	}
	e.cache = cache.New(root)
	return nil
}

func (e *MLFS) Pause() error {
	atomic.StoreInt32(&e.paused, 1)
	return nil
}

func (e *MLFS) AddIndex(req *AddIdxRequest) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	idx, err := vfile.LoadIdxFile(req.File)
	if err != nil {
		return err
	}
	e.indexes[req.Name] = newDSIDX(idx)
	prefix := path.Join(`/indexes`, req.Name)
	e.tree.MkdirAll(prefix)
	if _, err = e.tree.TouchOrReplaceText(path.Join(prefix, `idx.txt`), func() string {
		bs := &bytes.Buffer{}
		vfile.SaveIdx(bs, idx)
		return bs.String()
	}()); err != nil {
		return err
	}
	if _, err = e.tree.TouchOrReplaceText(path.Join(prefix, `files.txt`), func() string {
		bs := &bytes.Buffer{}
		for _, f := range idx {
			fmt.Fprintf(bs, "%s\n", f.Filepath)
		}
		return bs.String()
	}()); err != nil {
		return err
	}
	e.tree.Stat()
	return nil
}

func (e *MLFS) GetPeers() PeerList {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.peers[:]
}

func (e *MLFS) SetPeers(req *SetPeerRequest) error {
	if e.redundency >= len(req.Peers) {
		return errInvalidRedundancy
	}
	e.peers = req.Peers
	return nil
}

func (e *MLFS) SetRedundency(req *SetRedundencyRequest) error {
	if req.Redundency >= len(e.peers) {
		return errInvalidRedundancy
	}
	e.redundency = req.Redundency
	return nil
}

func (e *MLFS) GetRoot(resp *string) error {
	*resp = e.root
	return nil
}

var errCacheNotEnabled = errors.New(`cache not enabled`)

func (e *MLFS) Fetch(req *FetchRequest) error {
	if e.cache == nil {
		return errCacheNotEnabled
	}
	if req.Async {
		go e.cache.Fetch(req.URL, req.MD5)
		return nil
	}
	_, err := e.cache.Fetch(req.URL, req.MD5)
	return err
}

func (e *MLFS) FetchAll(req *FetchAllRequest) error {
	if e.cache == nil {
		return errCacheNotEnabled
	}
	idx := e.indexes[req.Name]
	if idx == nil {
		return fmt.Errorf("%v: %s", errNotFound, req.Name)
	}
	if req.Async {
		go e.fetchAll(idx.idx)
		return nil
	}
	return e.fetchAll(idx.idx)
}

func (e *MLFS) fetchAll(fs vfile.IndexedFiles) error {
	for i, f := range fs {
		log.Printf("fetchAll() %d/%d ...", i, len(fs))
		if atomic.LoadInt32(&e.paused) > 0 {
			log.Printf("fetchAll() paused")
			break
		}
		if _, err := e.cache.Fetch(f.Filepath, ``); err != nil {
			return err
		}
	}
	return nil
}

func (e *MLFS) FetchPart(req *FetchPartRequest) error {
	if e.cache == nil {
		return errCacheNotEnabled
	}
	idx := e.indexes[req.Name]
	if idx == nil {
		return fmt.Errorf("%v: %s", errNotFound, req.Name)
	}
	if req.ShardInfo.Size <= 0 {
		return fmt.Errorf("%v: %#v", errInvalidArgument, req)
	}
	if req.ShardInfo.Rank < 0 || req.ShardInfo.Rank >= req.ShardInfo.Size {
		return fmt.Errorf("%v: %#v", errInvalidArgument, req)
	}
	vf := idx.idx.Shard(req.ShardInfo.Rank, req.ShardInfo.Size)
	log.Printf("%d/%d: %d ranges, size: %s", req.ShardInfo.Rank, req.ShardInfo.Size, vf.NumRanges(), utils.ShowSize(vf.Size()))
	log.Printf("TODO: FetchPart")
	return nil
}

var errNotFound = errors.New("not found")

func (e *MLFS) Mount(req *MountRequest) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	dsidx, ok := e.indexes[req.Name]
	if !ok {
		return errNotFound
	}
	ds := trds.New(dsidx.idx, req.Seed)
	ds.SetCache(e.cache)
	if err := ds.Mount(e.tree, req.JobID, req.Progress, req.GlobalBatchSize, req.ClusterSize, req.MinSamplesPerFile); err != nil {
		return err
	}
	e.tree.Stat()
	return nil
}

func (e *MLFS) SetSAS(req *SetSASRequest) error {
	uri.SetSAS(req.SA, req.SAS)
	return nil
}

func (e *MLFS) RunCtrl(port int) error {
	s := rpc.NewServer()
	if err := s.Register(e.proxy); err != nil {
		return err
	}
	mux := &http.ServeMux{}
	mux.Handle(`/`, newWebUI(e, e.LogReq))
	mux.Handle(`/buildinfo`, &buildinfo.Default)
	mux.Handle(rpc.DefaultRPCPath, s)
	log.Printf("enabling control API: http://127.0.0.1:%d/", port)
	addr := net.JoinHostPort(``, strconv.Itoa(port))
	hs := http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}
	e.stop = append(e.stop, func() { hs.Close() })
	return hs.ListenAndServe()
}

func (e *MLFS) RunHTTP(port int) error {
	s := hfs.HS(e.tree)
	log.Printf("enabling http endpoint: http://127.0.0.1:%d/", port)
	addr := net.JoinHostPort(``, strconv.Itoa(port))
	hs := http.Server{
		Addr:         addr,
		Handler:      s,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}
	e.stop = append(e.stop, func() { hs.Close() })
	return hs.ListenAndServe()
}

func (e *MLFS) RunFuse(mnt string, super bool) {
	ufs.Umount(mnt)
	e.stop = append(e.stop, func() { ufs.Umount(mnt) })
	if _, err := os.Stat(mnt); err != nil {
		if err := os.MkdirAll(mnt, os.ModePerm); err != nil {
			utils.ExitErr(err)
		}
	}
	log.Printf("enabling fuse mount: file://%s", mnt)
	e.root = mnt
	ufs.Start(mnt, e.tree, super)
}

func (e *MLFS) Stop() {
	t0 := time.Now()
	for _, f := range e.stop {
		f()
	}
	log.Printf("MLFS Stop took %s", time.Since(t0))
}
