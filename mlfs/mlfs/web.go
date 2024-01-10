package mlfs

import (
	"archive/tar"
	"fmt"
	"image/png"
	"io"
	golog "log"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/kungfu-team/mlfs/buildinfo"
	"github.com/kungfu-team/mlfs/fsutil"
	"github.com/kungfu-team/mlfs/pid"
	"github.com/kungfu-team/mlfs/uri"
	"github.com/kungfu-team/mlfs/vfs"
	"github.com/kungfu-team/tensor"
)

type webUI struct {
	e       *MLFS
	handler http.Handler
}

func newWebUI(e *MLFS, doLogging bool) *webUI {
	s := &webUI{e: e}
	var mux http.ServeMux
	mux.HandleFunc(`/`, s.index)
	mux.HandleFunc(`/bmp`, s.serveBitMap)
	mux.Handle(`/js/`, http.FileServer(http.Dir(`./www`)))
	mux.HandleFunc(`/jsonrpc/`, s.serveJSONRPC)
	mux.HandleFunc(`/upload`, s.replicated(s.upload))
	mux.HandleFunc(`/delete`, s.replicated(s.delete))
	mux.HandleFunc(`/upload1`, s.upload)
	mux.HandleFunc(`/delete1`, s.delete)
	mux.HandleFunc(`/query`, s.query)
	mux.HandleFunc(`/debug`, s.debug)
	mux.HandleFunc(`/peers`, s.peers)
	if doLogging {
		s.handler = withLogReq(&mux)
	} else {
		s.handler = &mux
	}
	return s
}

func (s *webUI) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	s.handler.ServeHTTP(w, req)
}

func (s *webUI) serveJSONRPC(w http.ResponseWriter, req *http.Request) {
	s.e.proxy.serveJSONRPC(w, req)
}

func (s *webUI) debug(w http.ResponseWriter, req *http.Request) {
	var hostname, _ = os.Hostname()
	fmt.Fprintf(w, "hostname: %s\n", hostname)
	uri.Debug(w)
	buildinfo.Default.Show(w)
}

func (s *webUI) index(w http.ResponseWriter, req *http.Request) {
	fmt.Fprintf(w, `<a href="./bmp" target="_blank">bmp</a>`)
	fmt.Fprintf(w, `<p>%d indexes</p>`, len(s.e.indexes))
	for name, dsidx := range s.e.indexes {
		fmt.Fprintf(w, `<p>%s: %d files</p>`, name, len(dsidx.idx))
	}
	fmt.Fprintf(w, `<script src="./js/bmp.js"></script>`)
}

func (s *webUI) upload(w http.ResponseWriter, req *http.Request) {
	switch ct := req.Header.Get(`Content-Type`); ct {
	case `x-tensor`:
		s.uploadTensor(w, req)
	case `application/x-tar`:
		s.uploadTar(w, req)
	default:
		s.uploadFile(w, req)
	}
}

func (s *webUI) uploadTar(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		log.Printf("Error uploadTar http.StatusMethodNotAllowed")
		http.Error(w, "ERROR uploadTar", http.StatusMethodNotAllowed)
		return
	}
	tr := tar.NewReader(req.Body)
	prefix := req.FormValue(`prefix`)
	for {
		f, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("ERROR uploadTar http.StatusBadRequest")
			http.Error(w, "ERROR uploadTar", http.StatusBadRequest)
			return
		}
		bs, err := io.ReadAll(tr)
		if err != nil {
			log.Printf("ERROR uploadTar http.StatusBadRequest")
			http.Error(w, "ERROR uploadTar", http.StatusBadRequest)
			return
		}
		filepath := path.Join(prefix, f.Name)
		if f.FileInfo().IsDir() {
			s.e.tree.MkdirAll(filepath)
			continue
		}
		s.e.tree.MkdirAll(path.Dir(filepath))
		if _, err := s.e.tree.TouchFile(filepath, vfs.ToFile(bs)); err != nil {
			log.Printf("ERROR uploadTar %s %v", filepath, err)
			http.Error(w, fmt.Sprintf("ERROR uploadTar %s %v", filepath, err), http.StatusInternalServerError)
		}
	}
}

func (s *webUI) uploadFile(w http.ResponseWriter, req *http.Request) {
	p := req.FormValue(`path`)
	dir := path.Dir(p)
	log.Printf("dir %s", dir)
	err := s.e.tree.MkdirAll(path.Dir(p))
	if err != nil {
		log.Printf("ERROR uploadFile MkdirAll %s %v", p, err)
		return
	} else {
		log.Printf("%q MkdirAll WORKED", p)
	}

	bs, err := io.ReadAll(req.Body)
	if err != nil {
		log.Printf("ERROR uploadFile ReadAll %s %v", p, err)
		http.Error(w, ``, http.StatusBadRequest)
		return
	}

	if req.Header.Get("x-replace") == "true" {
		if _, err := s.e.tree.TouchOrReplaceBytes(p, bs); err != nil {
			log.Printf("ERROR uploadFile TouchOrReplaceBytes data: %q", p)
		}
	} else {
		if _, err := s.e.tree.TouchFile(p, vfs.ToFile(bs)); err != nil {
			log.Printf("ERROR uploadFile TouchFile %s %v", p, err)
		}
	}
}

func (s *webUI) delete(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodDelete {
		http.Error(w, `method must be `+http.MethodDelete, http.StatusMethodNotAllowed)
		return
	}
	p := req.FormValue(`path`)
	log.Printf("delete: %s %s path:%s", req.Method, req.URL, p)
	nf, nd, err := vfs.RmRecursive(s.e.tree, p)
	if err != nil {
		log.Printf("Error in delete %v", err)
		http.Error(w, err.Error(), http.StatusExpectationFailed)
		return
	}
	fmt.Fprintf(w, "%d %d\n", nf, nd)
}

func (s *webUI) peers(w http.ResponseWriter, req *http.Request) {
	switch req.Method {
	case http.MethodGet:
		s.getPeers(w, req)
	case http.MethodPost:
		s.postPeers(w, req)
	default:
		http.Error(w, ``, http.StatusMethodNotAllowed)
	}
}

func (s *webUI) getPeers(w http.ResponseWriter, req *http.Request) {
	if s.e.redundency == 0 {
		fmt.Fprintf(w, "replication not enabled!")
	}
	// fmt.Fprintf(w, "%s", ipv4.FormatIPv4List(s.e.GetPeers()))
	fmt.Fprintf(w, "%s", s.e.GetPeers().String())
}

func (s *webUI) postPeers(w http.ResponseWriter, req *http.Request) {
	bs, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, ``, http.StatusBadRequest)
		return
	}
	ids, err := pid.ParsePeerList(strings.TrimSpace(string(bs)))
	if err != nil {
		http.Error(w, ``, http.StatusBadRequest)
		return
	}
	s.e.peers = ids
}

func parseDims(s string) []int {
	var dims []int
	parts := strings.Split(s, `,`)
	for _, p := range parts {
		d, err := strconv.Atoi(p)
		if err == nil {
			dims = append(dims, d)
		}
	}
	return dims
}

var dtMap = map[string]string{
	`int8`:  `i8`,
	`int16`: `i16`,
	`int32`: `i32`,
	`int64`: `i64`,
	//
	`uint8`:  `u8`,
	`uint16`: `u16`,
	`uint32`: `u32`,
	`uint64`: `u64`,
	//
	`float16`: `f16`,
	`float32`: `f32`,
	`float64`: `f64`,
}

func parseDType(dt string) string {
	if t, ok := dtMap[dt]; ok {
		return t
	}
	return dt
}

func (s *webUI) uploadTensor(w http.ResponseWriter, req *http.Request) {
	dt := parseDType(req.FormValue(`dtype`))
	dims := parseDims(req.FormValue(`dims`))
	p := req.FormValue(`path`)
	t := tensor.New(dt, dims...)
	_, err := io.ReadFull(req.Body, t.Data)
	if err != nil {
		log.Printf("ERROR uploadTensor ReadFull %s %v", p, err)
		http.Error(w, fmt.Sprintf("ERROR uploadTensor ReadFull %s %v", p, err), http.StatusInternalServerError)
		return
	}
	err = s.e.tree.MkdirAll(path.Dir(p))
	if err != nil {
		log.Printf("ERROR uploadTensor MkdirAll %s %v", p, err)
		http.Error(w, fmt.Sprintf("ERROR uploadTensor MkdirAll %s %v", p, err), http.StatusInternalServerError)
		return
	}
	if err := s.e.TouchTensor(p, t); err != nil {
		log.Printf("ERROR uploadTensor TouchTensor %s %v", p, err)
		http.Error(w, fmt.Sprintf("ERROR uploadTensor TouchTensor %s %v", p, err), http.StatusInternalServerError)
		return
	}
}

func parseRange(rg string) *tensor.Range {
	var ss tensor.Range
	if rg == "" {
		return nil
	}
	for _, p := range strings.Split(rg, `,`) {
		parts := strings.Split(p, `:`)
		a := -1
		if len(parts[0]) > 0 {
			a = mustParseInt(parts[0])
		}
		b := -1
		if len(parts[1]) > 0 {
			b = mustParseInt(parts[1])
		}
		ss = append(ss, tensor.Slice(a, b))
	}
	return &ss
}

func parseInt(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

func mustParseInt(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return n
}

func parseMeta(s string) (string, []int) {
	parts := strings.Split(s, "\n")
	dtype := parts[0]
	rank := mustParseInt(parts[1])
	var dims []int
	for _, p := range parts[2 : 2+rank] {
		dims = append(dims, mustParseInt(p))
	}
	return dtype, dims
}

func (s *webUI) query(w http.ResponseWriter, req *http.Request) {
	switch ct := req.Header.Get(`Content-Type`); ct {
	case `x-tensor`:
		s.queryTensor(w, req)
	case `x-dir`:
		s.queryDir(w, req)
	default:
		s.queryFile(w, req)
	}
}

func (s *webUI) queryTensor(w http.ResponseWriter, req *http.Request) {
	p := req.FormValue(`path`)
	r := parseRange(req.FormValue(`range`))
	bs, err := vfs.ReadFile(s.e.tree, p+`.meta`)
	if err != nil {
		log.Printf("ERROR queryTensor ReadFile %s, %v", p+`.meta`, err)
		http.NotFound(w, req)
		return
	}
	dtype, dims := parseMeta(string(bs))
	data, err := vfs.ReadFile(s.e.tree, p)
	if err != nil {
		log.Printf("ERROR queryTensor ReadFile %s %v", p, err)
		http.Error(w, fmt.Sprintf("ERROR queryTensor ReadFile %s %v", p, err), http.StatusInternalServerError)
		return
	}
	x := tensor.NewWith(dtype, dims, data)
	if r != nil {
		x = x.Range(*r...)
	}
	w.Header().Set(`x-tensor-dtype`, x.Dtype)
	w.Header().Set(`x-tensor-dims`, join(`,`, fmap(str[int], x.Dims)...))
	w.Write(x.Data)
}

func (s *webUI) queryFile(w http.ResponseWriter, req *http.Request) {
	p := req.FormValue(`path`)
	nodes := strings.Split(p, "/")
	filename := nodes[len(nodes)-1]
	dirPath := strings.Join(nodes[0:len(nodes)-1], "/")
	names, err := vfs.ReadDir(s.e.tree, dirPath)
	if err != nil {
		log.Printf("ERROR queryFile ReadDir %s %v", p, err)
		http.Error(w, fmt.Sprintf("ERROR queryFile ReadDir %s %v", p, err), http.StatusInternalServerError)
		return
	}
	for _, n := range names {
		splitName := strings.SplitN(n, ".", 2)
		nFilename := splitName[0]
		if nFilename == filename {
			bs, err := vfs.ReadFile(s.e.tree, path.Join(dirPath, n))
			if err != nil {
				log.Printf("ERROR queryFile ReadFile %s %v", p, err)
				http.Error(w, fmt.Sprintf("ERROR queryFile ReadFile %s %v", p, err), http.StatusInternalServerError)
				return
			}
			var dtype string
			if len(splitName) == 1 {
				dtype = "text"
			} else {
				dtype = splitName[1]
			}
			w.Header().Set(`dtype`, dtype)
			w.WriteHeader(http.StatusOK)
			w.Write(bs)
			return
		}
	}
	log.Printf("ERROR queryFile cannot match path %s", p)
	http.Error(w, fmt.Sprintf("ERROR queryFile cannot match path %s", p), http.StatusInternalServerError)
}

func createFilesList(s *webUI, p string) (string, error) {
	node, id, ok := s.e.tree.Get(p)
	if !ok {
		return "", fmt.Errorf("createFilesList cannot get node %s", p)
	}
	if !node.IsDir() {
		return s.e.tree.FullPath(id), nil
	} else {
		list := ""
		for _, it := range node.AsDir().Items() {
			pa := path.Join(p, it.Name)
			newItems, err := createFilesList(s, pa)
			if err != nil {
				return "", err
			}
			if list == "" {
				list = newItems
			} else {
				list = list + "\n" + newItems
			}
		}
		return list, nil
	}
}

func (s *webUI) queryDir(w http.ResponseWriter, req *http.Request) {
	p := req.FormValue(`path`)
	fileList, err := createFilesList(s, p)
	if err != nil {
		http.Error(w, fmt.Sprintf("%v", err), http.StatusNotFound)
		return
	}
	w.Header().Set(`dtype`, `txt`)
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(fileList))
}

func join(s string, ss ...string) string { return strings.Join(ss, s) }

func fmap[X any, Y any](f func(x X) Y, xs []X) []Y {
	var ys []Y
	for _, x := range xs {
		ys = append(ys, f(x))
	}
	return ys
}

func str[T any](i T) string { return fmt.Sprintf("%v", i) }

func (s *webUI) serveBitMap(w http.ResponseWriter, req *http.Request) {
	if !isBrowser(req.UserAgent()) {
		setFilename(w.Header(), `a.png`)
	}
	dsidx := s.e.indexes[`imagenet`]
	ids, err := fsutil.ReadIntLines(s.e.tree, `/progress-0/cluster-of-4/rank-003/0027.tf_record.meta`)
	if err != nil {
		return
	}
	log.Printf("%d regions amoug %d", len(ids), dsidx.totalRegions)
	img := dsidx.bmap(ids)
	// width := 1024
	// height := 128
	// img := makeBitmap(height, width)
	if err := png.Encode(w, img); err != nil {
		log.Printf("png.Encode: %v", err)
	}
}

func setFilename(h http.Header, filename string) {
	h.Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, filename))
}

type logger struct {
	l *golog.Logger
}

func (l *logger) Printf(format string, v ...interface{}) {
	l.l.Output(2, fmt.Sprintf(format, v...))
}

var accessLog = logger{l: golog.New(os.Stderr, "", golog.LstdFlags)}

func logRequest(prefix string, req *http.Request) {
	accessLog.Printf("%s%s %s | %s", prefix, req.Method, req.URL, req.RemoteAddr)
}

func logRequestDetail(prefix string, req *http.Request) {
	accessLog.Printf("%s%s %s | %s", prefix, req.Method, req.URL, req.UserAgent())
	for k, vs := range req.Header {
		for _, v := range vs {
			log.Printf("%s: %s", k, v)
		}
	}
}

func isBrowser(ua string) bool {
	return strings.Contains(ua, `Mozilla`)
}

func withLogReq(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// logRequest(``, req)
		logRequestDetail(``, req)
		h.ServeHTTP(w, req)
	})
}
