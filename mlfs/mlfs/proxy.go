package mlfs

import (
	"encoding/json"
	"errors"
	"net/http"
)

type ShardInfo struct {
	Rank int `json:"rank"`
	Size int `json:"size"`
}

type AddIdxRequest struct {
	Name string `json:"name"`
	File string `json:"file"`
}

type MountRequest struct {
	JobID             string `json:"job-id"`
	Name              string `json:"name"`
	Progress          int64  `json:"progress"`
	GlobalBatchSize   int    `json:"global-batch-size"`
	ClusterSize       int    `json:"cluster-size"`
	MinSamplesPerFile int    `json:"min-samples-per-file"`
	Seed              int    `json:"seed"`
	NoShuffle         bool   `json:"no-shuffle"`
}

type FetchRequest struct {
	URL   string `json:"url"`
	MD5   string `json:"md5"`
	Async bool   `json:"async"`
}

type FetchAllRequest struct {
	Name  string `json:"name"`
	Async bool   `json:"async"`
}

type FetchPartRequest struct {
	Name      string    `json:"name"`
	ShardInfo ShardInfo `json:"shard-info"`
}

type SetSASRequest struct {
	SA  string `json:"sa"`
	SAS string `json:"sas"`
}

type SetPeerRequest struct {
	Peers PeerList `json:"peers"`
}

type SetRedundencyRequest struct {
	Redundency int `json:"redundency"`
}

type Void struct{}

type Proxy struct {
	AddIdxFn        func(req *AddIdxRequest) error
	MountFn         func(req *MountRequest) error
	FetchFn         func(req *FetchRequest) error
	FetchAllFn      func(req *FetchAllRequest) error
	FetchPartFn     func(req *FetchPartRequest) error
	GetRootFn       func(resp *string) error
	PauseFn         func() error
	SetSASFn        func(req *SetSASRequest) error
	SetPeersFn      func(req *SetPeerRequest) error
	SetRedundencyFn func(req *SetRedundencyRequest) error
}

var errInvalidArgument = errors.New("invalid argument")

func (o *Proxy) GetRoot(req *Void, resp *string) error {
	return logErr("GetRoot", o.GetRootFn(resp))
}

func (o *Proxy) AddIndex(req *AddIdxRequest, resp *Void) error {
	return logErr("AddIndex", o.AddIdxFn(req))
}

func (o *Proxy) Fetch(req *FetchRequest, resp *Void) error {
	return logErr("Fetch", o.FetchFn(req))
}

func (o *Proxy) FetchAll(req *FetchAllRequest, resp *Void) error {
	return logErr("FetchAll", o.FetchAllFn(req))
}

func (o *Proxy) FetchPart(req *FetchPartRequest, resp *Void) error {
	return logErr("FetchPart", o.FetchPartFn(req))
}

func (o *Proxy) Mount(req *MountRequest, resp *Void) error {
	if req.MinSamplesPerFile <= 0 {
		return errInvalidArgument
	}
	if req.ClusterSize <= 0 {
		return errInvalidArgument
	}
	return logErr("Mount", o.MountFn(req))
}

func (o *Proxy) Pause(req *Void, resp *Void) error {
	return logErr("Pause", o.PauseFn())
}

func (o *Proxy) SetSAS(req *SetSASRequest, resp *Void) error {
	return logErr("SetSAS", o.SetSASFn(req))
}

func (o *Proxy) SetPeers(req *SetPeerRequest, resp *Void) error {
	return logErr("SetPeers", o.SetPeersFn(req))
}

func (o *Proxy) SetRedundency(req *SetRedundencyRequest, resp *Void) error {
	return logErr("SetRedundency", o.SetRedundencyFn(req))
}

func logErr(name string, err error) error {
	if err != nil {
		log.Printf("%s: %v", name, err)
	}
	return err
}

// server Python API
func (p *Proxy) serveJSONRPC(w http.ResponseWriter, req *http.Request) {
	// logRequestDetail(`serveJSONRPC: `, req)
	type serverRequest struct {
		Method string           `json:"method"`
		Params *json.RawMessage `json:"params"`
		Id     *json.RawMessage `json:"id"`
	}
	var reqMsg serverRequest
	if err := json.NewDecoder(req.Body).Decode(&reqMsg); err != nil {
		log.Printf("%v", err)
		return
	}
	log.Printf("medhod: %s", reqMsg.Method)
	switch reqMsg.Method {
	case "Proxy.AddIndex":
		{
			var x []AddIdxRequest
			var y Void
			if err := json.Unmarshal(*reqMsg.Params, &x); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			logErrorIf(p.AddIndex(&x[0], &y))
			json.NewEncoder(w).Encode(&y)
			return
		}
	case "Proxy.Mount":
		{
			var x []MountRequest
			var y Void
			if err := json.Unmarshal(*reqMsg.Params, &x); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			logErrorIf(p.Mount(&x[0], &y))
			json.NewEncoder(w).Encode(&y)
			return
		}
	case "Proxy.GetRoot":
		{
			var x Void
			var y string
			logErrorIf(p.GetRoot(&x, &y))
			json.NewEncoder(w).Encode(&y)
			return
		}
	default:
		http.Error(w, "", http.StatusBadRequest)
	}

	// bs, err :=
	// io.Copy(os.Stdout, req.Body)

	// rs := rpc.NewServer()
	// if err := rs.Register(p); err != nil {
	// 	log.Printf("%v", err)
	// }
	// conn, _, err := w.(http.Hijacker).Hijack()
	// if err != nil {
	// 	log.Printf("%v", err)
	// }
	// log.Printf("Hijacker OK")
	// srv := jsonrpc.NewServerCodec(conn)
	// if err := rs.ServeRequest(srv); err != nil {
	// 	log.Printf("%v", err)
	// }
	// // srv.ReadRequestHeader(req)
}

func logErrorIf(err error) error {
	if err != nil {
		log.Printf("err: %v", err)
	}
	return err
}
