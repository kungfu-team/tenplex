package mlfs

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/rpc"
	"net/url"
	"os"
	"path"
	"strconv"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/pid"
)

type Client struct {
	cli      *rpc.Client
	hc       http.Client
	endpoint url.URL
}

func NewClient(port int) (*Client, error) { return NewClientTo(`127.0.0.1`, port) }

func NewClientTo(host string, port int) (*Client, error) {
	addr := net.JoinHostPort(host, strconv.Itoa(port))
	cli, err := rpc.DialHTTP("tcp", addr)
	if err != nil {
		return nil, err
	}
	c := &Client{
		cli:      cli,
		endpoint: url.URL{Scheme: `http`, Host: addr},
	}
	return c, nil
}

func (c *Client) call(name string, req, resp interface{}) error {
	t0 := time.Now()
	defer func() { log.Printf("%s took %s", name, time.Since((t0))) }()
	if err := c.cli.Call(name, req, resp); err != nil {
		return fmt.Errorf("call %s: %v", name, err)
	}
	return nil
}

func (c *Client) AddIndex(name, idxFile string) error {
	req := AddIdxRequest{
		Name: name,
		File: idxFile,
	}
	return c.call("Proxy.AddIndex", req, &Void{})
}

func (c *Client) Fetch(url string, md5 string) error {
	req := FetchRequest{
		URL: url,
		MD5: md5,
	}
	return c.call("Proxy.Fetch", req, &Void{})
}

func (c *Client) FetchAsync(url string, md5 string) error {
	req := FetchRequest{
		URL:   url,
		MD5:   md5,
		Async: true,
	}
	return c.call("Proxy.Fetch", req, &Void{})
}

func (c *Client) FetchAll(name string, async bool) error {
	req := FetchAllRequest{
		Name:  name,
		Async: async,
	}
	return c.call("Proxy.FetchAll", req, &Void{})
}

func (c *Client) FetchPart(name string, i, n int) error {
	req := FetchPartRequest{
		Name: name,
		ShardInfo: ShardInfo{
			Rank: i,
			Size: n,
		},
	}
	return c.call("Proxy.FetchPart", req, &Void{})
}

func (c *Client) Mount(jobID string, name string, progress int64, globalBatchSize int, clusterSize int, seed int, noShuffle bool) error {
	req := MountRequest{
		JobID:             jobID,
		Name:              name,
		Progress:          progress,
		GlobalBatchSize:   globalBatchSize,
		ClusterSize:       clusterSize,
		MinSamplesPerFile: 8192,
		Seed:              seed,
		NoShuffle:         noShuffle,
	}
	return c.call("Proxy.Mount", req, &Void{})
}

func (c *Client) GetRoot(s *string) error {
	return c.call("Proxy.GetRoot", &Void{}, s)
}

func (c *Client) File(name string) ([]byte, error) {
	var r string
	c.GetRoot(&r)
	return os.ReadFile(path.Join(r, name))
}

func (c *Client) uploadTar(f io.Reader, prefix string, replace bool) error {
	q := url.Values{}
	q.Set(`prefix`, prefix)
	u := c.endpoint
	u.Path = `/upload`
	u.RawQuery = q.Encode()
	req, err := http.NewRequest(http.MethodPost, u.String(), f)
	if err != nil {
		return err
	}
	req.Header.Set(`Content-Type`, `application/x-tar`)
	if replace {
		req.Header.Set("x-replace", "true")
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

func (c *Client) UploadTar(prefix string, filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return c.uploadTar(f, prefix, false)
}

func (c *Client) Upload(name string, bs []byte) error {
	return c.UploadReplace(name, bs, false)
}

func (c *Client) UploadReplace(name string, bs []byte, replace bool) error {
	buf := &bytes.Buffer{}
	tw := tar.NewWriter(buf)
	th := &tar.Header{
		Name: name,
		Size: int64(len(bs)),
	}
	tw.WriteHeader(th)
	tw.Write(bs)
	return c.uploadTar(buf, ``, true)
}

func (c *Client) UploadFile(remotePath string, localPath string) error {
	bs, err := os.ReadFile(localPath)
	if err != nil {
		return err
	}
	return c.Upload(remotePath, bs)
}

func (c *Client) Pause() error {
	return c.call("Proxy.Pause", Void{}, &Void{})
}

func (c *Client) SetSAS(sa, sas string) error {
	req := SetSASRequest{
		SA:  sa,
		SAS: sas,
	}
	return c.call("Proxy.SetSAS", &req, &Void{})
}

func (c *Client) SetPeers(peers pid.PeerList) error {
	req := SetPeerRequest{
		Peers: peers,
	}
	return c.call("Proxy.SetPeers", &req, &Void{})
}

func (c *Client) SetRedundency(r int) error {
	req := SetRedundencyRequest{
		Redundency: r,
	}
	return c.call("Proxy.SetRedundency", &req, &Void{})
}
