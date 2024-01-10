package client

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex/state_transformer/go/meta"
	"github.com/kungfu-team/tenplex/tensor"
	"golang.org/x/exp/slices"
)

type Slice struct {
	Range []int
	Dim   int
}

type CheckpointClient struct {
	Conf              *meta.Config
	SourceRankMap     *meta.RankMap
	TargetRankMap     *meta.RankMap
	FailedSourceHosts []string
	HttpClient        http.Client
	central           bool
}

func New(conf *meta.Config, srcMap, dstMap *meta.RankMap) CheckpointClient {
	return CheckpointClient{
		Conf:          conf,
		SourceRankMap: srcMap,
		TargetRankMap: dstMap,
		HttpClient:    http.Client{Timeout: 30 * time.Second},
		central:       conf.Central,
	}
}

func (c CheckpointClient) getSourceHostIP(rank int) string {
	return c.getHostIP(rank, true)
}

func (c CheckpointClient) getTargetHostIP(rank int) string {
	return c.getHostIP(rank, false)
}

func (c CheckpointClient) createURL(ip string, pat string, query url.Values) url.URL {
	return url.URL{
		Scheme:   `http`,
		Host:     fmt.Sprintf("%s:%d", ip, c.Conf.Port),
		Path:     pat,
		RawQuery: query.Encode(),
	}
}

func (c CheckpointClient) getHostIP(rank int, source bool) string {
	var hosts []string
	if source {
		hosts = c.Conf.SourceHosts
	} else {
		hosts = c.Conf.TargetHosts
	}
	hostIdx := rank / c.Conf.GpusPerHost
	if c.central {
		hostIdx = 0
	}
	if hostIdx < 0 || hostIdx >= len(hosts) {
		panic(fmt.Sprintf("index out of range. rank %d, index %d, length %d, hosts %v, source %v", rank, hostIdx, len(hosts), hosts, source))
	}
	return hosts[hostIdx]
}

func (c CheckpointClient) QueryTensor(ip string, pa string, slice *Slice) (*tensor.Tensor, error) {
	var rangeStr string
	if slice != nil {
		if slice.Dim == 0 {
			rangeStr = fmt.Sprintf("%d:%d", slice.Range[0], slice.Range[1])
		} else if slice.Dim == 1 {
			rangeStr = fmt.Sprintf(":,%d:%d", slice.Range[0], slice.Range[1])
		} else {
			return nil, fmt.Errorf("dimension %d not supported", slice.Dim)
		}
	}
	q := url.Values{}
	q.Set("path", path.Join("/job", c.Conf.JobID, pa))
	if slice != nil {
		q.Set("range", rangeStr)
	}
	u := c.createURL(ip, `/query`, q)
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Content-Type", "x-tensor")
	resp, err := c.HttpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("query tensor failed, status: %s, url: %s", resp.Status, req.URL.String())
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	dtype := resp.Header.Get("x-tensor-dtype")
	strDims := resp.Header.Get("x-tensor-dims")
	splitDims := strings.Split(strDims, ",")
	dims := make([]int, len(splitDims))
	for i, strDim := range splitDims {
		dims[i], err = strconv.Atoi(strDim)
		if err != nil {
			return nil, err
		}
	}

	t := tensor.Tensor{Dtype: dtype,
		Dims: dims,
		Data: body}
	return &t, nil
}

func (c CheckpointClient) QueryTensorRedundancy(mdpRank *meta.MDPRank, pa string, slice *Slice) (*tensor.Tensor, error) {
	device, ok := c.SourceRankMap.Rank[*mdpRank]
	if !ok {
		return nil, fmt.Errorf("cannot get source device for source MDP rank %v", mdpRank)
	}
	hosts := c.Conf.SourceHosts
	firstHostIdx := device / c.Conf.GpusPerHost
	for i := 0; i < len(hosts); i++ {
		hostIdx := (firstHostIdx + i) % len(hosts)
		hostIP := hosts[hostIdx]
		if slices.Contains(c.FailedSourceHosts, hostIP) {
			continue
		}
		ten, err := c.QueryTensor(hostIP, pa, slice)
		if err == nil {
			return ten, nil
		}
		c.FailedSourceHosts = append(c.FailedSourceHosts, hostIP)
	}
	return nil, fmt.Errorf("cannot get %s from any host in %v", pa, hosts)
}

func (c *CheckpointClient) QueryMegatronTensor(mdpRank *meta.MDPRank, timestamp, path string, slice *Slice) (*tensor.Tensor, error) {
	device, ok := c.SourceRankMap.Rank[*mdpRank]
	if !ok {
		return nil, fmt.Errorf("cannot get source device for source MDP rank %v", mdpRank)
	}
	dirName := fmt.Sprintf("%s/%d", timestamp, device)
	megatronPath := fmt.Sprintf("%s/%s", dirName, path)

	return c.QueryTensorRedundancy(mdpRank, megatronPath, slice)
}

func (c CheckpointClient) QueryValue(ip string, pa string) ([]byte, string, error) {
	queryPath := path.Join(fmt.Sprintf("/job/%s", c.Conf.JobID), pa)
	q := url.Values{}
	q.Set("path", queryPath)
	u := c.createURL(ip, `/query`, q)
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := c.HttpClient.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("query value failed, status: %s, path: %s, IP: %s", resp.Status, queryPath, ip)
	}
	dtype := resp.Header.Get("dtype")

	return body, dtype, nil
}

func (c CheckpointClient) QueryValueRedundancy(mdpRank *meta.MDPRank, pa string) ([]byte, string, error) {
	device, ok := c.SourceRankMap.Rank[*mdpRank]
	if !ok {
		return nil, "", fmt.Errorf("cannot get source device for source MDP rank %v", mdpRank)
	}
	hosts := c.Conf.SourceHosts
	firstHostIdx := device / c.Conf.GpusPerHost
	for i := 0; i < len(hosts); i++ {
		hostIdx := (firstHostIdx + i) % len(hosts)
		hostIP := hosts[hostIdx]
		if slices.Contains(c.FailedSourceHosts, hostIP) {
			continue
		}
		val, dtype, err := c.QueryValue(hostIP, pa)
		if err == nil {
			return val, dtype, nil
		}
		c.FailedSourceHosts = append(c.FailedSourceHosts, hostIP)
	}
	return nil, "", fmt.Errorf("cannot get %s from any host in %v", pa, hosts)
}

func (c *CheckpointClient) QueryMegatronValue(mdpRank *meta.MDPRank, timestamp, path string) ([]byte, string, error) {
	device, ok := c.SourceRankMap.Rank[*mdpRank]
	if !ok {
		return nil, "", fmt.Errorf("cannot get source device for source MDP rank %v", mdpRank)
	}
	dirName := fmt.Sprintf("%s/%d", timestamp, device)
	megatronPath := fmt.Sprintf("%s/%s", dirName, path)

	return c.QueryValueRedundancy(mdpRank, megatronPath)
}

func (c *CheckpointClient) QueryTargetDir(device int, timestamp string) (string, error) {
	ip := c.getTargetHostIP(device)
	queryPath := path.Join("/job", c.Conf.JobID, timestamp, strconv.Itoa(device))
	q := url.Values{}
	q.Set("path", queryPath)
	u := c.createURL(ip, `/query`, q)
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "x-dir")
	resp, err := c.HttpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("query target dir failed, status: %s, path: %s, IP: %s", resp.Status, queryPath, ip)
	}
	return string(body), nil
}

func (c *CheckpointClient) DeleteTargetDir(device int, timestamp string) error {
	ip := c.getTargetHostIP(device)
	queryPath := path.Join("/job", c.Conf.JobID, timestamp, strconv.Itoa(device))
	q := url.Values{}
	q.Set("path", queryPath)
	u := c.createURL(ip, `/delete`, q)
	req, err := http.NewRequest(http.MethodDelete, u.String(), nil)
	if err != nil {
		return err
	}
	resp, err := c.HttpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("delete target dir failed, status: %s, error: %s, url: %s", resp.Status, string(body), u.String())
	}

	log.Printf("Deleted target directory at %s %s", u.String(), queryPath)
	return nil
}

var uploadedTensors []string

func (c *CheckpointClient) UploadTensor(ten *tensor.Tensor, pa string, device int) error {
	if slices.Contains(uploadedTensors, pa) {
		panic("already uploaded")
	} else {
		uploadedTensors = append(uploadedTensors, pa)
	}
	ip := c.getTargetHostIP(device)
	buf := bytes.NewBuffer(ten.Data)
	queryPath := path.Join("/job", c.Conf.JobID, pa)
	var dims []string
	for _, d := range ten.Dims {
		dims = append(dims, strconv.Itoa(d))
	}
	q := url.Values{}
	q.Set("path", queryPath)
	q.Set("dtype", ten.Dtype)
	q.Set("dims", strings.Join(dims, ","))
	u := c.createURL(ip, `/upload`, q)
	req, err := http.NewRequest(http.MethodPost, u.String(), buf)
	if err != nil {
		return err
	}
	req.Header.Add("Content-Type", "x-tensor")
	resp, err := c.HttpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bs, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upload tensor failed, status: %s, error: %s, path: %s, IP: %s", resp.Status, string(bs), queryPath, ip)
	}
	return nil
}

func (c *CheckpointClient) UploadValue(val []byte, pa string, device int, replace bool) error {
	ip := c.getTargetHostIP(device)
	queryPath := path.Join("/job", c.Conf.JobID, pa)
	q := url.Values{}
	q.Set("path", queryPath)
	u := c.createURL(ip, `/upload`, q)
	req, err := http.NewRequest(http.MethodPost, u.String(), bytes.NewBuffer(val))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "x-value")
	if replace {
		req.Header.Set("x-replace", "true")
	}
	resp, err := c.HttpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("upload value failed, status: %s, path: %s, IP: %s", resp.Status, queryPath, ip)
	}
	return nil
}

func (c *CheckpointClient) UploadMegatronTensor(ten *tensor.Tensor, mdpRank *meta.MDPRank, timestamp, path string) error {
	device, ok := c.TargetRankMap.Rank[*mdpRank]
	if !ok {
		return fmt.Errorf("cannot get target device for target MDP rank %v", mdpRank)
	}
	dirName := fmt.Sprintf("%s/%d", timestamp, device)
	megatronPath := fmt.Sprintf("%s/%s", dirName, path)

	return c.UploadTensor(ten, megatronPath, device)
}

func (c *CheckpointClient) UploadMegatronValue(val []byte, mdpRank *meta.MDPRank, timestamp, path string) error {
	device, ok := c.TargetRankMap.Rank[*mdpRank]
	if !ok {
		return fmt.Errorf("cannot get target device for target MDP rank %v", mdpRank)
	}
	dirName := fmt.Sprintf("%s/%d", timestamp, device)
	megatronPath := fmt.Sprintf("%s/%s", dirName, path)

	return c.UploadValue(val, megatronPath, device, false)
}
