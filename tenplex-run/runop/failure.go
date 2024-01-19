package runop

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"path"
	"time"

	"github.com/kungfu-team/tenplex/tenplex-run/job"
)

var client = http.Client{Timeout: 3 * time.Second}

func simulateFailures(jc *job.JobConfig, jobID string, n int) error {
	for i := 0; i < n; i++ {
		if err := simulateOneFailure(jc, jobID, i/jc.Cluster.GPUsPerHost, i); err != nil {
			log.Printf("%s failed: %v", "simulateOneFailure", err)
			return err
		}
	}
	return nil
}

func simulateOneFailure(jobConf *job.JobConfig, jobID string, hostIdx int, workerID int) error {
	ip := jobConf.Cluster.Hosts[hostIdx]
	p := path.Join("/job", jobID, "save", str(workerID))
	log.Printf("simulateOneFailure by del: %s from [%d]=%s", p, hostIdx, ip)
	q := url.Values{}
	q.Set(`path`, p)
	u := url.URL{
		Scheme:   `http`,
		Host:     fmt.Sprintf("%s:%d", ip, jobConf.MLFSPort),
		Path:     "/delete1",
		RawQuery: q.Encode(),
	}
	req, err := http.NewRequest(http.MethodDelete, u.String(), nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	bs, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("delete target dir failed, status: %s, error: %s, url: %s", resp.Status, string(bs), u.String())
	}
	return nil
}
