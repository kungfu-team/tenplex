package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/kungfu-team/mlfs/ds"
	"github.com/kungfu-team/mlfs/mlfs/t"
	"github.com/kungfu-team/mlfs/par"
)

var pwd, _ = os.Getwd()

var testds = ds.Dataset{
	Name:     `fake-ds`,
	IndexURL: `http://127.0.0.1:8888/mlfs/test/a.idx`,
}

var (
	port = flag.Int(`port`, 8888, ``)
)

func main() {
	flag.Parse()
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	run()
}

func run() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := &t.Cloud{
		Port:           *port,
		NFiles:         256,
		RecordsPerFile: 1024,
		SizePerRecord:  1024 * 16,
	}
	dt := t.DistTest{
		HTTPPort: 30000,
		CtrlPort: 40000,
		Mount:    path.Join(pwd, `mnt`),
		Tmp:      path.Join(pwd, `tmp`),
		JobID:    `A`,
		DP:       4,
		DS:       testds,
	}
	p := par.New(2)
	p.Do(func() {
		c.Run(ctx)
		log.Printf("fake Cloud stopped")
	})
	if ok := waitHTTP(ctx, dt.DS.IndexURL); !ok {
		log.Fatalf("wait %s timeout", dt.DS.IndexURL)
	}
	dt.Run()
	log.Printf("distributed test finished")
	cancel()
	p.Wait()
}

func waitHTTP(ctx context.Context, url string) bool {
	tk := time.NewTicker(1 * time.Second)
	defer tk.Stop()
	for {
		if httpURLExist(url) {
			break
		}
		select {
		case <-ctx.Done():
			return false
		case <-tk.C:
			continue
		}
	}
	log.Printf("%s is ready", url)
	return true
}

var hc = http.Client{Timeout: 1 * time.Second}

func httpURLExist(url string) bool {
	resp, err := hc.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
