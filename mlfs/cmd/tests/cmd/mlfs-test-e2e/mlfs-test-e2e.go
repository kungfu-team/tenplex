package main

import (
	"context"
	"log"
	"os"
	"path"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/mlfs/utils"
)

var pwd, _ = os.Getwd()

type E2eTest struct {
	HTTPPort int
	CtrlPort int
	Mount    string
	JobID    string
	ds       ds.Dataset
}

func newTest(name string, ds ds.Dataset) E2eTest {
	return E2eTest{
		HTTPPort: 30000,
		CtrlPort: 30001,
		JobID:    name,
		ds:       ds,
	}
}

func testData(ds ds.Dataset) {
	c := newTest(`test-job-A`, ds)
	c.Run()
}

func main() {
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	testData(ds.SQuAD1Test)
	// testData(ds.Imagenet)
}

func (c E2eTest) Run() {
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	d := mlfs.Daemon{
		HTTPPort: c.HTTPPort,
		CtrlPort: c.CtrlPort,
		Mount:    c.Mount,
		Tmp:      path.Join(pwd, `tmp`),
	}
	t := mlfs.Test{
		Port:  c.HTTPPort,
		JobID: c.JobID,
	}
	go d.RunCtx(context.Background())
	if ok := mlfs.WaitTCP(``, c.HTTPPort); !ok {
		log.Panic("wait timeout")
	}

	cli, err := mlfs.NewClient(c.CtrlPort)
	if err != nil {
		utils.ExitErr(err)
	}

	ds := c.ds
	if err := cli.AddIndex(ds.Name, ds.IndexURL); err != nil {
		utils.ExitErr(err)
	}
	bs := 100
	if err := cli.Mount(t.JobID, ds.Name, int64(0), bs, 1, 0, false); err != nil {
		utils.ExitErr(err)
	}
	go func() {
		// if err := cli.FetchAll(ds.Name); err != nil {
		// 	log.Printf("FetchAll(%s) failed: %v", ds.Name, err)
		// }
		rank, size := 0, 1
		if err := cli.FetchPart(ds.Name, rank, size); err != nil {
			log.Printf("FetchPart(%s, %d, %d) failed: %v", ds.Name, rank, size, err)
		}
	}()
	t.Run()
}
