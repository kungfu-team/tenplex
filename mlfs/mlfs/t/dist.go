package t

import (
	"context"
	"log"
	"path"
	"sync"

	"github.com/kungfu-team/mlfs/ds"
	"github.com/kungfu-team/mlfs/mlfs"
	"github.com/kungfu-team/mlfs/utils"
)

type DistTest struct {
	HTTPPort int
	CtrlPort int
	Mount    string
	Tmp      string
	JobID    string
	DS       ds.Dataset
	DP       int
}

func (dt DistTest) genPeers() ([]mlfs.Daemon, []mlfs.Test) {
	var ds []mlfs.Daemon
	for i := 0; i < dt.DP; i++ {
		d := mlfs.Daemon{
			HTTPPort: dt.HTTPPort + i,
			CtrlPort: dt.CtrlPort + i,
			Mount:    path.Join(dt.Mount, str(i)),
			Tmp:      path.Join(dt.Tmp, str(i)),
		}
		ds = append(ds, d)
	}

	var ts []mlfs.Test
	for i, d := range ds {
		t := mlfs.Test{
			Port:  d.HTTPPort,
			JobID: dt.JobID,
			Rank:  newP(i),
		}
		ts = append(ts, t)
	}
	return ds, ts
}

func newP[T any](v T) *T { return &v }

func (dt DistTest) Run() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var wg sync.WaitGroup
	ds, ts := dt.genPeers()
	for i, d := range ds {
		wg.Add(1)
		go func(i int, d mlfs.Daemon) {
			d.RunCtx(ctx)
			log.Printf("daemon[%d] stopped", i)
			wg.Done()
		}(i, d)
	}

	for _, d := range ds {
		if ok := mlfs.WaitTCP(``, d.CtrlPort); !ok {
			log.Panic("wait timeout")
		}
	}
	log.Printf("all %d daemons are up", len(ds))

	for _, d := range ds {
		cli, err := mlfs.NewClient(d.CtrlPort)
		if err != nil {
			utils.ExitErr(err)
		}
		ds := dt.DS
		if err := cli.AddIndex(ds.Name, ds.IndexURL); err != nil {
			utils.ExitErr(err)
		}
		bs := 100
		if err := cli.Mount(dt.JobID, ds.Name, int64(0), bs, dt.DP, 0); err != nil {
			utils.ExitErr(err)
		}
	}
	log.Printf("dataset mounted to %d daemons", len(ds))

	for i, d := range ds {
		// go
		func(i int) error {
			cli, err := mlfs.NewClient(d.CtrlPort)
			if err != nil {
				utils.ExitErr(err)
			}
			ds := dt.DS
			if err := cli.FetchPart(ds.Name, i, dt.DP); err != nil {
				log.Printf("FetchPart(%s, %d, %d) failed: %v", ds.Name, i, dt.DP, err)
			}
			return nil
		}(i)
	}
	log.Printf("all %d daemons started FetchPart", len(ds))

	for _, t := range ts {
		wg.Add(1)
		go func(t mlfs.Test) {
			// t.Run()
			wg.Done()
		}(t)
	}
	cancel()
	wg.Wait()
}
