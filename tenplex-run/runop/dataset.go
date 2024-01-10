package runop

import (
	"fmt"
	"log"

	"github.com/kungfu-team/mlfs/ds"
	"github.com/kungfu-team/mlfs/mlfs"
	"github.com/kungfu-team/tenplex-run/job"
)

func mount(cli *mlfs.Client, ds ds.Dataset, jobID string, batchSize, progress, dpSize, seed int) error {
	var err error
	if err = cli.AddIndex(ds.Name, ds.IndexURL); err != nil {
		log.Printf("%v", err)
	}
	if err = cli.Mount(jobID, ds.Name, int64(progress), batchSize, dpSize, seed); err != nil {
		log.Printf("%v", err)
	}
	var s string
	if err = cli.GetRoot(&s); err != nil {
		log.Printf("%v", err)
	}
	log.Printf("root: %s", s)
	return err
}

func addDataset(dpSize, progress int, jobConf *job.JobConfig, jobID string) error {
	seed := 42
	log.Printf("MLFS dataset seed %d", seed)
	for _, host := range jobConf.Cluster.Hosts {
		cli, err := mlfs.NewClientTo(host, mlfs.DefaultCtrlPort)
		if err != nil {
			return fmt.Errorf("%s %v", host, err)
		}
		if err := mount(cli, jobConf.Dataset, jobID, jobConf.BatchSize, progress, dpSize, seed); err != nil {
			return fmt.Errorf("%s %v", host, err)
		}
	}
	return nil
}
