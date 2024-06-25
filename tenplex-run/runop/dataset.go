package runop

import (
	"fmt"
	"log"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
)

func mount(cli *mlfs.Client, ds ds.Dataset, jobID string, batchSize, progress, dpSize, seed int, noShuffle bool) error {
	if err := cli.AddIndex(ds.Name, ds.IndexURL); err != nil {
		return err
	}
	if err := cli.Mount(jobID, ds.Name, int64(progress), batchSize, dpSize, seed, noShuffle); err != nil {
		return err
	}
	var s string
	if err := cli.GetRoot(&s); err != nil {
		return err
	}
	return nil
}

func AddDataset(dpSize, progress int, jobConf *job.JobConfig) error {
	for _, host := range jobConf.Cluster.Hosts {
		cli, err := mlfs.NewClientTo(host, jobConf.MLFSPort)
		if err != nil {
			return fmt.Errorf("%s %v", host, err)
		}
		if err := mount(cli, jobConf.Dataset, jobConf.ID, jobConf.BatchSize, progress, dpSize, jobConf.Seed, jobConf.NoShuffle); err != nil {
			return fmt.Errorf("%s %v", host, err)
		}
		log.Printf("Dataset added: host %s, batch size %d, progress %d, DP size %d", host, jobConf.BatchSize, progress, dpSize)
	}
	return nil
}
