package runop

import (
	"fmt"

	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/mlfs/pid"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
)

func setRedundancy(jobConf *job.JobConfig) error {
	redu := 1

	var peerList mlfs.PeerList
	for _, host := range jobConf.Cluster.Hosts {
		peerList = append(peerList, mlfs.Peer{IPv4: pid.MustParseIPv4(host), Port: mlfs.DefaultCtrlPort})
	}

	for _, host := range jobConf.Cluster.Hosts {
		cli, err := mlfs.NewClientTo(host, mlfs.DefaultCtrlPort)
		if err != nil {
			return fmt.Errorf("%s %v", host, err)
		}
		err = cli.SetPeers(peerList)
		if err != nil {
			return err
		}
		err = cli.SetRedundency(redu)
		if err != nil {
			return err
		}
	}
	return nil
}
