package cluster

import (
	"flag"

	"github.com/kungfu-team/tenplex-run/listflag"
)

type Cluster struct {
	GPUsPerHost      int
	GPUsPerContainer int
	Hosts            listflag.Strings
}

func NewCluster(gpuPerHost int, gpusPerContainer int, hosts ...string) *Cluster {
	return &Cluster{
		GPUsPerHost:      gpuPerHost,
		GPUsPerContainer: gpusPerContainer,
		Hosts:            hosts,
	}
}

func (c *Cluster) RegisterFlags(flag *flag.FlagSet) {
	flag.IntVar(&c.GPUsPerHost, "gpu-per-host", 4, ``)
	flag.IntVar(&c.GPUsPerContainer, "gpu-per-container", 4, ``)
	flag.Var(&c.Hosts, `hosts`, `IPs separated by ,`)
}
