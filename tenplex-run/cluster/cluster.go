package cluster

import (
	"flag"

	"github.com/kungfu-team/tenplex/tenplex-run/listflag"
	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

type Cluster struct {
	GPUsPerHost      int              `flag:"gpu-per-host" default:"4"`
	GPUsPerContainer int              `flag:"gpu-per-container" default:"4"`
	Hosts            listflag.Strings `flag:"hosts"`
}

func NewCluster(gpuPerHost int, gpusPerContainer int, hosts ...string) *Cluster {
	return &Cluster{
		GPUsPerHost:      gpuPerHost,
		GPUsPerContainer: gpusPerContainer,
		Hosts:            hosts,
	}
}

func (c *Cluster) RegisterFlags(flag *flag.FlagSet) { structflag.RegisterFlags(c, flag) }
