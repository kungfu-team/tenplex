package cluster_test

import (
	"flag"
	"testing"

	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
)

func Test_1(t *testing.T) {
	var c cluster.Cluster
	f := flag.NewFlagSet(`prog`, flag.ExitOnError)
	c.RegisterFlags(f)
	f.Parse([]string{
		`-gpu-per-host`, `8`,
		`-hosts`, `1.2.3.4,4.3.2.1`,
	})
	t.Logf("%#v", c)
	if c.GPUsPerHost != 8 {
		t.Errorf("parse -gpu-per-host failed: %q", c.GPUsPerHost)
	}
	if c.GPUsPerContainer != 4 {
		t.Errorf("default -gpu-per-container failed: %q", c.GPUsPerContainer)
	}
	if c.Hosts[0] != `1.2.3.4` {
		t.Errorf("parse -hosts failed: %q", c.Hosts)
	}
}
