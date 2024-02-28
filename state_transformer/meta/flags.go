package meta

import (
	"flag"

	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

func ReadFlags() (*Config, int) {
	var conf Config
	structflag.RegisterFlags(&conf, flag.CommandLine)
	var targetRank int
	flag.IntVar(&targetRank, "target-rank", 0, "Target rank")

	flag.Parse()

	conf.SourceDPDegree = conf.SourceSize / (conf.SourcePPDegree * conf.SourceMPDegree)
	conf.TargetDPDegree = conf.TargetSize / (conf.TargetPPDegree * conf.TargetMPDegree)

	return &conf, targetRank
}
