package meta

import (
	"flag"

	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

func ReadFlags() *Config {
	var conf Config
	structflag.RegisterFlags(&conf, flag.CommandLine)
	flag.Parse()
	conf.SourceDPDegree = conf.SourceSize / (conf.SourcePPDegree * conf.SourceMPDegree)
	conf.TargetDPDegree = conf.TargetSize / (conf.TargetPPDegree * conf.TargetMPDegree)
	return &conf
}
