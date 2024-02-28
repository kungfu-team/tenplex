package meta

import "github.com/kungfu-team/tenplex/tenplex-run/listflag"

type Config struct {
	CkptStructDir   string `flag:"ckpt-struct-dir"`
	SourceMPDegree  int    `flag:"source-mp-degree"`
	SourcePPDegree  int    `flag:"source-pp-degree"`
	TargetMPDegree  int    `flag:"target-mp-degree"`
	TargetPPDegree  int    `flag:"target-pp-degree"`
	SourceSize      int    `flag:"source-size"`
	TargetSize      int    `flag:"target-size"`
	SourceDPDegree  int
	TargetDPDegree  int
	Precision       string           `flag:"precision"`
	InputTimestamp  string           `flag:"input-timestamp"`
	OutputTimestamp string           `flag:"output-timestamp"`
	SourceHosts     listflag.Strings `flag:"source-hosts"`
	TargetHosts     listflag.Strings `flag:"target-hosts"`
	Port            int              `flag:"port" default:"20010"`
	GpusPerHost     int              `flag:"gpus-per-host" default:"4"`
	MdpLibrary      string           `flag:"mdp-library"`
	SeqLength       int              `flag:"sequence-length"`
	JobID           string           `flag:"jobid" default:"0"`
	NumLayers       int              `flag:"num-layers"`
	Model           string           `flag:"model"`
	ModelSize       string           `flag:"model-size"`
	VocabSize       int              `flag:"vocab-size"`
	Step            int              `flag:"step"`
	Central         bool             `flag:"central"`
}
