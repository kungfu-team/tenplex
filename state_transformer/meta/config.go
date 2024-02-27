package meta

import "github.com/kungfu-team/tenplex/tenplex-run/listflag"

type Config struct {
	CkptStructDir   string
	SourceMPDegree  int
	TargetMPDegree  int
	SourcePPDegree  int
	TargetPPDegree  int
	SourceSize      int
	TargetSize      int
	SourceDPDegree  int
	TargetDPDegree  int
	Precision       string
	OutputTimestamp string
	InputTimestamp  string
	SourceHosts     listflag.Strings
	TargetHosts     listflag.Strings
	Port            int
	GpusPerHost     int
	MdpLibrary      string
	SeqLength       int
	JobID           string
	NumLayers       int
	Model           string
	ModelSize       string
	VocabSize       int
	Step            int
	Central         bool
}
