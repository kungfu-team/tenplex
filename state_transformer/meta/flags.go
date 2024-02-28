package meta

import (
	"flag"
)

func ReadFlags() (*Config, int) {
	var conf Config
	flag.StringVar(&conf.CkptStructDir, "ckpt-struct-dir", "", "Checkpoint structure directory")
	flag.IntVar(&conf.SourcePPDegree, "source-pp-degree", 0, "Source PP degree")
	flag.IntVar(&conf.TargetPPDegree, "target-pp-degree", 0, "Target PP degree")
	flag.IntVar(&conf.SourceMPDegree, "source-mp-degree", 0, "Source MP degree")
	flag.IntVar(&conf.TargetMPDegree, "target-mp-degree", 0, "Target MP degree")
	flag.IntVar(&conf.SourceSize, "source-size", 0, "Source size")
	flag.IntVar(&conf.TargetSize, "target-size", 0, "Target size")
	flag.IntVar(&conf.SeqLength, "sequence-length", 0, "Sequence length")
	flag.StringVar(&conf.Precision, "precision", "", "Precision")
	flag.StringVar(&conf.InputTimestamp, "input-timestamp", "", "Input timestamp")
	flag.StringVar(&conf.OutputTimestamp, "output-timestamp", "", "Output timestamp")
	flag.StringVar(&conf.MdpLibrary, "mdp-library", "", "Multi-dimensional parallelism library")
	flag.StringVar(&conf.Model, "model", "", "gpt or bert")
	flag.StringVar(&conf.ModelSize, "model-size", "", "Model size")
	flag.Var(&conf.SourceHosts, "source-hosts", "Host IPs separated by a comma")
	flag.Var(&conf.TargetHosts, "target-hosts", "Host IPs separated by a comma")
	flag.IntVar(&conf.Port, "port", 20010, "Control port")
	flag.IntVar(&conf.GpusPerHost, "gpus-per-host", 4, "Control port")
	var targetRank int
	flag.IntVar(&targetRank, "target-rank", 0, "Target rank")
	flag.StringVar(&conf.JobID, "jobid", "0", "Job ID")
	flag.IntVar(&conf.NumLayers, "num-layers", 0, "Number of layers")
	flag.IntVar(&conf.VocabSize, "vocab-size", 0, "Vocablulary size")
	flag.IntVar(&conf.Step, "step", 0, "Step/Iteration")
	flag.BoolVar(&conf.Central, "central", false, "")

	flag.Parse()

	conf.SourceDPDegree = conf.SourceSize / (conf.SourcePPDegree * conf.SourceMPDegree)
	conf.TargetDPDegree = conf.TargetSize / (conf.TargetPPDegree * conf.TargetMPDegree)

	return &conf, targetRank
}
