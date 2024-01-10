package meta

import "testing"

func Test_Load(t *testing.T) {
	conf := Config{
		CkptStructDir:   "/home/marcel/.tenplex/transformer-checkpoint",
		SourceMPDegree:  4,
		TargetMPDegree:  2,
		SourcePPDegree:  3,
		TargetPPDegree:  2,
		SourceSize:      12,
		TargetSize:      8,
		SourceDPDegree:  1,
		TargetDPDegree:  2,
		Precision:       "fp16",
		OutputTimestamp: "",
		InputTimestamp:  "",
		SourceHosts:     []string{"a", "b", "c"},
		TargetHosts:     []string{"a", "b"},
		Port:            20010,
		GpusPerHost:     4,
		MdpLibrary:      "megatron-lm",
		SeqLength:       1024,
		JobID:           "jobid",
		NumLayers:       24,
	}
	rankMap, err := CreateRankMap(&conf, true)
	stru, err := LoadStructs(&conf, rankMap, true)
	if err != nil {
		t.Logf("Error %v", err)
		return
	}
	t.Logf("Structures length %d", len(stru))
}
