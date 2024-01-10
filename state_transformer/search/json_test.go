package search

import (
	"testing"

	"github.com/kungfu-team/tenplex/state_transformer/go/meta"
)

func TestSearchJson(t *testing.T) {
	conf := meta.Config{
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

	targetRankMap, err := meta.CreateRankMap(&conf, false)
	if err != nil {
		t.Fail()
	}
	targetStructs, err := meta.LoadStructs(&conf, targetRankMap, false)
	if err != nil {
		t.Fail()
	}
	device := 1
	stru := targetStructs[device]
	tensors, nonTensors, err := SearchJsonForTensors(stru, []string{})
	if err != nil {
		t.Fail()
	}
	_ = nonTensors
	_ = tensors
	// for i, ten := range tensors {
	// 	t.Logf("%d %v", i, ten)
	// }
}

func equal(a, b []string) bool {
	for i, el := range a {
		if el != b[i] {
			return false
		}
	}
	return true
}

func duplicates(arr [][]string) []int {
	nums := make([]int, len(arr))
	for i, a := range arr {
		for j, b := range arr {
			if i != j {
				if equal(a, b) {
					nums[i]++
				}
			}
		}
	}

	return nums
}

func duplicatesBetween(arr, brr [][]string) []int {
	nums := make([]int, len(arr))
	for i, a := range arr {
		for _, b := range brr {
			if equal(a, b) {
				nums[i]++
			}

		}
	}

	return nums
}

func checkAllZero(a []int) bool {
	for _, el := range a {
		if el != 0 {
			return false
		}
	}
	return true
}

func TestSearchJsonDuplicates(t *testing.T) {
	conf := meta.Config{
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

	targetRankMap, err := meta.CreateRankMap(&conf, false)
	if err != nil {
		t.Fail()
	}
	targetStructs, err := meta.LoadStructs(&conf, targetRankMap, false)
	if err != nil {
		t.Fail()
	}
	device := 1
	stru := targetStructs[device]
	tensors, nonTensors, err := SearchJsonForTensors(stru, []string{})
	if err != nil {
		t.Fail()
	}

	dup := duplicates(tensors)
	zer := checkAllZero(dup)
	if zer {
		t.Logf("NO duplicates in tensors")
	} else {
		t.Logf("Duplicates in tensors")
	}

	dup = duplicates(nonTensors)
	zer = checkAllZero(dup)
	if zer {
		t.Logf("NO duplicates in non-tensors")
	} else {
		t.Logf("Duplicates in non-tensors")
	}

	dup = duplicatesBetween(tensors, nonTensors)
	zer = checkAllZero(dup)
	if zer {
		t.Logf("NO duplicates in between")
	} else {
		t.Logf("Duplicates in between")
	}
}
