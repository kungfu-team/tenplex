package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path"

	"github.com/kungfu-team/tenplex/state_transformer/client"
	"github.com/kungfu-team/tenplex/state_transformer/meta"
	"github.com/kungfu-team/tenplex/state_transformer/search"
)

func testSearchJsonForTensors(config *meta.Config, targetRank int, basePath string) {
	jsonPath := path.Join(basePath, fmt.Sprintf("rank%02d", targetRank))
	mpRank := 0
	ppRank := 0
	jsonPath = path.Join(jsonPath, fmt.Sprintf("mp_rank_%02d_%03d.json", mpRank, ppRank))
	content, err := os.ReadFile(jsonPath)
	if err != nil {
		log.Printf("testSearchJsonForTensors Error %v", err)
	}
	var payload map[string]interface{}
	err = json.Unmarshal(content, &payload)
	if err != nil {
		log.Printf(" testSearchJsonForTensorsError %v", err)
	}

	tensors, nonTensors, err := search.SearchJsonForTensors(payload, []string{})
	if err != nil {
		log.Printf("testSearchJsonForTensors Error %v", err)
		return
	}
	log.Printf("testSearchJsonForTensors Number of tensors %d", len(tensors))
	log.Printf("testSearchJsonForTensors Number of non tensors %d", len(nonTensors))
}

func testCkptClient(config *meta.Config) {
	ckptClient := client.CheckpointClient{
		Conf:          config,
		SourceRankMap: nil, // TODO: fix
		TargetRankMap: nil, // TODO: fix
	}
	path := "optimizer/optimizer/state/73/exp_avg.numpy.ndarray"
	mdpRank := meta.MDPRank{PPRank: 0, MPRank: 0, DPRank: 0}
	t, err := ckptClient.QueryMegatronTensor(&mdpRank, config.InputTimestamp, path, nil)
	if err != nil {
		log.Printf("%v", err)
	}
	log.Printf("testCkptClient Dims %v", t.Dims)
	log.Printf("testCkptClient Number of bytes %d", len(t.Data))
}

func testLoadStructs(conf *meta.Config) {
	before := true
	rankMap, err := meta.CreateRankMap(conf, before)
	if err != nil {
		log.Printf("testLoadStructs Error %v", err)
	}
	log.Printf("testLoadStructs rank map %v", rankMap.MDPRank)
	structs, err := meta.LoadStructs(conf, rankMap, before)
	if err != nil {
		log.Printf("testLoadStructs Error %v", err)
	}
	var keys []int
	for k := range structs {
		keys = append(keys, k)
	}
	log.Printf("testLoadStructs structs keys %v", keys)
	// log.Printf("%v", structs)
}

func main() {
	conf, targetRank := meta.ReadFlags()
	log.Printf("config %v", conf)
	log.Printf("target rank %v", targetRank)

	testCkptClient(conf)
	testSearchJsonForTensors(conf, targetRank, meta.GetStructPath(conf, false))
	testLoadStructs(conf)
}
