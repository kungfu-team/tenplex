package para_config

import (
	"encoding/json"
	"log"
	"os"
	"sort"
)

type ParallelismConfig struct {
	Size   int `json:"size"`
	PPSize int `json:"pp_size"`
	MPSize int `json:"mp_size"`
}

type ParaConfig = map[int]ParallelismConfig

func LoadFile(filename string) (ParaConfig, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	var config ParaConfig
	if err := json.NewDecoder(f).Decode(&config); err != nil {
		return nil, err
	}
	return config, nil
}

func Log(config ParaConfig) {
	log.Printf("Using Parallelization configuration:")
	var keys []int
	for k := range config {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, k := range keys {
		v := config[k]
		log.Printf("- %d: %#v", k, v)
	}
}
