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

func GenParaConfig() ParaConfig {
	para_conf := make(ParaConfig)

	para_conf[1] = ParallelismConfig{Size: 1, PPSize: 1, MPSize: 1}
	para_conf[2] = ParallelismConfig{Size: 2, PPSize: 1, MPSize: 2}
	para_conf[4] = ParallelismConfig{Size: 4, PPSize: 1, MPSize: 2}
	para_conf[8] = ParallelismConfig{Size: 8, PPSize: 2, MPSize: 2}
	para_conf[16] = ParallelismConfig{Size: 16, PPSize: 4, MPSize: 2}
	para_conf[32] = ParallelismConfig{Size: 32, PPSize: 4, MPSize: 2}

	return para_conf
}

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
