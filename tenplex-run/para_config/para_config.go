package para_config

import (
	"encoding/json"
	"log"
	"os"
	"sort"

	"github.com/kungfu-team/tenplex/tenplex-run/job"
)

type ParaConfig = map[int]job.ParallelismConfig

func GenParaConfig() ParaConfig {
	para_conf := make(ParaConfig)

	para_conf[1] = job.ParallelismConfig{Size: 1, PPSize: 1, MPSize: 1}
	para_conf[2] = job.ParallelismConfig{Size: 2, PPSize: 1, MPSize: 2}
	para_conf[4] = job.ParallelismConfig{Size: 4, PPSize: 1, MPSize: 2}
	para_conf[8] = job.ParallelismConfig{Size: 8, PPSize: 2, MPSize: 2}
	para_conf[16] = job.ParallelismConfig{Size: 16, PPSize: 4, MPSize: 2}
	para_conf[32] = job.ParallelismConfig{Size: 32, PPSize: 4, MPSize: 2}

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

func Log(da ParaConfig) {
	log.Printf("Using DeviceAllocations:")
	var keys []int
	for k := range da {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, k := range keys {
		v := da[k]
		log.Printf("- %d: %#v", k, v)
	}
}
