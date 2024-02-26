package deviceallocation

import (
	"encoding/json"
	"log"
	"os"
	"sort"

	"github.com/kungfu-team/tenplex/tenplex-run/job"
)

type DeviceAllocations = map[int]job.ParallelismConfig

func GenDeviceAllocations() DeviceAllocations {
	deviceAllos := make(DeviceAllocations)

	deviceAllos[1] = job.ParallelismConfig{Size: 1, PPSize: 1, MPSize: 1}
	deviceAllos[2] = job.ParallelismConfig{Size: 2, PPSize: 1, MPSize: 2}
	deviceAllos[4] = job.ParallelismConfig{Size: 4, PPSize: 1, MPSize: 2}
	deviceAllos[8] = job.ParallelismConfig{Size: 8, PPSize: 2, MPSize: 2}
	deviceAllos[16] = job.ParallelismConfig{Size: 16, PPSize: 4, MPSize: 2}
	deviceAllos[32] = job.ParallelismConfig{Size: 32, PPSize: 4, MPSize: 2}

	return deviceAllos
}

func LoadFile(filename string) (DeviceAllocations, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	var config DeviceAllocations
	if err := json.NewDecoder(f).Decode(&config); err != nil {
		return nil, err
	}
	return config, nil
}

func Log(da DeviceAllocations) {
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
