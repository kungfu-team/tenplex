package para_config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
)

type ParallelismConfig struct {
	Size   int `json:"size"`
	PPSize int `json:"pp_size"`
	MPSize int `json:"mp_size"`
}

func (c ParallelismConfig) GetDPSize() int {
	return c.Size / (c.PPSize * c.MPSize)
}

func (c ParallelismConfig) String() string {
	return fmt.Sprintf("size:%d = pp:%d x mp:%d x dp:%d", c.Size, c.PPSize, c.MPSize, c.GetDPSize())
}

type ParaConfig map[int]ParallelismConfig

func (pc ParaConfig) String() string {
	buf := &bytes.Buffer{}
	for i, s := range pc.Sizes() {
		mdp := pc[s]
		if i > 0 {
			fmt.Fprintf(buf, ", ")
		}
		fmt.Fprintf(buf, "%s", mdp)
	}
	return `ParaConf{` + buf.String() + `}`
}

func (p ParaConfig) Sizes() []int {
	var ss []int
	for k := range p {
		ss = append(ss, k)
	}
	sort.Ints(ss)
	return ss
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
