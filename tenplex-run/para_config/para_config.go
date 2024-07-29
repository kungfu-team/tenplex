package para_config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
)

type MultiDimensionalParallelism struct {
	MPSize int `json:"mp_size"`
	DPSize int `json:"dp_size"`
	PPSize int `json:"pp_size"`
}

func (c MultiDimensionalParallelism) GetTotalSize() int {
	return c.MPSize * c.DPSize * c.PPSize
}

func (c MultiDimensionalParallelism) ID() string {
	return fmt.Sprintf("mp%d-dp%d-pp%d", c.MPSize, c.DPSize, c.PPSize)
}

func (c MultiDimensionalParallelism) String() string {
	return fmt.Sprintf("size:%d = mp:%d x dp:%d x pp:%d", c.GetTotalSize(), c.MPSize, c.DPSize, c.PPSize)
}

type ParaConfig map[int]MultiDimensionalParallelism

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

var errInvalidMDP = errors.New(`invalid MDP`)

func LoadFile(filename string) (ParaConfig, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	var config ParaConfig
	if err := json.NewDecoder(f).Decode(&config); err != nil {
		return nil, err
	}
	for s, mdp := range config {
		if s != mdp.GetTotalSize() {
			return nil, errInvalidMDP
		}
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
