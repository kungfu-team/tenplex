package para_config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
)

type MDP struct {
	MPSize int `json:"mp_size"`
	DPSize int `json:"dp_size"`
	PPSize int `json:"pp_size"`
}

func (c MDP) GetTotalSize() int {
	return c.MPSize * c.DPSize * c.PPSize
}

func (c MDP) ID() string {
	return fmt.Sprintf("mp%d-dp%d-pp%d", c.MPSize, c.DPSize, c.PPSize)
}

func (c MDP) String() string {
	return fmt.Sprintf("size:%d = mp:%d x dp:%d x pp:%d", c.GetTotalSize(), c.MPSize, c.DPSize, c.PPSize)
}

type ParaConfig map[int]MDP

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
