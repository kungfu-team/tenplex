package job

import (
	"encoding/json"
	"log"
	"os"
)

type ScalingPoint struct {
	Step     int                `json:"step"`
	ParaConf *ParallelismConfig `json:"para_conf"`
}

var Empty ParallelismConfig // Size == PPSize == MPSize == 0

type Schedule = []ScalingPoint

func GenSchedule(scheduleFile string) Schedule {
	if len(scheduleFile) > 0 {
		s, err := LoadFile(scheduleFile)
		if err != nil {
			log.Panicf("scheduler.LoadFile: %v", err)
		}
		log.Printf("schedule")
		for _, p := range s {
			log.Printf("schedule step %d %v", p.Step, *p.ParaConf)
		}
		log.Printf("schedule end")
		return s
	}

	schedule := Schedule{
		{0, &ParallelismConfig{Size: 4, PPSize: 2, MPSize: 2}},
		{100, &Empty},
	}
	return schedule
}

func LoadFile(filename string) (Schedule, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	var obj Schedule
	if err := json.NewDecoder(f).Decode(&obj); err != nil {
		return nil, err
	}
	return obj, nil
}
