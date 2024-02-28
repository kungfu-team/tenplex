package para_config

import (
	"encoding/json"
	"log"
	"os"
)

type ScalingPoint struct {
	Step *int `json:"step"`
	Time *int `json:"time"`
	Size int  `json:"size"`
}

var Empty ParallelismConfig // Size == PPSize == MPSize == 0

type Schedule = []ScalingPoint

func GenSchedule(scheduleFile string) Schedule {
	s, err := LoadScheduleFile(scheduleFile)
	if err != nil {
		log.Panicf("LoadScheduleFile: %v", err)
	}
	log.Printf("schedule")
	for _, p := range s {
		log.Printf("schedule step %#v", p)
	}
	log.Printf("schedule end")
	return s
}

func LoadScheduleFile(filename string) (Schedule, error) {
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
