package para_config

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
)

type ScalingPoint struct {
	Step *int `json:"step"`
	Time *int `json:"time"`
	Size int  `json:"size"`
}

func (s ScalingPoint) String() string {
	showIntPtr := func(p *int) string {
		if p == nil {
			return `nil`
		}
		return strconv.Itoa(*p)
	}
	return fmt.Sprintf("Step: %s, time: %s, size: %d", showIntPtr(s.Step), showIntPtr(s.Time), s.Size)
}

var Empty ParallelismConfig // Size == PPSize == MPSize == 0

type Schedule = []ScalingPoint

func GenSchedule(scheduleFile string) Schedule {
	s, err := LoadScheduleFile(scheduleFile)
	if err != nil {
		log.Panicf("LoadScheduleFile: %v", err)
	}
	for i, p := range s {
		log.Printf("schedule[%d/%d]: %s", i+1, len(s), p)
	}
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
