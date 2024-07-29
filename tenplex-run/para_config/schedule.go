package para_config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
)

type ScalingPoint struct {
	Step *int `json:"step"`
	Time *int `json:"time"`
	Size int  `json:"size"`
}

func (s ScalingPoint) String() string {
	buf := &bytes.Buffer{}
	if s.Step != nil {
		fmt.Fprintf(buf, "step: %d", *s.Step)
	}
	if s.Time != nil {
		fmt.Fprintf(buf, "time: %d", *s.Time)
	}
	fmt.Fprintf(buf, ", size: %d", s.Size)
	return buf.String()
}

var Empty MultiDimensionalParallelism // Size == PPSize == MPSize == 0

type Schedule []ScalingPoint

func (s Schedule) String() string {
	buf := &bytes.Buffer{}
	for i, sp := range s {
		if i > 0 {
			fmt.Fprintf(buf, ", ")
		}
		fmt.Fprintf(buf, "%s", sp)
	}
	return `Schedule{` + buf.String() + `}`
}

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
