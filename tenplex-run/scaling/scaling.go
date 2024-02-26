package scaling

type TimeScalingPoint struct {
	Time     int `json:"time"` // time in minutes
	ParaConf `json:"para_conf"`
}

type StepScalingPoint struct {
	Step     int `json:"step"`
	ParaConf `json:"para_conf"`
}

type ParaConf struct {
	Size   int `json:"size"`
	PPSize int `json:"pp_size"`
	MPSize int `json:"mp_size"`
}

type Schedule = []StepScalingPoint

var Empty ParaConf // Size == PPSize == MPSize == 0
