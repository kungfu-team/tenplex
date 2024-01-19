package megatronlm

import (
	"fmt"

	"github.com/kungfu-team/tenplex/state_transformer/meta"
)

func getValMapToMap(obj map[string]interface{}, key string) (map[string]interface{}, error) {
	if newObj, ok := obj[key].(map[string]interface{}); ok {
		return newObj, nil
	}
	return nil, fmt.Errorf("getValMapToMap cast failed")
}

func getValMapToSlice(obj map[string]interface{}, key string) ([]interface{}, error) {
	if newObj, ok := obj[key].([]interface{}); ok {
		return newObj, nil
	}
	return nil, fmt.Errorf("getValMapToSlice cast failed")
}

func getValSliceToMap(obj []interface{}, key int) (map[string]interface{}, error) {
	if newObj, ok := obj[key].(map[string]interface{}); ok {
		return newObj, nil
	}
	return nil, fmt.Errorf("getValSliceToMap cast failed")
}

func getGroupSizeFP16(stru map[string]interface{}, group int) (int, error) {
	x, err := getValMapToMap(stru, "optimizer")
	if err != nil {
		return -1, err
	}
	x, err = getValMapToMap(x, "optimizer")
	if err != nil {
		return -1, err
	}
	y, err := getValMapToSlice(x, "param_groups")
	if err != nil {
		return -1, err
	}
	x, err = getValSliceToMap(y, group)
	if err != nil {
		return -1, err
	}
	y, err = getValMapToSlice(x, "params")
	if err != nil {
		return -1, err
	}
	return len(y), nil
}

func getGroupSizeFP32(stru map[string]interface{}, group int) (int, error) {
	x, err := getValMapToMap(stru, "optimizer")
	if err != nil {
		return -1, err
	}
	y, err := getValMapToSlice(x, "param_groups")
	if err != nil {
		return -1, err
	}
	x, err = getValSliceToMap(y, group)
	if err != nil {
		return -1, err
	}
	y, err = getValMapToSlice(x, "params")
	if err != nil {
		return -1, err
	}
	return len(y), nil
}

func GetGroupSizes(conf *meta.Config, before bool, rankMap *meta.RankMap, structs map[int]map[string]interface{}) (map[int]map[int]int, error) {
	mpRank := 0
	dpRank := 0
	var ppSize int
	if before {
		ppSize = conf.SourcePPDegree
	} else {
		ppSize = conf.TargetPPDegree
	}
	groupSizes := make(map[int]map[int]int)
	for ppRank := 0; ppRank < ppSize; ppRank++ {
		groupSizes[ppRank] = make(map[int]int)
		mdpRank := meta.MDPRank{PPRank: ppRank, MPRank: mpRank, DPRank: dpRank}
		rank, ok := rankMap.Rank[mdpRank]
		if !ok {
			return nil, fmt.Errorf("cannot access RankMap with %v", mdpRank)
		}
		stru, ok := structs[rank]
		if !ok {
			return nil, fmt.Errorf("cannot access Structs with %v", rank)
		}

		if conf.Precision == "fp16" {
			size, err := getGroupSizeFP16(stru, 0)
			if err != nil {
				return nil, err
			}
			groupSizes[ppRank][0] = size
			size, err = getGroupSizeFP16(stru, 1)
			if err != nil {
				return nil, err
			}
			groupSizes[ppRank][1] = size
		} else if conf.Precision == "fp32" {
			size, err := getGroupSizeFP32(stru, 0)
			if err != nil {
				return nil, err
			}
			groupSizes[ppRank][0] = size
			size, err = getGroupSizeFP32(stru, 1)
			if err != nil {
				return nil, err
			}
			groupSizes[ppRank][1] = size
		}
	}
	return groupSizes, nil
}
