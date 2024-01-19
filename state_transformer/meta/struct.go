package meta

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"
	"reflect"
	"strconv"
	"strings"
)

func flattenSlice(obj []interface{}, keys []string) ([]string, error) {
	var values []string
	for i, val := range obj {
		key := strconv.Itoa(i)
		switch v := reflect.ValueOf(val); v.Kind() {
		case reflect.Slice:
			newValues, err := flattenSlice(val.([]interface{}), append(keys, key))
			if err != nil {
				return nil, err
			}
			values = append(values, newValues...)
		case reflect.Map:
			newValues, err := FlattenMap(val.(map[string]interface{}), append(keys, key))
			if err != nil {
				return nil, err
			}
			values = append(values, newValues...)
		default:
			ident := strings.Join(append(keys, key), "/")
			values = append(values, ident)
		}

	}

	return values, nil
}

func FlattenMap(obj map[string]interface{}, keys []string) ([]string, error) {
	var values []string
	for key, val := range obj {
		if key == "tensor" {
			values = append(values, strings.Join(keys, "/"))
			continue
		}
		switch v := reflect.ValueOf(val); v.Kind() {
		case reflect.Slice:
			newValues, err := flattenSlice(val.([]interface{}), append(keys, key))
			if err != nil {
				return nil, err
			}
			values = append(values, newValues...)
		case reflect.Map:
			newValues, err := FlattenMap(val.(map[string]interface{}), append(keys, key))
			if err != nil {
				return nil, err
			}
			values = append(values, newValues...)
		default:
			ident := strings.Join(append(keys, key), "/")
			values = append(values, ident)
		}
	}

	return values, nil
}

func FlattenStructs(structs map[int]map[string]interface{}) (map[int][]string, error) {
	flatStructs := make(map[int][]string)
	var err error
	for r, v := range structs {
		flatStructs[r], err = FlattenMap(v, []string{})
		if err != nil {
			return nil, err
		}
	}
	return flatStructs, nil
}

func LoadStructs(conf *Config, rankMap *RankMap, before bool) (map[int]map[string]interface{}, error) {
	structs := make(map[int]map[string]interface{})
	structPath := GetStructPath(conf, before)
	var size int
	if before {
		size = conf.SourceSize
	} else {
		size = conf.TargetSize
	}
	for rank := 0; rank < size; rank++ {
		mdpRank, ok := rankMap.MDPRank[rank]
		if !ok {
			return nil, fmt.Errorf("LoadStructs no value for %d", rank)
		}
		var megatronPath string // TODO: make framework independent
		if (before && conf.SourcePPDegree == 1) || (!before && conf.TargetPPDegree == 1) {
			megatronPath = fmt.Sprintf("mp_rank_%02d.json", mdpRank.MPRank)
		} else {
			megatronPath = fmt.Sprintf("mp_rank_%02d_%03d.json", mdpRank.MPRank, mdpRank.PPRank)
		}
		rankPath := path.Join(structPath,
			fmt.Sprintf("rank%02d", rank),
			megatronPath)
		_, err := os.Stat(rankPath)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		var payload map[string]interface{}
		content, err := os.ReadFile(rankPath)
		if err != nil {
			return nil, err
		}
		err = json.Unmarshal(content, &payload)
		if err != nil {
			return nil, err
		}
		structs[rank] = payload
	}
	return structs, nil
}

func getShapeSlice(stru []interface{}, keySeq []string) ([]int, error) {
	k := keySeq[0]
	i, err := strconv.Atoi(k)
	if err != nil {
		return nil, err
	}
	if i >= len(stru) {
		return nil, fmt.Errorf("index %d >= %d length for key seq %v", i, len(stru), keySeq)
	}
	val := stru[i]
	switch v := reflect.ValueOf(val); v.Kind() {
	case reflect.Slice:
		shape, err := getShapeSlice(val.([]interface{}), keySeq[1:])
		if err != nil {
			return nil, err
		}
		return shape, nil
	case reflect.Map:
		shape, err := GetShape(val.(map[string]interface{}), keySeq[1:])
		if err != nil {
			return nil, err
		}
		return shape, nil
	default:
		return nil, fmt.Errorf("type %v not supported", v.Kind())
	}
}

func GetShape(stru map[string]interface{}, keySeq []string) ([]int, error) {
	if len(keySeq) == 0 {
		shapeInter, ok := stru["tensor"]
		if !ok {
			return nil, fmt.Errorf("GetShape cannot access tensor shape")
		}
		shapeInterSlice, ok := shapeInter.([]interface{})
		if !ok {
			return nil, fmt.Errorf("cannot cast interface to []interface")
		}
		var shape []int
		for _, dimSizeInter := range shapeInterSlice {
			dimSizeFloat, ok := dimSizeInter.(float64)
			if !ok {
				return nil, fmt.Errorf("cannot cast interface %v %T to int", dimSizeInter, dimSizeInter)
			}
			dimSize := int(dimSizeFloat)
			shape = append(shape, dimSize)
		}
		return shape, nil
	}
	k := keySeq[0]
	val, ok := stru[k]
	if !ok {
		return nil, fmt.Errorf("key %s not in struct", k)
	}
	switch v := reflect.ValueOf(val); v.Kind() {
	case reflect.Slice:
		shape, err := getShapeSlice(val.([]interface{}), keySeq[1:])
		if err != nil {
			return nil, err
		}
		return shape, nil
	case reflect.Map:
		shape, err := GetShape(val.(map[string]interface{}), keySeq[1:])
		if err != nil {
			return nil, err
		}
		return shape, nil
	default:
		return nil, fmt.Errorf("type %v not supported", v.Kind())
	}
}
