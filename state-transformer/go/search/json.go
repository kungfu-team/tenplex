package search

import (
	"log"
	"path"
	"reflect"
	"strconv"
	"strings"
)

type Counters struct {
	Slices   int
	Maps     int
	Defaults int
}

func (c *Counters) Print(prefix string, total int) {
	log.Printf("%s Number of total %d, slices %d, maps %d, defaults %d", prefix, total, c.Slices, c.Maps, c.Defaults)

}

func searchJsonForTensorsSlice(structure []interface{}, keySeq []string) ([][]string, [][]string, error) {
	var tensors [][]string
	var nonTensors [][]string
	var c Counters
	// defer c.Print(strings.Join(keySeq, "/"), len(structure))
	for idx, val := range structure {
		key := strconv.Itoa(idx)
		switch v := reflect.ValueOf(val); v.Kind() {
		case reflect.Slice:
			c.Slices++
			newTensors, newNonTensors, err := searchJsonForTensorsSlice(val.([]interface{}), append(keySeq, key))
			if err != nil {
				return nil, nil, err
			}
			tensors = append(tensors, newTensors...)
			nonTensors = append(nonTensors, newNonTensors...)
		case reflect.Map:
			c.Maps++
			newTensors, newNonTensors, err := SearchJsonForTensors(val.(map[string]interface{}), append(keySeq, key))
			if err != nil {
				return nil, nil, err
			}
			tensors = append(tensors, newTensors...)
			nonTensors = append(nonTensors, newNonTensors...)
		default:
			c.Defaults++
			keys := make([]string, len(keySeq)+1)
			copy(keys, keySeq)
			keys[len(keys)-1] = key
			nonTensors = append(nonTensors, keys)
		}
	}
	return tensors, nonTensors, nil
}

func SearchJsonForTensors(structure map[string]interface{}, keySeq []string) ([][]string, [][]string, error) {
	var tensors [][]string
	var nonTensors [][]string
	for key, val := range structure {
		if key == "tensor" {
			keys := make([]string, len(keySeq))
			copy(keys, keySeq)
			tensors = append(tensors, keys)
			continue
		}
		if strings.Contains(key, ".meta") {
			log.Printf("key contains .meta %s", key)
			continue
		}
		switch v := reflect.ValueOf(val); v.Kind() {
		case reflect.Slice:
			newTensors, newNonTensors, err := searchJsonForTensorsSlice(val.([]interface{}), append(keySeq, key))
			if err != nil {
				return nil, nil, err
			}
			tensors = append(tensors, newTensors...)
			nonTensors = append(nonTensors, newNonTensors...)
		case reflect.Map:
			newTensors, newNonTensors, err := SearchJsonForTensors(val.(map[string]interface{}), append(keySeq, key))
			if err != nil {
				return nil, nil, err
			}
			tensors = append(tensors, newTensors...)
			nonTensors = append(nonTensors, newNonTensors...)
		default:
			keys := make([]string, len(keySeq)+1)
			copy(keys, keySeq)
			keys[len(keys)-1] = key
			nonTensors = append(nonTensors, keys)
		}
	}
	return tensors, nonTensors, nil
}

func searchJsonForListsSlice(structure []interface{}, keySeq []string) (map[string]int, error) {
	lists := make(map[string]int)
	for idx, val := range structure {
		key := strconv.Itoa(idx)
		switch v := reflect.ValueOf(val); v.Kind() {
		case reflect.Slice:
			keys := strings.Join(keySeq, "/")
			keys = path.Join(keys, key)
			newVal := val.([]interface{})
			newList := map[string]int{keys: len(newVal)}
			lists = mergeMap(lists, newList)
			newLists, err := searchJsonForListsSlice(newVal, append(keySeq, key))
			if err != nil {
				return nil, err
			}
			lists = mergeMap(lists, newLists)
		case reflect.Map:
			newLists, err := SearchJsonForLists(val.(map[string]interface{}), append(keySeq, key))
			if err != nil {
				return nil, err
			}
			lists = mergeMap(lists, newLists)
		default:
		}
	}
	return lists, nil
}

func mergeMap(a, b map[string]int) map[string]int {
	for k, v := range b {
		a[k] = v
	}
	return a
}

func SearchJsonForLists(structure map[string]interface{}, keySeq []string) (map[string]int, error) {
	lists := make(map[string]int)
	for key, val := range structure {
		if key == "tensor" {
			continue
		}
		switch v := reflect.ValueOf(val); v.Kind() {
		case reflect.Slice:
			keys := strings.Join(keySeq, "/")
			keys = path.Join(keys, key)
			newVal := val.([]interface{})
			newList := map[string]int{keys: len(newVal)}
			lists = mergeMap(lists, newList)
			newLists, err := searchJsonForListsSlice(newVal, append(keySeq, key))
			if err != nil {
				return nil, err
			}
			lists = mergeMap(lists, newLists)
		case reflect.Map:
			newLists, err := SearchJsonForLists(val.(map[string]interface{}), append(keySeq, key))
			if err != nil {
				return nil, err
			}
			lists = mergeMap(lists, newLists)
		default:
		}
	}
	return lists, nil
}
