package search

import (
	"fmt"
	"os"
	"path"
	"strings"
)

func isTensor(name string) (bool, error) {
	split := strings.SplitN(name, ".", 2)
	if len(split) != 2 {
		return false, fmt.Errorf("string split has not exactly 2 parts")
	}
	return split[1] == "numpy.ndarray", nil
}

func SearchFSForTensors(basePath string) ([]string, error) {
	dirEntries, err := os.ReadDir(basePath)
	if err != nil {
		return nil, err
	}

	var tensors []string
	for _, dirEntry := range dirEntries {
		info, err := dirEntry.Info()
		if err != nil {
			return nil, err
		}

		if dirEntry.IsDir() {
			newTensors, err := SearchFSForTensors(path.Join(basePath, info.Name()))
			tensors = append(tensors, newTensors...)
			if err != nil {
				return nil, err
			}
		} else { // isFile
			isTen, err := isTensor(info.Name())
			if err != nil {
				return nil, err
			}
			if isTen {
				tensors = append(tensors, path.Join(basePath, info.Name()))
			}
		}
	}
	return tensors, nil
}
