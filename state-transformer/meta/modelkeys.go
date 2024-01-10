package meta

import (
	"encoding/json"
	"os"
	"path"
	"strconv"
)

func LoadModelKeys(conf *Config, before bool) (map[int][][]string, error) {
	structPath := GetStructPath(conf, before)
	modelKeysPath := path.Join(structPath, "model_keys.json")
	var payload map[string][][]string
	content, err := os.ReadFile(modelKeysPath)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(content, &payload)
	if err != nil {
		return nil, err
	}
	modelKeys := make(map[int][][]string)
	for k, v := range payload {
		i, err := strconv.Atoi(k)
		if err != nil {
			return nil, err
		}
		modelKeys[i] = v
	}
	return modelKeys, nil
}
