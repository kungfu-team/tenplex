package meta

import (
	"encoding/json"
	"os"
	"path"
	"strconv"
)

type MDPRank struct {
	PPRank int
	MPRank int
	DPRank int
}

type RankMap struct {
	Rank    map[MDPRank]int
	MDPRank map[int]MDPRank
}

func CreateRankMap(config *Config, before bool) (*RankMap, error) {
	structPath := GetStructPath(config, before)
	jsonPath := path.Join(structPath, "rank_map.json")
	content, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, err
	}
	var payload map[string]map[string]int
	err = json.Unmarshal(content, &payload)
	if err != nil {
		return nil, err
	}

	ranks := make(map[MDPRank]int)
	MDPRanks := make(map[int]MDPRank)
	for r, val := range payload {
		rInt, err := strconv.Atoi(r)
		if err != nil {
			return nil, err
		}
		mdpRank := MDPRank{PPRank: val["pp_rank"], MPRank: val["mp_rank"], DPRank: val["dp_rank"]}
		MDPRanks[rInt] = mdpRank
		ranks[mdpRank] = rInt
	}
	rankMap := RankMap{Rank: ranks, MDPRank: MDPRanks}
	return &rankMap, nil
}
