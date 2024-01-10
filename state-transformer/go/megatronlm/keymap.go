package megatronlm

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/kungfu-team/state-migrator/go/lib"
	"github.com/kungfu-team/state-migrator/go/meta"
)

func layerInKeys(sl []string) (bool, string) {
	for _, s := range sl {
		if strings.Contains(s, "layers.") {
			return true, s
		}
	}
	return false, ""
}

func getLayerNumberSlice(key []string) (int, error) {
	var err error
	for _, k := range key {
		var num int
		num, err = getLayerNumber(k)
		if err == nil {
			return num, nil
		}

	}
	return -1, err
}

func getLayerNumber(key string) (int, error) {
	re := regexp.MustCompile(`layers\.(\d+)\.`)
	matched := re.FindStringSubmatch(key)
	if len(matched) < 2 {
		return -1, fmt.Errorf("key %s match %v does not have two groups", key, matched)
	}
	num, err := strconv.Atoi(matched[1])
	if err != nil {
		return -1, err
	}
	return num, nil
}

func layerNumToSourcePPRank(conf *meta.Config, numLayers, targetPPRank, targetLayerNum int) (int, error) {
	// TODO: check correctness
	targetNumLayers := numLayers / conf.TargetPPDegree
	globalLayerNum := targetLayerNum + targetPPRank*targetNumLayers
	sourceNumLayers := numLayers / conf.SourcePPDegree
	var sourcePPRank int
	for r := 0; r < conf.SourcePPDegree; r++ {
		if r*sourceNumLayers <= globalLayerNum && globalLayerNum < (r+1)*sourceNumLayers {
			sourcePPRank = r
			return sourcePPRank, nil
		}
	}
	return -1, fmt.Errorf("source PP rank not found")
}

func inferSourcePPRankByKey(conf *meta.Config, tenKey []string) (int, error) {
	if lib.InSlice("embedding", tenKey) {
		return 0, nil
	}
	if lib.InSlice("word_embeddings_for_head", tenKey) ||
		lib.InSlice("final_layernorm.weight", tenKey) ||
		lib.InSlice("final_layernorm.bias", tenKey) ||
		lib.InSlice("pooler", tenKey) ||
		lib.InSlice("lm_head", tenKey) ||
		lib.InSlice("binary_head", tenKey) {
		return conf.SourcePPDegree - 1, nil
	}
	return -1, fmt.Errorf("inferSourcePPRankByKey no match for key %v", tenKey)
}

func replaceLayerNum(conf *meta.Config, targetKey []string, targetLayerNum, numLayers, targetPPRank, sourcePPRank int) ([]string, error) {
	// TODO: check correctness
	sourceKey := make([]string, len(targetKey))
	copy(sourceKey, targetKey)
	targetNumLayers := numLayers / conf.TargetPPDegree
	globalLayerNum := targetLayerNum + targetPPRank*targetNumLayers
	sourceNumLayers := numLayers / conf.SourcePPDegree
	sourceLayerNum := globalLayerNum - sourcePPRank*sourceNumLayers
	for i, k := range sourceKey {
		if is, _ := layerInKeys([]string{k}); is {
			re := regexp.MustCompile(`layers\.(\d+)`)
			sourceKey[i] = re.ReplaceAllString(k, fmt.Sprintf("layers.%d", sourceLayerNum))
			return sourceKey, nil
		}
	}
	return nil, fmt.Errorf("did not replace layer number")
}

// func fixes the problem that with PP=1 there is no word_embeddings_for_head
func replaceWordEmbedPP0(targetKey []string) (int, []string, error) {
	sourceKey := make([]string, len(targetKey)+2)
	offset := 0
	for i, tk := range targetKey {
		if tk == "word_embeddings_for_head" {
			sourceKey[i] = "language_model"
			sourceKey[i+1] = "embedding"
			sourceKey[i+2] = "word_embeddings"
			offset = 2
		} else {
			sourceKey[i+offset] = targetKey[i]
		}
	}
	return 0, sourceKey, nil
}

func inferSourcePPRankModel(conf *meta.Config, targetShape []int, targetKey []string, numLayers, targetPPRank int) (int, []string, error) {
	var sourcePPRank int
	sourceKey := make([]string, len(targetKey))
	copy(sourceKey, targetKey)

	hasLayer, layerKey := layerInKeys(targetKey)
	if hasLayer {
		targetLayerNum, err := getLayerNumber(layerKey)
		if err != nil {
			return -1, sourceKey, err
		}
		sourcePPRank, err = layerNumToSourcePPRank(conf, numLayers, targetPPRank, targetLayerNum)
		if err != nil {
			return -1, sourceKey, err
		}
		sourceKey, err = replaceLayerNum(conf, targetKey, targetLayerNum, numLayers, targetPPRank, sourcePPRank)
		if err != nil {
			return -1, sourceKey, err
		}

		return sourcePPRank, sourceKey, nil
	} else {
		if lib.InSlice("word_embeddings_for_head", targetKey) {
			return replaceWordEmbedPP0(targetKey)
		}
		sourcePPRank, err := inferSourcePPRankByKey(conf, targetKey)
		if err != nil {
			return -1, sourceKey, err
		}

		return sourcePPRank, sourceKey, nil
	}
}

func calcSourceIndexState(indexInGroup, group, sourcePPRank, targetPPRank int, sourceGroupSizes, targetGroupSizes map[int]map[int]int) (int, error) {
	sourceIndex, err := calcSourceIndexDefault(indexInGroup, group, sourcePPRank, targetPPRank, sourceGroupSizes, targetGroupSizes)
	if err != nil {
		return -1, err
	}

	if group == 1 {
		sourceGroupZeroSize, ok := sourceGroupSizes[sourcePPRank][0]
		if !ok {
			return -1, fmt.Errorf("cannot get source group size for PP rank %d, group %d", sourcePPRank, 0)
		}
		sourceIndex = sourceIndex + sourceGroupZeroSize
	}

	return sourceIndex, nil
}

func calcSourceIndexDefault(targetIndex, group, sourcePPRank, targetPPRank int, sourceGroupSizes, targetGroupSizes map[int]map[int]int) (int, error) {
	globalIndex := targetIndex
	for prevRank := 0; prevRank < targetPPRank; prevRank++ {
		size, ok := targetGroupSizes[prevRank][group]
		if !ok {
			return -1, fmt.Errorf("cannot get target group size for PP rank %d group %d", prevRank, group)
		}
		globalIndex = globalIndex + size
	}
	totalSize := 0
	for prevRank := 0; prevRank < sourcePPRank; prevRank++ {
		size, ok := sourceGroupSizes[prevRank][group]
		if !ok {
			return -1, fmt.Errorf("cannot get source group size for PP rank %d, group %d", prevRank, group)
		}
		totalSize = totalSize + size
	}
	sourceIndex := globalIndex - totalSize
	if sourceIndex < 0 {
		return -1, fmt.Errorf("source index is smaller than 0. else")
	}

	return sourceIndex, nil
}

func getSourcePPRank(conf *meta.Config, ten []string, numLayers, targetPPRank int) (int, error) {
	hasLayer, _ := layerInKeys(ten)
	if hasLayer {
		targetLayerNum, err := getLayerNumberSlice(ten)
		if err != nil {
			return -1, err
		}
		sourcePPRank, err := layerNumToSourcePPRank(conf, numLayers, targetPPRank, targetLayerNum)
		if err != nil {
			return -1, err
		}
		return sourcePPRank, nil
	} else {
		var err error
		sourcePPRank, err := inferSourcePPRankByKey(conf, ten)
		if err != nil {
			return -1, err
		}
		return sourcePPRank, nil
	}
}

func createSourceKey(targetKey []string, sourceIndex int) []string {
	sourceKey := make([]string, len(targetKey))
	copy(sourceKey, targetKey)
	sourceKey[len(sourceKey)-1] = strconv.Itoa(sourceIndex)
	return sourceKey
}

func inferSourcePPRankOptimiser(conf *meta.Config, targetShape []int, targetKey []string, numLayers, targetPPRank int, weightDecay, noWeightDecay [][]string, metadata *meta.Metadata) (int, []string, error) {
	if lib.InSlice("state", targetKey) {
		targetIndex := -1
		for _, k := range targetKey {
			var err error
			targetIndex, err = strconv.Atoi(k)
			if err == nil {
				break
			}
		}
		if targetIndex == -1 {
			return -1, nil, fmt.Errorf("target index is nil")
		}

		targetGroupZeroSize, ok := metadata.TargetGroupSizes[targetPPRank][0]
		if !ok {
			return -1, nil, fmt.Errorf("cannot get target group size")
		}
		var indexInGroup int
		var group int
		var ten []string
		if targetIndex < targetGroupZeroSize {
			indexInGroup = targetIndex
			group = 0
			if indexInGroup < 0 || indexInGroup >= len(weightDecay) {
				return -1, nil, fmt.Errorf("index out of range for weightDecay")
			}
			ten = weightDecay[indexInGroup]
		} else {
			indexInGroup = targetIndex - targetGroupZeroSize
			group = 1
			ten = noWeightDecay[indexInGroup]
			if indexInGroup < 0 || indexInGroup >= len(noWeightDecay) {
				return -1, nil, fmt.Errorf("index out of range for noWeightDecay")
			}
		}

		sourcePPRank, err := getSourcePPRank(conf, ten, numLayers, targetPPRank)
		if err != nil {
			return -1, nil, err
		}

		sourceIndex, err := calcSourceIndexState(indexInGroup, group, sourcePPRank, targetPPRank, metadata.SourceGroupSizes, metadata.TargetGroupSizes)
		if err != nil {
			return -1, nil, err
		}
		if sourceIndex < 0 {
			return -1, nil, fmt.Errorf("source index is smaller than 0. state")
		}

		sourceKey := make([]string, len(targetKey))
		copy(sourceKey, targetKey)
		targetIndexStr := strconv.Itoa(targetIndex)
		for i, k := range sourceKey {
			if k == targetIndexStr {
				sourceKey[i] = strconv.Itoa(sourceIndex)
			}
		}

		return sourcePPRank, sourceKey, nil
	} else {
		group, err := strconv.Atoi(targetKey[len(targetKey)-2])
		if err != nil {
			return -1, nil, err
		}
		targetIndex, err := strconv.Atoi(targetKey[len(targetKey)-1])
		if err != nil {
			return -1, nil, err
		}
		var groupTens [][]string
		if group == 0 {
			groupTens = weightDecay
		} else if group == 1 {
			groupTens = noWeightDecay
		} else {
			return -1, nil, fmt.Errorf("group is larger 1 %d", group)
		}
		ten := groupTens[targetIndex]

		var sourcePPRank int
		hasLayer, _ := layerInKeys(ten)
		if hasLayer {
			targetLayerNum, err := getLayerNumberSlice(ten)
			if err != nil {
				return -1, nil, err
			}
			sourcePPRank, err = layerNumToSourcePPRank(conf, numLayers, targetPPRank, targetLayerNum)
			if err != nil {
				return -1, nil, err
			}
		} else if lib.InSlice("word_embeddings_for_head", ten) && conf.SourcePPDegree == 1 {
			// replace sourceIndex for word_embeddings_for_head with embedding
			allZero := meta.MDPRank{PPRank: 0, MPRank: 0, DPRank: 0}
			allZeroDevice := metadata.SourceRankMap.Rank[allZero]
			allZeroDeviceStru, ok := metadata.SourceStructs[allZeroDevice]
			if !ok {
				return -1, nil, fmt.Errorf("cannot get target struct for device %d", allZeroDevice)
			}
			sourceModelKeys, err := meta.LoadModelKeys(conf, true)
			if err != nil {
				return -1, nil, err
			}
			weightDecayTens, _, err := SplitWeightDecay(sourceModelKeys[allZeroDevice], allZeroDeviceStru)
			if err != nil {
				return -1, nil, err
			}

			replaceKeys := []string{"language_model", "embedding", "word_embeddings"}
			sourceIndex := -1
			for i, t := range weightDecayTens {
				if lib.IsSubSlice(replaceKeys, t) {
					sourceIndex = i
					break
				}
			}
			if sourceIndex == -1 {
				return -1, nil, fmt.Errorf("cannot find keys %v in group tensors", replaceKeys)
			}

			sourceKey := createSourceKey(targetKey, sourceIndex)
			return 0, sourceKey, nil
		} else {
			sourcePPRank, err = inferSourcePPRankByKey(conf, ten)
			if err != nil {
				return -1, nil, err
			}
		}

		sourceIndex, err := calcSourceIndexDefault(targetIndex, group, sourcePPRank, targetPPRank, metadata.SourceGroupSizes, metadata.TargetGroupSizes)
		if err != nil {
			return -1, nil, err
		}
		sourceKey := createSourceKey(targetKey, sourceIndex)
		return sourcePPRank, sourceKey, nil
	}
}

func InferSourcePPRank(conf *meta.Config, targetShape []int, targetKey []string, numLayers int, targetMDPRank *meta.MDPRank, weightDecay, noWeightDecay [][]string, metadata *meta.Metadata) (int, []string, error) {
	if len(targetShape) == 1 && targetShape[0] == 1 {
		return 0, targetKey, nil
	}

	if lib.InSlice("model", targetKey) {
		return inferSourcePPRankModel(conf, targetShape, targetKey, numLayers, targetMDPRank.PPRank)
	}

	if lib.InSlice("optimizer", targetKey) {
		return inferSourcePPRankOptimiser(conf, targetShape, targetKey, numLayers, targetMDPRank.PPRank, weightDecay, noWeightDecay, metadata)
	}

	return -1, nil, fmt.Errorf("InferSourcePPRank no match for key %v", targetKey)
}

func biasInKey(key []string) bool {
	for _, k := range key {
		if strings.Contains(k, "bias") {
			return true
		}
	}
	return false
}

func SplitWeightDecay(tens [][]string, stru map[string]interface{}) ([][]string, [][]string, error) {
	var weightDecay [][]string
	var noWeightDecay [][]string
	var modelTens [][]string
	for _, ten := range tens {
		if ten[0] == "model" {
			modelTens = append(modelTens, ten)
		}
	}

	for _, ten := range modelTens {
		shape, err := meta.GetShape(stru, ten)
		if err != nil {
			return nil, nil, err
		}
		if biasInKey(ten) || len(shape) == 1 {
			noWeightDecay = append(noWeightDecay, ten)
		} else {
			weightDecay = append(weightDecay, ten)
		}
	}
	return weightDecay, noWeightDecay, nil
}
