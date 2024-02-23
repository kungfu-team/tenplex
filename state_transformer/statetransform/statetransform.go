package statetransform

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex/state_transformer/client"
	"github.com/kungfu-team/tenplex/state_transformer/lib"
	"github.com/kungfu-team/tenplex/state_transformer/megatronlm"
	"github.com/kungfu-team/tenplex/state_transformer/meta"
	"github.com/kungfu-team/tenplex/state_transformer/search"
	"golang.org/x/sync/errgroup"
)

func createGroup() *errgroup.Group {
	g, _ := errgroup.WithContext(context.TODO())
	g.SetLimit(8)
	return g
}

func addListMeta(conf *meta.Config, ckptClient *client.CheckpointClient, targetStru map[string]interface{}, mdpRank *meta.MDPRank) error {
	lists, err := search.SearchJsonForLists(targetStru, []string{})
	if err != nil {
		return err
	}

	for key, length := range lists {
		metaPath := path.Join(key, "dir.meta")
		data := []byte(fmt.Sprintf("list\n%d", length))
		err = ckptClient.UploadMegatronValue(data, mdpRank, conf.OutputTimestamp, metaPath)
		if err != nil {
			return err
		}
	}

	return nil
}

func setNonTensor(conf *meta.Config, ckptClient *client.CheckpointClient, nonTensor []string, weightDecayTens [][]string, targetMDPRank *meta.MDPRank) error {
	sourcePPRank := rand.Intn(conf.SourcePPDegree)
	sourceMPRank := rand.Intn(conf.SourceMPDegree)
	// sourceDPRank := rand.Intn(conf.SourceDPDegree)
	sourceDPRank := 0 // With Megatron-LM there are no replicated ckpts

	if len(nonTensor) >= 6 &&
		equal(nonTensor[:3], []string{"optimizer", "optimizer", "param_groups"}) &&
		isInt(nonTensor[3]) &&
		nonTensor[4] == "params" &&
		isInt(nonTensor[5]) { // Megatron-LM

		group, err := strconv.Atoi(nonTensor[3])
		if err != nil {
			return err
		}
		indexInGroup, err := strconv.Atoi(nonTensor[5])
		if err != nil {
			return err
		}
		offset := 0
		if group == 1 {
			offset = len(weightDecayTens)
		}
		newIndex := indexInGroup + offset
		data := []byte(strconv.Itoa(newIndex))
		targetPath := strings.Join(nonTensor, "/")
		targetPath = fmt.Sprintf("%s.%s", targetPath, "int")
		err = ckptClient.UploadMegatronValue(data, targetMDPRank, conf.OutputTimestamp, targetPath)
		if err != nil {
			return err
		}
		return nil
	}

	sourceMDPRank := meta.MDPRank{PPRank: sourcePPRank, MPRank: sourceMPRank, DPRank: sourceDPRank}
	sourcePath := strings.Join(nonTensor, "/")

	if strings.Contains(sourcePath, "np_rng_state/1") { // HACK
		dtype := "numpy.ndarray"
		hackSourcePath := fmt.Sprintf("%s.%s", sourcePath, dtype)
		data, err := ckptClient.QueryMegatronTensor(&sourceMDPRank, conf.InputTimestamp, hackSourcePath, nil)
		if err != nil {
			return err
		}

		targetPath := strings.Join(nonTensor, "/")
		targetPath = fmt.Sprintf("%s.%s", targetPath, dtype)
		err = ckptClient.UploadMegatronTensor(data, targetMDPRank, conf.OutputTimestamp, targetPath)
		if err != nil {
			return err
		}
		return nil
	}

	data, dtype, err := ckptClient.QueryMegatronValue(&sourceMDPRank, conf.InputTimestamp, sourcePath)
	if strings.HasSuffix(dtype, "meta") {
		return err
	}
	if strings.Contains(dtype, "numpy.ndarray") {
		return err
	}
	if err != nil {
		return err
	}

	targetPath := strings.Join(nonTensor, "/")
	targetPath = fmt.Sprintf("%s.%s", targetPath, dtype)
	err = ckptClient.UploadMegatronValue(data, targetMDPRank, conf.OutputTimestamp, targetPath)
	if err != nil {
		return err
	}

	return nil
}

func setNonTensors(conf *meta.Config, ckptClient *client.CheckpointClient, nonTensors [][]string, weightDecayTens [][]string, targetMDPRank *meta.MDPRank) error {
	g := createGroup()
	for _, nonTensor := range nonTensors {
		nt := nonTensor
		g.Go(func() error {
			return setNonTensor(conf, ckptClient, nt, weightDecayTens, targetMDPRank)
		})
	}
	return g.Wait()
}

func transformTensor(conf *meta.Config, ckptClient *client.CheckpointClient, targetKey []string, targetStru map[string]interface{}, targetMDPRank *meta.MDPRank, weightDecayTens, noWeightDecayTens [][]string, metadata *meta.Metadata) error {
	targetShape, err := meta.GetShape(targetStru, targetKey)
	if err != nil {
		return err
	}
	var sourcePPRank int
	var sourceKey []string
	if lib.InSlice("rng_state", targetKey) { // Megatron-LM
		sourcePPRank = 0
		sourceKey = targetKey
	} else {
		sourcePPRank, sourceKey, err = megatronlm.InferSourcePPRank(conf, targetShape, targetKey, conf.NumLayers, targetMDPRank, weightDecayTens, noWeightDecayTens, metadata)
		if err != nil {
			return err
		}
	}
	sourceMDPRank := meta.MDPRank{PPRank: sourcePPRank, MPRank: 0, DPRank: 0}
	sourceRank, ok := metadata.SourceRankMap.Rank[sourceMDPRank]
	if !ok {
		return fmt.Errorf("cannot get source rank for mdp rank %v and source key %v", sourceMDPRank, sourceKey)
	}
	sourceMP0Stru, ok := metadata.SourceStructs[sourceRank]
	if !ok {
		return fmt.Errorf("cannot get source MP0 structure for source rank %d", sourceRank)
	}
	sourceShape, err := meta.GetShape(sourceMP0Stru, sourceKey)
	if err != nil {
		// DEBUG
		log.Panicf("GetShape failed for %v with %s", sourceKey, err)
		return err
	}

	if equal(sourceShape, targetShape) {
		sourceMPRank := 0
		if conf.TargetMPDegree == conf.SourceMPDegree { // MP size does not change
			sourceMPRank = targetMDPRank.MPRank
		}
		sourceDPRank := 0
		sourceMDPRank := meta.MDPRank{PPRank: sourcePPRank, MPRank: sourceMPRank, DPRank: sourceDPRank}
		err = replicateTensor(conf, ckptClient, sourceKey, targetKey, &sourceMDPRank, targetMDPRank)
		if err != nil {
			return err
		}
	} else {
		if len(sourceShape) != len(targetShape) {
			return err
		}
		err = repartitionTensor(conf, ckptClient, metadata.SourceRankMap, sourceShape, targetShape, targetMDPRank, sourcePPRank, &sourceKey, &targetKey)
		if err != nil {
			return err
		}
	}

	return nil
}

func loadMetadata(conf *meta.Config) (*meta.Metadata, error) {
	sourceRankMap, err := meta.CreateRankMap(conf, true)
	if err != nil {
		return nil, err
	}
	targetRankMap, err := meta.CreateRankMap(conf, false)
	if err != nil {
		return nil, err
	}
	sourceStructs, err := meta.LoadStructs(conf, sourceRankMap, true)
	if err != nil {
		return nil, err
	}
	targetStructs, err := meta.LoadStructs(conf, targetRankMap, false)
	if err != nil {
		return nil, err
	}
	sourceGroupSizes, err := megatronlm.GetGroupSizes(conf, true, sourceRankMap, sourceStructs)
	if err != nil {
		return nil, err
	}
	targetGroupSizes, err := megatronlm.GetGroupSizes(conf, false, targetRankMap, targetStructs)
	if err != nil {
		return nil, err
	}
	sourceModelKeys, err := meta.LoadModelKeys(conf, true)
	if err != nil {
		return nil, err
	}
	targetModelKeys, err := meta.LoadModelKeys(conf, false)
	if err != nil {
		return nil, err
	}

	metadata := meta.Metadata{
		SourceRankMap:    sourceRankMap,
		TargetRankMap:    targetRankMap,
		SourceStructs:    sourceStructs,
		TargetStructs:    targetStructs,
		SourceGroupSizes: sourceGroupSizes,
		TargetGroupSizes: targetGroupSizes,
		SourceModelKeys:  sourceModelKeys,
		TargetModelKeys:  targetModelKeys,
	}
	return &metadata, nil
}

func MigrateState(conf *meta.Config, targetDevice int) error {
	log.Printf("START MigrateState")

	metadata, err := loadMetadata(conf)
	if err != nil {
		return err
	}

	targetMDPRank, ok := metadata.TargetRankMap.MDPRank[targetDevice]
	if !ok {
		return fmt.Errorf("cannot get target MDP rank for device %d", targetDevice)
	}
	targetMdpRankDPZero := meta.MDPRank{PPRank: targetMDPRank.PPRank, MPRank: targetMDPRank.MPRank, DPRank: 0}
	targetDeviceDPZero := metadata.TargetRankMap.Rank[targetMdpRankDPZero] // Megatron-LM DP=0
	targetStru, ok := metadata.TargetStructs[targetDeviceDPZero]           // Megatron-LM DP=0
	if !ok {
		return fmt.Errorf("cannot get target struct for device %d", targetDevice)
	}

	ckptClient := client.New(conf, metadata.SourceRankMap, metadata.TargetRankMap)

	// Clean load directory
	_, err = ckptClient.QueryTargetDir(targetDevice, conf.OutputTimestamp)
	if err == nil {
		err = ckptClient.DeleteTargetDir(targetDevice, conf.OutputTimestamp)
		if err != nil {
			return err
		}
	}

	tensors, nonTensors, err := search.SearchJsonForTensors(targetStru, []string{})
	if err != nil {
		return err
	}
	log.Printf("number of tensors: %d, number of non-tensors: %d", len(tensors), len(nonTensors))

	weightDecayTens, noWeightDecayTens, err := megatronlm.SplitWeightDecay(metadata.TargetModelKeys[targetDeviceDPZero], targetStru) // Megatron-LM DP=0
	if err != nil {
		return err
	}

	startTransTensor := time.Now()
	g := createGroup()
	for _, targetKey := range tensors {
		tk := targetKey
		g.Go(func() error {
			return transformTensor(conf, &ckptClient, tk, targetStru, &targetMDPRank, weightDecayTens, noWeightDecayTens, metadata)
		})
	}
	err = g.Wait()
	if err != nil {
		return err
	}
	log.Printf("Transform tensors took %s", time.Since(startTransTensor))

	startTransNonTensor := time.Now()
	err = setNonTensors(conf, &ckptClient, nonTensors, weightDecayTens, &targetMDPRank)
	if err != nil {
		return err
	}
	log.Printf("Transform non tensors took %s", time.Since(startTransNonTensor))

	err = addListMeta(conf, &ckptClient, targetStru, &targetMDPRank)
	if err != nil {
		return err
	}

	err = setIter(conf, targetDevice, ckptClient)
	if err != nil {
		return err
	}

	log.Printf("FINISHED MigrateState for device %d", targetDevice)
	return nil
}
