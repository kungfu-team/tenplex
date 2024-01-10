package statetransform

import (
	"fmt"
	"strings"

	"github.com/kungfu-team/tenplex/state_transformer/client"
	"github.com/kungfu-team/tenplex/state_transformer/meta"
	"github.com/kungfu-team/tenplex/tensor"
)

type Requests = map[int][]int

func mapToUnitedRequests(sourceDim, targetDim, sourceMPSize, targetMPRank int) (Requests, error) {
	reqs := Requests{}
	lowerBound := targetMPRank * targetDim
	upperBound := lowerBound + targetDim
	for sourceMPRank := 0; sourceMPRank < sourceMPSize; sourceMPRank++ {
		deviceLowerBound := sourceMPRank * sourceDim
		deviceUpperBound := deviceLowerBound + sourceDim
		if lowerBound < deviceLowerBound {
			// before
			if upperBound <= deviceLowerBound {
				// before
			} else if upperBound <= deviceUpperBound && upperBound > deviceLowerBound {
				// within
				reqs[sourceMPRank] = []int{deviceLowerBound, upperBound}
				continue
			} else {
				// after
				reqs[sourceMPRank] = []int{deviceLowerBound, deviceUpperBound}
				continue
			}
		} else if lowerBound >= deviceLowerBound && lowerBound < deviceUpperBound {
			// within
			if upperBound <= deviceLowerBound {
				// before
				return nil, fmt.Errorf("lower bound %d larger than upper bound %d", lowerBound, upperBound)
			} else if upperBound <= deviceUpperBound && upperBound > deviceLowerBound {
				// within
				reqs[sourceMPRank] = []int{lowerBound, upperBound}
				continue
			} else {
				// after
				reqs[sourceMPRank] = []int{lowerBound, deviceUpperBound}
				continue
			}
		} else {
			// after
		}
	}
	return reqs, nil
}

func mapToSourceRequests(reqs Requests, sourceDim int) Requests {
	newReqs := Requests{}
	for sourceMPRank, rang := range reqs {
		lowerOffset := rang[0] % sourceDim
		var lowerBound int
		if lowerOffset == 0 {
			lowerBound = lowerOffset
		} else {
			lowerBound = lowerOffset
		}

		upperOffset := rang[1] % sourceDim
		var upperBound int
		if upperOffset == 0 {
			upperBound = sourceDim
		} else {

			upperBound = upperOffset
		}

		newReqs[sourceMPRank] = []int{lowerBound, upperBound}
	}

	return newReqs
}

func mapRequests(sourceDim, targetDim, sourceMPSize, targetMPRank int) (Requests, error) {
	reqs, err := mapToUnitedRequests(sourceDim, targetDim, sourceMPSize, targetMPRank)
	if err != nil {
		return nil, err
	}

	reqs = mapToSourceRequests(reqs, sourceDim)

	return reqs, nil
}

func requestTensors(dim int, reqs *Requests, ckptCl *client.CheckpointClient, sourceKey *[]string, inputTimestamp string, srcRankMap *meta.RankMap, srcPPRank int) ([]*tensor.Tensor, error) {
	var tensors []*tensor.Tensor
	sourcePath := strings.Join(*sourceKey, "/")
	sourcePath = fmt.Sprintf("%s.numpy.ndarray", sourcePath)
	for sourceMPRank, rang := range *reqs {
		sourceMDPRank := meta.MDPRank{PPRank: srcPPRank, MPRank: sourceMPRank, DPRank: 0}
		slice := client.Slice{Range: rang, Dim: dim}
		ten, err := ckptCl.QueryMegatronTensor(&sourceMDPRank, inputTimestamp, sourcePath, &slice)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, ten)
	}

	return tensors, nil
}

func addPadding(inTen *tensor.Tensor, targetShape []int) (*tensor.Tensor, error) {
	numPads := targetShape[0] - inTen.Dims[0]
	var padTen *tensor.Tensor
	if len(inTen.Dims) > 1 {
		newDim := make([]int, len(inTen.Dims))
		copy(newDim, inTen.Dims)
		newDim[0] = numPads
		padTen = tensor.New(inTen.Dtype, newDim...)
	} else {
		padTen = tensor.New(inTen.Dtype, numPads)
	}
	ten, err := tensor.Concat([]*tensor.Tensor{inTen, padTen}, 0)
	if err != nil {
		return nil, err
	}

	if !equal(ten.Dims, targetShape) {
		return nil, fmt.Errorf("new padded tensor does not have target shape %v != %v", ten.Dims, targetShape)
	}

	return ten, nil
}

func repartitionTensor(conf *meta.Config, ckptCl *client.CheckpointClient, sourceRankMap *meta.RankMap, sourceShape, targetShape []int, targetMDPRank *meta.MDPRank, sourcePPRank int, sourceKey, targetKey *[]string) error {
	for dim, srcDimSize := range sourceShape {
		trgDimSize := targetShape[dim]
		if srcDimSize == trgDimSize { // Check for dimension that is unequal
			continue
		}

		reqs, err := mapRequests(srcDimSize, trgDimSize, conf.SourceMPDegree, targetMDPRank.MPRank)
		if err != nil {
			return err
		}

		tensors, err := requestTensors(dim, &reqs, ckptCl, sourceKey, conf.InputTimestamp, sourceRankMap, sourcePPRank)
		if err != nil {
			return err
		}
		ten, err := tensor.Concat(tensors, dim)
		if err != nil {
			return err
		}

		// TODO: recover better checking for padding and wrong transformations
		// if !equal(ten.Dims, targetShape) {
		// 	vocabSizePad := VocabSizePadding(conf.VocabSize, conf.TargetMPDegree)
		// 	deviceVocabSize := vocabSizePad / conf.TargetMPDegree
		// 	if targetShape[0] == deviceVocabSize {
		// 		ten, err = addPadding(ten, targetShape)
		// 		if err != nil {
		// 			return err
		// 		}
		// 	} else {
		// 		return fmt.Errorf("new tensor does not have target shape %v != %v", ten.Dims, targetShape)
		// 	}
		// }
		if !equal(ten.Dims, targetShape) {
			ten, err = addPadding(ten, targetShape)
			if err != nil {
				return err
			}
		}

		targetPath := strings.Join(*targetKey, "/")
		targetPath = fmt.Sprintf("%s.numpy.ndarray", targetPath)
		err = ckptCl.UploadMegatronTensor(ten, targetMDPRank, conf.OutputTimestamp, targetPath)
		if err != nil {
			return err
		}

		// err = logRepartitionTensor(conf, sourceShape, targetShape, targetDevice, targetMDPRank, sourcePPRank, sourceKey, targetKey, reqs)
		if err != nil {
			return err
		}

		return nil
	}

	return fmt.Errorf("all dimensions have equal size")
}
