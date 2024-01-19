package statetransform

import (
	"fmt"
	"log"
	"strings"

	"github.com/kungfu-team/tenplex/state_transformer/client"
	"github.com/kungfu-team/tenplex/state_transformer/meta"
)

func replicateTensor(conf *meta.Config, ckptCl *client.CheckpointClient, sourceKey, targetKey []string, sourceMDPRank, targetMDPRank *meta.MDPRank) error {
	sourcePath := strings.Join(sourceKey, "/")
	sourcePath = fmt.Sprintf("%s.numpy.ndarray", sourcePath)
	ten, err := ckptCl.QueryMegatronTensor(sourceMDPRank, conf.InputTimestamp, sourcePath, nil)
	if err != nil {
		log.Printf("query tensor to replicate failed.\nwith error %v.\nsource key %v, target key %v, source MDP rank %v, target MDP rank %v", err, sourceKey, targetKey, sourceMDPRank, targetMDPRank)
		return err
	}
	targetPath := strings.Join(targetKey, "/")
	targetPath = fmt.Sprintf("%s.numpy.ndarray", targetPath)
	err = ckptCl.UploadMegatronTensor(ten, targetMDPRank, conf.OutputTimestamp, targetPath)
	if err != nil {
		return err
	}

	if err != nil {
		return err
	}

	return nil
}
