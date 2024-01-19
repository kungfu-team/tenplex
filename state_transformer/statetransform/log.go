package statetransform

import (
	"fmt"
	"log"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/kungfu-team/tenplex/state_transformer/meta"
)

func logReplicateTensor(conf *meta.Config, sourceKey, targetKey []string, sourceDevice int, sourceMDPRank *meta.MDPRank, targetDevice int, targetMDPRank *meta.MDPRank) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	p := path.Join(home, ".tenplex/training", conf.JobID, fmt.Sprintf("replicate-%02d.csv", targetDevice))

	_, err = os.Stat(p)
	if err != nil { // File does not exist
		f, err := os.Create(p)
		if err != nil {
			log.Printf("failed creating %s", p)
			return err
		}
		f.WriteString("source-key,target-key,source-device,source-mdp-rank,target-device,target-mdp-rank\n")
		f.Close()
	}

	f, err := os.OpenFile(p, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("failed opening %s", p)
		return err
	}
	defer f.Close()
	newLine := strings.Join(sourceKey, "-") + "," + strings.Join(targetKey, "-") + "," +
		strconv.Itoa(sourceDevice) + "," + fmt.Sprintf("pp%d-mp%d-dp%d", sourceMDPRank.PPRank, sourceMDPRank.MPRank, sourceMDPRank.DPRank) + "," +
		strconv.Itoa(targetDevice) + "," + fmt.Sprintf("pp%d-mp%d-dp%d", targetMDPRank.PPRank, targetMDPRank.MPRank, targetMDPRank.DPRank)
	_, err = fmt.Fprintln(f, newLine)
	if err != nil {
		return err
	}
	return nil
}

func logRepartitionTensor(conf *meta.Config, sourceShape, targetShape []int, targetDevice int, targetMDPRank *meta.MDPRank, sourcePPRank int, sourceKey, targetKey *[]string, reqs map[int][]int) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	p := path.Join(home, ".tenplex/training", conf.JobID, fmt.Sprintf("repartition-%02d.csv", targetDevice))

	_, err = os.Stat(p)
	if err != nil { // File does not exist
		f, err := os.Create(p)
		if err != nil {
			log.Printf("failed creating %s", p)
			return err
		}
		f.WriteString("source-shape,target-shape,target-device,target-mdp-rank,source-pp-rank,source-key,target-key,requests\n")
		f.Close()
	}

	f, err := os.OpenFile(p, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("failed opening %s", p)
		return err
	}
	defer f.Close()

	srcShapeStr := make([]string, len(sourceShape))
	tgtShapeStr := make([]string, len(targetShape))
	for i := range sourceShape {
		srcShapeStr[i] = strconv.Itoa(sourceShape[i])
		tgtShapeStr[i] = strconv.Itoa(targetShape[i])
	}
	newLine := strings.Join(srcShapeStr, "-") + "," +
		strings.Join(tgtShapeStr, "-") + "," +
		strconv.Itoa(targetDevice) + "," +
		fmt.Sprintf("pp%d-mp%d-dp%d", targetMDPRank.PPRank, targetMDPRank.MPRank, targetMDPRank.DPRank) + "," +
		strconv.Itoa(sourcePPRank) + "," +
		strings.Join(*sourceKey, "-") + "," +
		strings.Join(*targetKey, "-") + "," +
		fmt.Sprintf("%v", reqs)
	_, err = fmt.Fprintln(f, newLine)
	if err != nil {
		return err
	}
	return nil
}
