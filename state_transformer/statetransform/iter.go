package statetransform

import (
	"strconv"

	"github.com/kungfu-team/tenplex/state_transformer/go/client"
	"github.com/kungfu-team/tenplex/state_transformer/go/meta"
)

func setIter(conf *meta.Config, targetDevice int, cl client.CheckpointClient) error {
	if targetDevice%conf.GpusPerHost != 0 { // only once per host
		return nil
	}

	err := cl.UploadValue([]byte(strconv.Itoa(conf.Step)), "iter", targetDevice, true)
	if err != nil {
		return err
	}

	return nil
}
