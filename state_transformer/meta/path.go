package meta

import (
	"fmt"
	"path"
)

func GetStructPath(conf *Config, before bool) string {
	structPath := path.Join(conf.CkptStructDir,
		conf.MdpLibrary,
		conf.Model,
		conf.ModelSize)

	if before {
		structPath = path.Join(structPath,
			fmt.Sprintf("pp%02d/mp%02d/dp%02d",
				conf.SourcePPDegree,
				conf.SourceMPDegree,
				conf.SourceDPDegree))
	} else {
		structPath = path.Join(structPath,
			fmt.Sprintf("pp%02d/mp%02d/dp%02d",
				conf.TargetPPDegree,
				conf.TargetMPDegree,
				conf.TargetDPDegree))
	}
	return structPath
}
