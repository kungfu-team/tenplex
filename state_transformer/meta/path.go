package meta

import (
	"fmt"
	"path"
)

func GetStructPath(c *Config, before bool) string {
	suffix := func(pp, mp, dp int) string {
		return fmt.Sprintf("pp%02d/mp%02d/dp%02d", pp, mp, dp)
	}
	var sfx string
	if before {
		sfx = suffix(c.SourcePPDegree, c.SourceMPDegree, c.SourceDPDegree)
	} else {
		sfx = suffix(c.TargetPPDegree, c.TargetMPDegree, c.TargetDPDegree)
	}
	return path.Join(
		c.CkptStructDir,
		c.MdpLibrary,
		c.Precision,
		c.Model,
		c.ModelSize,
		sfx,
	)
}
