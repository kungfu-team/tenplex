package trds

import (
	"bytes"

	"github.com/kungfu-team/mlfs/vfs"
	"github.com/kungfu-team/mlfs/vfs/vfile"
)

func InitExample(r *vfs.Tree) {
	r.Mkdir(`/`)
	idx, err := vfile.LoadIdxFile(`a.idx.txt`)
	if err != nil {
		return
	}
	bs := &bytes.Buffer{}
	vfile.SaveIdx(bs, idx)
	r.TouchText(`/index.txt`, bs.String())
}
