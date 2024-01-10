package mlfs

import (
	"bytes"
	"fmt"

	"github.com/kungfu-team/mlfs/vfs"
	"github.com/kungfu-team/tensor"
)

type Tensor = tensor.Tensor

func (e *MLFS) TouchTensor(p string, t *Tensor) error {
	log.Printf("TouchTensor: %q", p)
	if _, err := e.tree.TouchText(p+`.meta`, func() string {
		bs := &bytes.Buffer{}
		fmt.Fprintf(bs, "%s\n", t.Dtype)
		dims := t.Dims
		fmt.Fprintf(bs, "%d\n", len(dims))
		for _, d := range dims {
			fmt.Fprintf(bs, "%d\n", d)
		}
		return bs.String()
	}()); err != nil {
		log.Printf("TouchTensor meta: %q", p)
		return err
	}
	// TODO: support write large bytes to read files instead of RAM
	if _, err := e.tree.TouchFile(p, vfs.ToFile(t.Data)); err != nil {
		log.Printf("TouchTensor data: %q", p)
		return err
	}
	return nil
}
