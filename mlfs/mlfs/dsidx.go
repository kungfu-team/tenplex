package mlfs

import (
	"image"
	"image/color"

	"github.com/kungfu-team/mlfs/vfs/vfile"
)

type DSIDX struct {
	idx          vfile.IndexedFiles
	ridx         []int
	lidx         []int
	maxRegions   int
	totalRegions int
}

func newDSIDX(idx vfile.IndexedFiles) *DSIDX {
	var maxRegions, totalRegions int
	var ridx, lidx []int
	for i, f := range idx {
		n := len(f.Ranges)
		if n > maxRegions {
			maxRegions = n
		}
		totalRegions += n
		for j := 0; j < n; j++ {
			ridx = append(ridx, i)
			lidx = append(lidx, j)
		}
	}
	d := &DSIDX{
		idx:          idx,
		ridx:         ridx,
		lidx:         lidx,
		maxRegions:   maxRegions,
		totalRegions: totalRegions,
	}
	log.Printf("maxRegions: %d", maxRegions)
	return d
}

func (d *DSIDX) bmap(ids []int) image.Image {
	img := makeBitmap(len(d.idx), d.maxRegions)
	for _, id := range ids {
		i := d.ridx[id]
		j := d.lidx[id]
		img.Set(j, i, color.White)
	}
	return img
}
