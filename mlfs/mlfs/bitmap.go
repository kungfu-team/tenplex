package mlfs

import (
	"image"
	"image/color"
)

type BitVec struct{}

func makeBitmap(h, w int) *image.RGBA {
	r := image.Rect(0, 0, w, h)
	img := image.NewRGBA(r)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			img.Set(j, i, color.Black)
		}
	}
	return img
}
