package tensor

import (
	"fmt"
)

func copyCatDim(t *Tensor, tens []*Tensor) *Tensor {
	tIdx := 0
	for _, ten := range tens {
		for tenIdx := 0; tenIdx < ten.Dims[0]; tenIdx++ {
			x := t.Sub(tIdx)
			y := ten.Sub(tenIdx)
			copy(x.Data, y.Data)
			tIdx += 1
		}

	}

	return t
}

func copyInto(t *Tensor, tens []*Tensor, dim int) *Tensor {
	if dim == 0 {
		return copyCatDim(t, tens)
	}
	for i := 0; i < t.Dims[0]; i++ {
		subTens := make([]*Tensor, len(tens))
		for j, ten := range tens {
			subTens[j] = ten.Sub(i)
		}
		x := t.Sub(i)
		copyInto(x, subTens, dim-1)
	}
	return t
}

func Concat(tens []*Tensor, dim int) (*Tensor, error) {
	l := len(tens[0].Dims)
	if l <= dim {
		return nil, fmt.Errorf("dim %d larger tensor rank %d", dim, l)
	}
	for tenDim := range tens[0].Dims {
		dtypeZero := tens[0].Dtype
		sizeZero := tens[0].Dims[tenDim]
		for i := 1; i < len(tens); i++ {
			if tenDim != dim {
				size := tens[i].Dims[tenDim]
				if sizeZero != size {
					return nil, fmt.Errorf("dimension tenDim is unequal for tensors")
				}
			}
			if tens[i].Dtype != dtypeZero {
				return nil, fmt.Errorf("dtype of tensor %d is unequal", i)
			}
		}
	}
	newDimSize := 0
	for _, t := range tens {
		newDimSize = newDimSize + t.Dims[dim]
	}
	newDims := make([]int, len(tens[0].Dims))
	copy(newDims, tens[0].Dims)
	newDims[dim] = newDimSize
	newTen := New(tens[0].Dtype, newDims...)
	newTen = copyInto(newTen, tens, dim)

	return newTen, nil
}
