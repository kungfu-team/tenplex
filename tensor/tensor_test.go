package tensor_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/kungfu-team/tenplex/tensor"
)

func Test_1(t *testing.T) {
	/*
		0  1  2  3
		4  5  6  7
		8  9  10 11
		12 13 14 15
	*/
	x := tensor.New(`i32`, 4, 4)
	{
		x := tensor.I32(x)
		for i := range x {
			x[i] = int32(i)
		}
	}

	y := x.Range(tensor.Slice(1, 3), tensor.Slice(1, 3))
	{
		y := tensor.I32(y)
		for _, e := range y {
			fmt.Fprintf(os.Stderr, "%d\n", e)
		}
	}
}
