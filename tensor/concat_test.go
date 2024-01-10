package tensor_test

import (
	"log"
	"strings"
	"testing"

	"github.com/kungfu-team/tensor"
)

func TestConcat(test *testing.T) {
	y := tensor.New(`i32`, 2, 2, 2)
	{
		y := tensor.I32(y)
		for i := range y {
			y[i] = int32(i)
		}
	}
	z := tensor.New(`i32`, 2, 2, 2)
	{
		z := tensor.I32(z)
		start := 8
		for i := range z {
			z[i] = int32(i + start)
		}
	}
	tens := make([]*tensor.Tensor, 2)
	tens[0] = y
	tens[1] = z

	x, err := tensor.Concat(tens, 1)
	if err != nil {
		test.Logf("testConcat error %v", err)
	}

	test.Logf("y %v, dim %v", tensor.I32(y), y.Dims)
	test.Logf("z %v, dim %v", tensor.I32(z), z.Dims)
	test.Logf("x %v, dim %v", tensor.I32(x), x.Dims)
}

func TestConcat2(test *testing.T) {
	y := tensor.New(`i32`, 4)
	{
		y := tensor.I32(y)
		for i := range y {
			y[i] = int32(i)
		}
	}
	z := tensor.New(`i32`, 4)
	{
		z := tensor.I32(z)
		start := 4
		for i := range z {
			z[i] = int32(i + start)
		}
	}
	tens := make([]*tensor.Tensor, 2)
	tens[0] = y
	tens[1] = z

	x, err := tensor.Concat(tens, 0)
	if err != nil {
		test.Logf("testConcat error %v", err)
		return
	}

	test.Logf("y %v, dim %v", tensor.I32(y), y.Dims)
	test.Logf("z %v, dim %v", tensor.I32(z), z.Dims)
	test.Logf("x %v, dim %v", tensor.I32(x), x.Dims)
}

func TestPlayground(t *testing.T) {
	s := "layers.2.something"
	if strings.Contains(s, "layers.") {
		log.Println("yes")
	} else {
		log.Println("no")
	}
}
