package tensor

import (
	"fmt"
	"reflect"
	"strings"
	"unsafe"
)

type Tensor struct {
	Dtype string
	Dims  []int
	Data  []byte
}

func NewOn(dtype string, dims []int, device string) *Tensor {
	return New(dtype, dims...)
}

func NewWith(dtype string, dims []int, data []byte) *Tensor {
	if len(data) != dsize(dtype)*prod(dims) {
		panic(`invalid tensor data size`)
	}
	return &Tensor{
		Dtype: dtype,
		Dims:  dims,
		Data:  data,
	}
}

func New(dtype string, dims ...int) *Tensor {
	// defer tr.Trace(fmt.Sprintf("tensor.New(...)")).Done()
	return NewWith(dtype, dims, make([]byte, dsize(dtype)*prod(dims)))
}

func (t *Tensor) Flatten() *Tensor {
	return &Tensor{
		Dtype: t.Dtype,
		Dims:  []int{prod(t.Dims)},
		Data:  t.Data,
	}
}

func (t *Tensor) Slice(i, j int) *Tensor {
	s := len(t.Data) / t.Dims[0]
	bgn := s * i
	end := s * j
	return &Tensor{
		Dtype: t.Dtype,
		Dims:  append([]int{j - i}, t.Dims[1:]...),
		Data:  t.Data[bgn:end],
	}
}

func (t *Tensor) Range(s ...Slicer) *Tensor {
	r := completeRange(Range(s), t)
	// log.Printf("%s [%s]", t, r)
	return r.Of(t)
}

func (t *Tensor) Sub(i int) *Tensor {
	s := len(t.Data) / t.Dims[0]
	bgn := s * i
	end := s * (i + 1)
	return &Tensor{
		Dtype: t.Dtype,
		Dims:  t.Dims[1:],
		Data:  t.Data[bgn:end],
	}
}

func (t *Tensor) sliceHeader() unsafe.Pointer {
	count := prod(t.Dims)
	sh := &reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&t.Data[0])),
		Len:  count,
		Cap:  count,
	}
	return unsafe.Pointer(sh)
}

func prod(dims []int) int {
	n := 1
	for _, x := range dims {
		n *= x
	}
	return n
}

func dsize(dt string) int {
	switch dt {
	case `i8`, `u8`:
		return 1
	case `i16`, `u16`, `f16`:
		return 2
	case `i32`, `u32`, `f32`:
		return 4
	case `i64`, `u64`, `f64`:
		return 8
	}
	panic(`invalid dt`)
}

func (v *Tensor) String() string { return v.Dtype + `(` + join(`,`, fmap(str[int], v.Dims)...) + `)` }

func join(s string, ss ...string) string { return strings.Join(ss, s) }

func fmap[X any, Y any](f func(x X) Y, xs []X) []Y {
	var ys []Y
	for _, x := range xs {
		ys = append(ys, f(x))
	}
	return ys
}

func str[T any](i T) string { return fmt.Sprintf("%v", i) }

type Slicer struct{ i, j int }

func Slice(i, j int) Slicer { return Slicer{i: i, j: j} }

func (s Slicer) Of(x *Tensor) *Tensor { return x.Slice(s.i, s.j) }

type Range []Slicer

func (r Range) copy(y, x *Tensor) {
	if len(y.Dims) == 0 {
		return
	}
	if len(y.Dims) == 1 {
		s := len(x.Data) / x.Dims[0]
		bgn := r[0].i * s
		end := r[0].j * s
		copy(y.Data, x.Data[bgn:end])
		return
	}
	r0, r1 := r[0], r[1:]
	for i := 0; i < y.Dims[0]; i++ {
		r1.copy(y.Sub(i), x.Sub(r0.i+i))
	}
}

func (r Range) Of(x *Tensor) *Tensor {
	if len(r) != len(x.Dims) {
		panic(`inconsistent rank`)
	}
	dims := make([]int, len(r))
	for i, s := range r {
		dims[i] = s.j - s.i
	}
	y := New(x.Dtype, dims...)
	r.copy(y, x)
	return y
}

func completeRange(r Range, x *Tensor) Range {
	dims := x.Dims
	for i := len(r); i < len(dims); i++ {
		r = append(r, Slice(0, dims[i]))
	}
	for i, d := range dims {
		if r[i].i < 0 {
			r[i].i = 0
		}
		if r[i].j < 0 {
			r[i].j = d
		}
	}
	return r
}

func (s Slicer) String() string {
	var a, b string
	if s.i >= 0 {
		a = str(s.i)
	}
	if s.j >= 0 {
		b = str(s.j)
	}
	return a + ":" + b
}

func (r Range) String() string {
	return join(`,`, fmap(func(s Slicer) string { return s.String() }, r)...)
}
