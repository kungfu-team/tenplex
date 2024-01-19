package tensor

import "fmt"

func eq(a, b string) {
	if a != b {
		panic(fmt.Errorf("%s != %s", a, b))
	}
}

func to[R any](name string, t *Tensor) []R {
	eq(t.Dtype, name)
	return *(*[]R)(t.sliceHeader())
}

func to_[R any](name string) func(*Tensor) []R { return func(t *Tensor) []R { return to[R](name, t) } }

var (
	U8  = to_[uint8](`u8`)
	U32 = to_[uint32](`u32`)
	I8  = to_[int8](`i8`)
	I32 = to_[int32](`i32`)
	F32 = to_[float32](`f32`)
)

func Raw(t *Tensor) []byte { return t.Data }
