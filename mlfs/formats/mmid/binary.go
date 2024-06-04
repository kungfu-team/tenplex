package mmid

import (
	"encoding/binary"
	"io"
	"log"
)

var endian = binary.LittleEndian

func readT[T any](r io.Reader, x *T) error { return binary.Read(r, endian, x) }

var (
	readU8  = readT[uint8]
	readU32 = readT[uint32]
	readU64 = readT[uint64]
)

func readArray(r io.Reader, dt uint8, n int) {
	switch dtypes[dt] {
	case `u16`:
		readArrayT[uint16](r, n)
	default:
		log.Panicf("unsupported dtype: %d", dt)
	}
}

func readArrayT[T any](r io.Reader, n int) []T {
	a := make([]T, n)
	if err := binary.Read(r, endian, a); err != nil {
		log.Printf("read error: %v", err)
	}
	log.Printf("got %d", n)
	// for _, x := range a {
	// 	fmt.Printf("%v ", x)
	// }
	// fmt.Printf("\n")
	return a
}
