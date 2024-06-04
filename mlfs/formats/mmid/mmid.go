package mmid

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

const MAGIC = "MMIDIDX\x00\x00"

var dtypes = map[uint8]string{
	1: `u8`,
	2: `i8`,
	3: `i16`,
	4: `i32`,
	5: `i64`,
	6: `f32`,
	7: `f64`,
	8: `u16`,
}

var dtypeSize = map[string]uint64{
	`u16`: 2,
}

type Header struct {
	version  uint64
	dtype    uint8
	count    uint64
	idxCount uint64
}

type MMIDX struct {
	Header
	Counts     []uint32
	Pointers   []uint64
	DocIndexes []uint64
}

func (idx *MMIDX) DataSize() uint64 {
	elemSize := dtypeSize[dtypes[idx.dtype]]

	var totSize uint64
	for _, s := range idx.Counts {
		totSize += uint64(s) * elemSize
	}
	return totSize
}

var (
	errInvalidMagic       = errors.New(`invalid magic`)
	errUnsupportedVersion = errors.New(`unsupported version`)
	errExtraBytesRemains  = errors.New(`extra bytes remains`)
)

func ReadMMIdx(r io.Reader) (*MMIDX, error) {
	idx, err := readMMapIdx(r)
	if err != nil {
		return nil, err
	}

	if false {
		fmt.Printf("%-12s %-12s %-8s\n", `size`, `ptr`, `idx`)
		for i := 0; i < int(idx.count); i++ {
			fmt.Printf("%-12d %-12d %-8d\n", idx.Counts[i], idx.Pointers[i], idx.DocIndexes[i])
		}
	}
	totSize := idx.DataSize()
	fmt.Printf("total size: %d\n", totSize)
	return idx, nil
}

/*
magic:    9 byte
version:  8 byte
dtype:    1 byte
length:   8 byte
count:    8 byte (length + 1)
size[]:   4 x length
ptr[]:    8 x length
docIdx[]: 8 x count
*/
func readMMapIdx(r io.Reader) (*MMIDX, error) {
	var hdr Header
	buf := make([]byte, len(MAGIC))
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	if string(buf) != MAGIC {
		return nil, errInvalidMagic
	}
	if err := readU64(r, &hdr.version); err != nil {
		return nil, err
	}
	if hdr.version != 1 {
		return nil, errUnsupportedVersion
	}
	if err := readU8(r, &hdr.dtype); err != nil {
		return nil, err
	}
	elemSize := dtypeSize[dtypes[hdr.dtype]]
	if err := readU64(r, &hdr.count); err != nil {
		return nil, err
	}
	if err := readU64(r, &hdr.idxCount); err != nil {
		return nil, err
	}

	fmt.Printf("version: %d, dtype: %d (%s), element size: %d\n", hdr.version, hdr.dtype, dtypes[hdr.dtype], elemSize)
	fmt.Printf("count: %d, idxCount: %d\n", hdr.count, hdr.idxCount)

	sizes := make([]uint32, hdr.count)
	pointers := make([]uint64, hdr.count)
	docIndexes := make([]uint64, hdr.idxCount)

	for i := range sizes {
		if err := readU32(r, &sizes[i]); err != nil {
			return nil, err
		}
	}
	for i := range pointers {
		if err := readU64(r, &pointers[i]); err != nil {
			return nil, err
		}
	}
	for i := range docIndexes {
		if err := readU64(r, &docIndexes[i]); err != nil {
			return nil, err
		}
	}

	bs, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if len(bs) != 0 {
		fmt.Printf("%d remains\n", len(bs))
		return nil, errExtraBytesRemains
	}
	idx := MMIDX{
		Header:     hdr,
		Counts:     sizes,
		Pointers:   pointers,
		DocIndexes: docIndexes,
	}
	return &idx, nil
}

func ReadMMFiles(name string) error {
	idx, err := ReadMMIdxFile(name)
	if err != nil {
		return err
	}
	binFile := strings.TrimSuffix(name, `.idx`) + `.bin`
	if err := ReadMMBinFile(binFile, idx); err != nil {
		return err
	}
	return nil
}

func ShowMMFiles(name string, vocab []string) error {
	idx, err := ReadMMIdxFile(name)
	if err != nil {
		return err
	}
	binFile := strings.TrimSuffix(name, `.idx`) + `.bin`
	if err := ShowMMBinFile(binFile, idx, vocab); err != nil {
		return err
	}
	return nil
}

func ReadMMIdxFile(filename string) (*MMIDX, error) {
	fmt.Printf("%s\n", filename)
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadMMIdx(f)
}

func ReadMMBinFile(filename string, idx *MMIDX) error {
	info, err := os.Stat(filename)
	if err != nil {
		return err
	}
	if size := info.Size(); size != int64(idx.DataSize()) {
		return fmt.Errorf("unexpected size: %d", size)
	}
	r, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, c := range idx.Counts {
		readArray(r, idx.dtype, int(c))
	}
	return nil
}

func ShowMMBinFile(filename string, idx *MMIDX, vocab []string) error {
	info, err := os.Stat(filename)
	if err != nil {
		return err
	}
	if size := info.Size(); size != int64(idx.DataSize()) {
		return fmt.Errorf("unexpected size: %d", size)
	}
	if dtypes[idx.dtype] != `u16` {
		return fmt.Errorf("unsupported dtype: %d", idx.Header.dtype)
	}
	r, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, c := range idx.Counts {
		// readArray(r, idx.Header.dtype, int(c))
		ids := readArrayT[uint16](r, int(c))
		text := reconstruct(ids, vocab)
		fmt.Printf("%s\n", text)
	}
	return nil
}

func reconstruct(ids []uint16, vocab []string) string {
	var s string
	for _, id := range ids {
		s += vocab[id]
	}
	return s
}
