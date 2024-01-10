package vfile

import (
	"io"
	"log"

	"github.com/kungfu-team/mlfs/cache"
)

type vfile struct {
	ranges []NamedRange
	cache  *cache.Cache
}

func VFile(ranges []NamedRange, cache *cache.Cache) *vfile {
	vf := &vfile{
		ranges: ranges,
		cache:  cache,
	}
	return vf
}

func (f *vfile) NumRanges() int { return len(f.ranges) }

func (f *vfile) Open() io.ReadCloser {
	r := io.NewSectionReader(f, 0, f.Size())
	return io.NopCloser(r)
}

func (f *vfile) Size() int64 {
	var n uint64
	for _, r := range f.ranges {
		n += r.Range.Len()
	}
	// log.Printf("vfile::Size() -> %d", n)
	return int64(n)
}

func (f *vfile) ReadAt(buf []byte, pos int64) (int, error) {
	// log.Printf("vfile::ReadAt(%d, [%d])", pos, len(buf))
	skip := pos
	var got int
	// t0 := time.Now()
	// var reading bool
	for _, r := range f.ranges { // TODO : binary search
		if skip >= int64(r.Range.Len()) {
			skip -= int64(r.Range.Len())
			continue
		}
		// log.Printf("buf: %d", len(buf))
		// if !reading {
		// 	log.Printf("scan took %s", time.Since(t0))
		// }
		// reading = true
		f, err := f.cache.OpenRange(r.Name, int64(r.Range.Begin)+skip, int64(r.Range.End))
		if err != nil {
			log.Printf("r.OpenAt(%d): %v", skip, err)
			return 0, err
		}
		n, err := readToEnd(f, buf)
		f.Close()
		// log.Printf("readToEnd([%d]): %d, %v", len(buf), n, err)
		got += n
		buf = buf[n:]
		if err != nil {
			if err == io.EOF {
				panic("readToEnd can't return io.EOF")
			}
			return got, err
		}
		skip = 0
		if len(buf) == 0 {
			// log.Printf("vfile::ReadAt(%d) -> ok(%d)", pos, got)
			return got, nil
		}
	}
	// log.Printf("%d not read", len(buf))
	return got, io.EOF // buf can be longer that data available
}

func readToEnd(f io.Reader, buf []byte) (int, error) {
	var tot int
	for {
		if len(buf) == 0 {
			break
		}
		n, err := f.Read(buf)
		buf = buf[n:]
		tot += n
		if err == io.EOF {
			break
		}
		if err != nil {
			return tot, err
		}
	}
	return tot, nil
}
