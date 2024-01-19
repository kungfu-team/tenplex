package vfile

import "net/url"

// Range represents [Begin, End) where Begin <= End
type Range struct {
	Begin uint64
	End   uint64
}

func (r Range) Len() uint64 {
	return r.End - r.Begin
}

type NamedRange struct {
	Name  string
	Range Range
}

type IndexedRange struct {
	ID    int
	Range Range
}

type Ranges []Range

type NamedRanges []NamedRange

type IndexedFile struct {
	Filepath string
	Ranges   Ranges
}

func (f IndexedFile) IndexedBytes() uint64 {
	var n uint64
	for _, r := range f.Ranges {
		n += r.Len()
	}
	return n
}

type IndexedFiles []IndexedFile

func (i IndexedFiles) NumRange() int {
	var n int
	for _, f := range i {
		n += len(f.Ranges)
	}
	return n
}

func (i IndexedFiles) NamedRanges() NamedRanges {
	var rs NamedRanges
	for _, f := range i {
		for _, r := range f.Ranges {
			rs = append(rs, NamedRange{f.Filepath, r})
		}
	}
	return rs
}

func (rs NamedRanges) Select(s []int) NamedRanges {
	var qs NamedRanges
	for _, i := range s {
		qs = append(qs, rs[i])
	}
	return qs
}

func (idx IndexedFiles) SetHost(host string) {
	for i, f := range idx {
		u, err := url.Parse(f.Filepath)
		if err != nil {
			continue
		}
		u.Host = host
		if u.Host == `` {
			u.Scheme = ``
		}
		idx[i].Filepath = u.String()
	}
}
