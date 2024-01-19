package vfile

import "github.com/kungfu-team/tenplex/mlfs/iseq"

func (f IndexedFiles) Shard(i, n int) *vfile {
	seq := iseq.Seq(iseq.Iota(f.NumRange()))
	seq = seq.Shard(i, n)
	return &vfile{
		ranges: f.NamedRanges().Select(seq.Get()),
	}
}
