package iseq

import (
	"math/rand"
)

type ISeq struct {
	seq []int
}

func Seq(s []int) ISeq {
	return ISeq{s}
}

func (is *ISeq) Empty() bool {
	return len(is.seq) == 0
}

func (is *ISeq) Take(n int) ISeq {
	a, b := split(n, is.seq)
	is.seq = b
	return ISeq{a}
}

func (is *ISeq) Shard(i, m int) ISeq {
	k := ceilDiv(len(is.seq), m)
	a := i * k
	b := min(a+k, len(is.seq))
	return ISeq{seq: is.seq[a:b]}
}

func (is *ISeq) Len() int {
	return len(is.seq)
}

func (is *ISeq) Get() []int {
	return is.seq[:]
}

func Iota(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}

func Shuffle(s []int, seed int) {
	rand.Seed(int64(seed))
	rand.Shuffle(len(s), func(i, j int) {
		s[i], s[j] = s[j], s[i]
	})
}

func split(n int, s []int) ([]int, []int) {
	if n >= len(s) {
		return s, nil
	}
	return s[:n], s[n:]
}

func ceilDiv(a, b int) int {
	if a%b == 0 {
		return a / b
	}
	return a/b + 1
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
