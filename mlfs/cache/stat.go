package cache

import (
	"log"
	"sync/atomic"
)

type Stat struct {
	miss int64
	hit  int64
}

func (s *Stat) Hit() {
	atomic.AddInt64(&s.hit, 1)
}

func (s *Stat) Miss() {
	atomic.AddInt64(&s.miss, 1)
}

func (s *Stat) Log() {
	h := (atomic.LoadInt64(&s.hit))
	m := (atomic.LoadInt64(&s.miss))
	r := float32(m) / float32(h+m)
	log.Printf("miss rate: %.2f%% (%d / %d)", r*100.0, m, m+h)
}

var LogCache = false
