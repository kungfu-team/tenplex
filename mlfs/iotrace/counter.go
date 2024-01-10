package iotrace

import (
	"fmt"
	"sync/atomic"
	"time"
)

type Counter struct {
	t0 time.Time
	n  int64
}

func NewCounter() *Counter {
	return &Counter{
		t0: time.Now(),
	}
}

func (c *Counter) Zero() bool {
	return atomic.LoadInt64(&c.n) == 0
}

func (c *Counter) Add(n int64) {
	atomic.AddInt64(&c.n, n)
}

func (c *Counter) Reset() {
	c.t0 = time.Now()
	atomic.StoreInt64(&c.n, 0)
}

func (c *Counter) ShowRate() string {
	n := atomic.LoadInt64(&c.n)
	return ShowRate(Rate(n, time.Since(c.t0)))
}

func Rate(n int64, d time.Duration) float64 {
	return float64(n) / (float64(d) / float64(time.Second))
}

func ShowRate(r float64) string {
	const Ki = 1 << 10
	const Mi = 1 << 20
	const Gi = 1 << 30
	switch {
	case r > Gi:
		return fmt.Sprintf("%.2f GiB/s", r/float64(Gi))
	case r > Mi:
		return fmt.Sprintf("%.2f MiB/s", r/float64(Mi))
	case r > Ki:
		return fmt.Sprintf("%.2f KiB/s", r/float64(Ki))
	default:
		return fmt.Sprintf("%.2f B/s", r)
	}
}
