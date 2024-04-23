package timeout

import (
	"context"
	"log"
	"sync/atomic"
	"time"
)

type timeout struct {
	done int32
}

func New(d time.Duration, cancel context.CancelFunc) *timeout {
	t := &timeout{}
	go func() {
		time.Sleep(d)
		done := atomic.LoadInt32(&t.done)
		if done != 0 {
			return
		}
		log.Printf("timeout adter %s", d)
		if cancel != nil {
			cancel()
		}
	}()
	return t
}

func (t *timeout) Done() {
	atomic.StoreInt32(&t.done, 1)
}
