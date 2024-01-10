package iotrace

import (
	golog "log"
	"os"
	"sync"
	"time"
)

var log = golog.New(os.Stderr, `[mlfs] io % `, 0)

type reporter struct {
	stopped chan struct{}
	wg      sync.WaitGroup
}

func Reporter(c *Counter, prefix string) *reporter {
	r := &reporter{stopped: make(chan struct{}, 1)}
	r.wg.Add(1)
	go func() {
		for {
			select {
			case <-r.stopped:
				log.Printf("%soverall rate: %s", prefix, c.ShowRate())
				r.wg.Done()
				return
			default:
				time.Sleep(1 * time.Second)
				log.Printf("%s%s", prefix, c.ShowRate())
			}
		}

	}()
	return r
}

func (r *reporter) Stop() {
	r.stopped <- struct{}{}
	r.wg.Wait()
}

func Monitor(c *Counter, prefix string) {
	r := &reporter{stopped: make(chan struct{}, 1)}
	r.wg.Add(1)
	go func() {
		for {
			time.Sleep(1 * time.Second)
			if !c.Zero() {
				log.Printf("%s%s", prefix, c.ShowRate())
				c.Reset()
			}
		}
	}()
}
