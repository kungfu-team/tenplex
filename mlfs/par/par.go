package par

import "sync"

type par struct {
	wg sync.WaitGroup
	ch chan struct{}
}

func New(m int) *par {
	p := &par{
		ch: make(chan struct{}, m),
	}
	return p
}

func (p *par) Do(f func()) {
	p.wg.Add(1)
	p.ch <- struct{}{}
	go func() {
		f()
		<-p.ch
		p.wg.Done()
	}()
}

func (p *par) Wait() {
	p.wg.Wait()
}
