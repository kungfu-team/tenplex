package fuse

import (
	"log"
	"os"
)

type FUSE struct {
	mnt string
	ch  chan struct{}
	dev *os.File
}

func New(mnt string) (*FUSE, error) {
	dev, err := os.Open(`/dev/fuse`)
	if err != nil {
		return nil, err
	}
	f := &FUSE{
		mnt: mnt,
		ch:  make(chan struct{}),
		dev: dev,
	}
	return f, nil
}

func (f *FUSE) Run() {
	for {
		buf := make([]byte, 1024)
		n, err := f.dev.Read(buf)
		log.Printf("%d,%v", n, err)
		_ = <-f.ch
	}
}
