package logging

import (
	"io"
	"log"
	"os"
	"path"
)

func SetupLogger(name string) {
	log.SetPrefix(`[` + name + `] `)
	log.SetFlags(0)
	r, w := io.Pipe()
	log.SetOutput(w)
	go func(r io.Reader) {
		logfile := path.Join(`logs`, name+`.log`)
		if err := os.MkdirAll(path.Dir(logfile), os.ModePerm); err != nil {
			log.Printf("create logdir failed: %v", err)
		}
		if lf, err := os.Create(logfile); err == nil {
			r = io.TeeReader(r, lf)
		}
		io.Copy(os.Stderr, r)
	}(r)
}
