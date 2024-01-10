package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/kungfu-team/mlfs/ds"
	"github.com/kungfu-team/mlfs/vfs/vfile"
)

var (
	dataset = flag.String(`ds`, ``, ``)
)

func main() {
	flag.Parse()
	var ds ds.Dataset
	panicErr(loadJSONFile(*dataset, &ds))
	log.Printf("%q", ds.IndexURL)
	i, err := vfile.LoadIdxFile(ds.IndexURL)
	if err != nil {
		panic(err)
	}
	i.SetHost(``)
	for _, f := range i {
		log.Printf("%q", f.Filepath)
	}
}

func loadJSONFile(filename string, i interface{}) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	return json.NewDecoder(f).Decode(i)
}
func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}
