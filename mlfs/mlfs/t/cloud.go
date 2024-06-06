package t

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"strconv"
	"sync"

	"github.com/kungfu-team/tenplex/mlfs/formats/tfrecord"
	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/hfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

type Cloud struct {
	Port           int
	NFiles         int
	RecordsPerFile int
	SizePerRecord  int
}

func (c *Cloud) Run(ctx context.Context) {
	tree := vfs.New()
	filenames, _ := c.genFakeData(tree)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		s := hfs.HS(tree)
		log.Printf("http://%s:%d/", `127.0.0.1`, c.Port)
		addr := net.JoinHostPort("", str(c.Port))
		hs := http.Server{Addr: addr, Handler: s}
		go func() {
			<-ctx.Done()
			hs.Close()
		}()
		hs.ListenAndServe()
		wg.Done()
	}()
	mlfs.WaitTCP(``, c.Port)
	genFakeIndex(tree, filenames)
	wg.Wait()
}

func genName(i int) string {
	const a int = 'a'
	var cs []rune
	for i > 0 {
		j := i % 26
		i /= 26
		cs = append(cs, rune(a+j))
	}
	return string(cs)
}

func genNames(n int) []string {
	var names []string
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("%04d", i+1)
		names = append(names, name) //genName(i+1))
	}
	return names
}

func (c Cloud) genFakeData(tree *vfs.Tree) ([]string, error) {
	prefix := fmt.Sprintf("http://127.0.0.1:%d", c.Port)
	tree.MkdirAll(`/mlfs/test`)
	var filenames []string
	f := func(name string, a vfs.FileNode) {
		path := fmt.Sprintf(`/mlfs/test/%s.tfrecord`, name)
		if _, err := tree.TouchFile(path, a); err != nil {
			log.Printf("%v", err)
			return
		}
		filenames = append(filenames, prefix+path)
	}
	for _, name := range genNames(c.NFiles) {
		f(name, newFakeTFRecord(repeat(c.SizePerRecord, c.RecordsPerFile)...))
	}
	return filenames, nil
}

func genFakeIndex(tree *vfs.Tree, filenames []string) error {
	idx, err := tfrecord.BuildIndex(filenames, 1)
	if err != nil {
		return err
	}
	buf := &bytes.Buffer{}
	if err := vfile.SaveIdx(buf, idx); err != nil {
		return err
	}
	tree.TouchFile(`/mlfs/test/a.idx`, vfs.ToFile(buf.Bytes()))
	return nil
}

func (c Cloud) ServeHTTP(w http.ResponseWriter, r *http.Request) {}

var str = strconv.Itoa

func newFakeTFRecord(sizes ...int) vfs.FileNode {
	buf := &bytes.Buffer{}
	for _, n := range sizes {
		bs := make([]byte, n)
		tfrecord.WriteTFRecord(bs, buf)
	}
	return vfs.ToFile(buf.Bytes())
}

func repeat[T any](x T, n int) []T {
	var xs []T
	for i := 0; i < n; i++ {
		xs = append(xs, x)
	}
	return xs
}
