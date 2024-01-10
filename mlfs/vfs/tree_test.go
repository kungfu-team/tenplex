package vfs_test

import (
	"log"
	"os"
	"testing"

	"github.com/kungfu-team/mlfs/vfs"
)

func Test_1(t *testing.T) {
	r := vfs.New()
	r.Mkdir(`/`)
	r.Mkdir(`/a`)
	r.Mkdir(`/a/b`)
	_, err := r.TouchText(`/a/b/c.txt`, `hello world`)
	if err != nil {
		log.Printf("%v", err)
	}
	r.Dump(os.Stdout)
	assertIntEq(r.Count(), 4, t)
}

func assertIntEq(a, b int, t *testing.T) {
	if a != b {
		t.Errorf("%d != %d", a, b)
	}
}
func Test_2(t *testing.T) {
	r := vfs.New()
	r.Mkdir(`/`)
	if _, err := r.TouchText(`/a`, ``); err != nil {
		t.Fail()
	}
	if _, err := r.TouchText(`/a`, ``); err == nil {
		t.Fail()
	}
}

func Test_3(t *testing.T) {
	r := vfs.New()
	if _, _, err := r.Mkdir(`/`); err != nil {
		t.Fail()
	}
	if _, err := r.TouchText(`/a`, ``); err != nil {
		t.Fail()
	}
	if _, err := r.Rm(`/a`); err != nil {
		t.Fail()
	}
	if r.Count() != 1 {
		t.Fail()
	}
	if _, err := r.TouchText(`/a`, ``); err != nil {
		t.Fail()
	}
}

func Test_4(t *testing.T) {
	r := vfs.New()
	script := `
	mkdir /
	rmdir /
	`
	if err := runScript(r, script); err != nil {
		t.Fail()
	}
}

func Test_5(t *testing.T) {
	r := vfs.New()
	script := `
	mkdir /
	! mkdir /
	mkdir /a
	touch /a/a.txt
	touch /a.txt
	! touch /a/a.txt
	rm /a.txt
	rm /a/a.txt
	rmdir /a
	`
	if err := runScript(r, script); err != nil {
		t.Fail()
	}
}

func Test_6(t *testing.T) {
	r := vfs.New()
	r.MkdirAll(`/a/b`)
	if r.Count() != 3 {
		t.Fatalf("%d != 3", r.Count())
	}
}

func TestReplaceBytes(t *testing.T) {
	r := vfs.New()
	r.Mkdir(`/`)

	_, err := r.TouchBytes(`/iter`, []byte(`0`))
	if err != nil {
		log.Printf("%v", err)
	}
	bs, err := vfs.ReadFile(r, `/iter`)
	if err != nil {
		log.Printf("%v", err)
	}
	log.Printf("first read %s", string(bs))

	_, err = r.TouchOrReplaceBytes(`/iter`, []byte(`1`))
	if err != nil {
		log.Printf("%v", err)
	}
	bs, err = vfs.ReadFile(r, `/iter`)
	if err != nil {
		log.Printf("%v", err)
	}
	log.Printf("second read %s", string(bs))
}
