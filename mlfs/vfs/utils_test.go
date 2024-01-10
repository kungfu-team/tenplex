package vfs_test

import (
	"testing"

	"github.com/kungfu-team/mlfs/vfs"
)

func Test_RmRecursive(t *testing.T) {
	r := vfs.New()
	script := `
	mkdir /
	mkdir /a
	mkdir /a/b
	mkdir /a/c
	touch /a/b/x.txt
	touch /a/b/y.txt
	touch /a/c/z.txt
	`
	if err := runScript(r, script); err != nil {
		t.Fail()
	}
	nf, nd, err := vfs.RmRecursive(r, `/a`)
	if err != nil {
		t.Fail()
	}
	if nf != 3 {
		t.Fail()
	}
	if nd != 3 {
		t.Fail()
	}
}
