package vfs

import (
	"fmt"
	"io"
)

func (t *Tree) Dump(w io.Writer) {
	for i, p := range t.ps {
		n := t.nodes[i]
		var c rune
		var size int
		var unit string
		if n.IsDir() {
			c = 'd'
			size = len(n.AsDir().Items())
			unit = `files`
		} else {
			c = '-'
			size = int(n.AsFile().Size())
			unit = `bytes`
		}
		fmt.Fprintf(w, "%8d    %c    %12d    %s    %s\n", i, c, size, unit, p)
	}
	fmt.Fprintf(w, "%d nodes\n", t.Count())
}

func (t *Tree) Stat() {
	n := t.Count()
	nd := t.nDirs
	nf := n - nd
	fmt.Printf("%d nodes, %d dirs, %d files\n", n, nd, nf)
}

func (t *Tree) AllFiles(w io.Writer) {
	for i, p := range t.ps {
		n := t.nodes[i]
		if !n.IsDir() {
			fmt.Fprintf(w, "%s\n", p)
		}
	}
}
