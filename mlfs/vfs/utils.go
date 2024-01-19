package vfs

import (
	"io"
	"path"
)

func ReadFile(t *Tree, filename string) ([]byte, error) {
	n, _, ok := t.Get(filename)
	if !ok {
		return nil, errNodeNotExists
	}
	f := n.AsFile()
	if f == nil {
		return nil, errNotFile
	}
	bs := make([]byte, f.Size())
	r := f.Open()
	io.ReadFull(r, bs)
	return bs, nil
}

func ReadDir(t *Tree, dirname string) ([]string, error) {
	n, _, ok := t.Get(dirname)
	if !ok {
		return nil, errNodeNotExists
	}
	d := n.AsDir()
	if d == nil {
		return nil, errNotDir
	}
	items := d.Items()
	names := make([]string, len(items))
	for i, it := range items {
		names[i] = it.Name
	}
	return names, nil
}

// RmRecursive returns the number of files and directories if successful
func RmRecursive(t *Tree, dirname string) (int, int, error) {
	n, _, ok := t.Get(dirname)
	if !ok {
		return 0, 0, errNodeNotExists
	}

	if !n.IsDir() {
		if _, err := t.Rm(dirname); err == nil {
			return 1, 0, nil
		} else {
			return 0, 0, err
		}
	}

	var totFiles, totDirs int
	items := n.AsDir().Items()
	names := make([]string, 0, len(items))
	for _, it := range items {
		names = append(names, it.Name)
	}
	for _, na := range names {
		cntFiles, cntDirs, err := RmRecursive(t, path.Join(dirname, na))
		totFiles += cntFiles
		totDirs += cntDirs
		if err != nil {
			return totFiles, totDirs, err
		}
	}
	if _, err := t.Rmdir(dirname); err != nil {
		return totFiles, totDirs, err
	}
	return totFiles, totDirs + 1, nil
}
