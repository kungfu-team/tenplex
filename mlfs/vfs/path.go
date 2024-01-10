package vfs

import (
	"errors"
	"strings"
)

type pstr string

type filepath []string

func (p filepath) P() pstr {
	s := `/` + strings.Join(p, `/`)
	return pstr(s)
}

var errNoParent = errors.New(`root has no parent`)

func (p filepath) parent() filepath {
	if len(p) < 1 {
		panic(errNoParent)
	}
	return p[:len(p)-1]
}

func (p filepath) basename() string {
	if len(p) < 1 {
		panic(errNoParent)
	}
	return p[len(p)-1]
}

func ParseP(p string) filepath {
	var parts []string
	for _, name := range strings.Split(p, `/`) {
		if len(name) > 0 {
			parts = append(parts, name)
		}
	}
	return parts
}
