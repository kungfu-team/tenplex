package ufs

import (
	"errors"
	"log"
	"os"

	"github.com/kungfu-team/tenplex/mlfs/vfs"
)

type FS struct {
	r          *vfs.Tree
	allowWrite bool
	log        *log.Logger
}

func New(r *vfs.Tree) *FS {
	return &FS{
		r:   r,
		log: log.New(os.Stderr, `[fuse] `, 0),
	}
}

type Dir struct {
	fs *FS
	r  *vfs.Tree
	id int
	n  vfs.DirNode
}

type File struct {
	fs *FS
	id int
	n  vfs.FileNode

	// debug
	name string
}

var errReadOnly = errors.New(`readonly`)
