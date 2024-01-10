package vfs

import "io"

type Node interface {
	IsDir() bool

	AsFile() FileNode
	AsDir() DirNode
}

type DirNode interface {
	Items() []Item
	Add(string, int, bool)
	Del(id int)
}

type FileNode interface {
	io.ReaderAt

	Open() io.ReadCloser
	Size() int64
}

type FileMode interface {
	IsExecutable() bool
}

type fileNode struct {
	f FileNode
}

func (f *fileNode) IsDir() bool { return false }

func (f *fileNode) AsFile() FileNode { return f.f }

func (f *fileNode) AsDir() DirNode { return nil }
