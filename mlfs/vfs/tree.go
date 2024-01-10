package vfs

import (
	"errors"
	"fmt"
	"sync"
)

var (
	errNodeExists    = errors.New("node exists")
	errNodeNotExists = errors.New("node not exist")
	// errFileExists   = errors.New("file exists")
	errDirNotExists = errors.New("dir not exist")
	errNotFile      = errors.New("not a file")
	errNotDir       = errors.New("not a dir")
	errDirNotEmpty  = errors.New("dir not empty")
)

type inode int

const invalidINode inode = -1

type Tree struct {
	lastID inode
	nDirs  int
	id     map[pstr]inode
	ps     map[inode]pstr
	nodes  map[inode]Node

	mu sync.RWMutex
}

func New() *Tree {
	return &Tree{
		id:    make(map[pstr]inode),
		ps:    make(map[inode]pstr),
		nodes: make(map[inode]Node),
	}
}

func (t *Tree) Count() int {
	return len(t.nodes)
}

func (t *Tree) add(s filepath, n Node) inode {
	t.lastID += 1
	id := t.lastID
	t.ps[id] = s.P()
	t.nodes[id] = n
	t.id[s.P()] = id
	if n.IsDir() {
		t.nDirs += 1
	}
	return id
}

func (t *Tree) del(s filepath) inode {
	id := t.id[s.P()]
	delete(t.id, s.P())
	delete(t.ps, id)
	delete(t.nodes, id)
	return id
}

func (t *Tree) exists(s pstr) bool {
	_, ok := t.id[s]
	return ok
}

func (t *Tree) getParent(p filepath) (DirNode, error) {
	var d DirNode
	for i := 0; i < len(p); i++ {
		id, ok := t.id[p[:i].P()]
		if !ok {
			return nil, errDirNotExists
		}
		n := t.nodes[id]
		d = n.AsDir()
	}
	return d, nil
}

func (t *Tree) FullPath(id int) string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return string(t.ps[inode(id)])
}

func (t *Tree) Get(s string) (Node, int, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	// log.Printf("Tree::Get(%s)", s)
	p := ParseP(s)
	id, ok := t.id[p.P()]
	// log.Printf("Tree::Get(%s) -> (%d, %v)", s, id, ok)
	return t.nodes[id], int(id), ok
}

func (t *Tree) GetById(id int) Node {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.nodes[inode(id)]
}

func (t *Tree) touch(s string, n Node) (Node, inode, error) {
	p := ParseP(s)
	up, err := t.getParent(p)
	if err != nil {
		return nil, -1, err
	}
	if t.exists(p.P()) {
		return nil, -1, errNodeExists
	}
	id := t.add(p, n)
	if up != nil {
		up.Add(p.basename(), int(id), n.IsDir())
	}
	return n, id, nil
}

func (t *Tree) mkdir(s string) (*dir, int, error) {
	d := &dir{}
	if _, id, err := t.touch(s, d); err != nil {
		return nil, int(invalidINode), fmt.Errorf("mkdir %s %v", s, err)
	} else {
		return d, int(id), nil
	}
}

func (t *Tree) Mkdir(s string) (*dir, int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.mkdir(s)
}

func (t *Tree) Rmdir(s string) (int, error) {
	p := ParseP(s)
	t.mu.Lock()
	defer t.mu.Unlock()
	id, ok := t.id[p.P()]
	if !ok {
		return int(invalidINode), errNodeNotExists
	}
	n := t.nodes[id]
	if !n.IsDir() {
		return int(invalidINode), errNotDir
	}
	if len(n.AsDir().Items()) > 0 {
		return int(invalidINode), errDirNotEmpty
	}
	fmt.Printf("find up\n")
	if up, _ := t.getParent(p); up != nil {
		up.Del(int(id))
	} else {
		fmt.Printf("has no up\n")
	}
	return int(t.del(p)), nil
}

func (t *Tree) Rm(s string) (int, error) {
	p := ParseP(s)
	t.mu.Lock()
	defer t.mu.Unlock()
	id, ok := t.id[p.P()]
	if !ok {
		return int(invalidINode), errNodeNotExists
	}
	n := t.nodes[id]
	if n.IsDir() {
		return int(invalidINode), errNotFile
	}
	up, err := t.getParent(p)
	if err != nil {
		return int(invalidINode), err
	}
	up.Del(int(id))
	return int(t.del(p)), nil
}

func (t *Tree) MkdirAll(s string) error {
	p := ParseP(s)
	for i := 0; i <= len(p); i++ {
		pp := filepath(p[:i])
		t.Mkdir(string(pp.P())) // TODO: check conflict
		// don't return err here!
	}
	return nil
}

func (t *Tree) TouchText(s string, text string) (*file, error) {
	return t.TouchBytes(s, []byte(text))
}

func (t *Tree) TouchBytes(s string, bs []byte) (*file, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	f := &file{
		bs: bs,
	}
	if _, _, err := t.touch(s, f); err != nil {
		return nil, fmt.Errorf("TouchBytes %s %v", s, err)
	}
	return f, nil
}

func (t *Tree) TouchOrReplaceText(s string, text string) (*file, error) {
	return t.TouchOrReplaceBytes(s, []byte(text))
}

func (t *Tree) TouchOrReplaceBytes(s string, bs []byte) (*file, error) {
	if f, err := t.TouchBytes(s, bs); err == nil {
		return f, nil
	}
	if _, err := t.Rm(s); err != nil {
		return nil, err
	}
	return t.TouchBytes(s, bs)
}

func (t *Tree) TouchFile(s string, n FileNode) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	_, id, err := t.touch(s, &fileNode{f: n})
	if err != nil {
		return -1, fmt.Errorf("TouchFile %s %v", s, err)
	}
	return int(id), nil
}
