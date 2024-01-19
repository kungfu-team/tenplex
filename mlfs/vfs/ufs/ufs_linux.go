package ufs

import (
	"context"
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"syscall"
	"time"

	"bazil.org/fuse"
	"bazil.org/fuse/fs"
	"github.com/kungfu-team/tenplex/mlfs/vfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

func (f *FS) Root() (fs.Node, error) {
	// log.Printf("FS::Root()")
	n, id, ok := f.r.Get(`/`)
	if !ok {
		return nil, syscall.ENOENT
	}
	d := n.AsDir()
	if d == nil {
		return nil, syscall.ENOENT
	}
	return &Dir{
		fs: f,
		r:  f.r,
		n:  d,
		id: id,
	}, nil
}

func (d *Dir) Attr(ctx context.Context, a *fuse.Attr) error {
	// log.Printf("Dir<%d>::Attr(...)", d.id)
	a.Inode = uint64(d.id)
	a.Mode = os.ModeDir | 0o777
	return nil
}

func (d *Dir) Lookup(ctx context.Context, name string) (fs.Node, error) {
	// log.Printf("Dir<%d>::Lookup(%s)", d.id, name)
	// p := d.r.FullPath(d.id)
	// log.Printf("found parent %d %s", d.id, p)
	for _, i := range d.n.Items() {
		if i.Name == name {
			f := d.r.GetById(i.Id)
			if f == nil {
				return nil, syscall.ENOENT
			}
			if i.IsDir {
				return &Dir{
					fs: d.fs,
					r:  d.r,
					id: i.Id,
					n:  f.AsDir(),
				}, nil
			} else {
				return &File{
					fs:   d.fs,
					id:   i.Id,
					n:    f.AsFile(),
					name: name,
				}, nil
			}
		}
	}
	// log.Printf("Dir<%d>::Lookup(%s) -> Not Found", d.id, name)
	return nil, syscall.ENOENT
}

func (d *Dir) Mkdir(ctx context.Context, req *fuse.MkdirRequest) (fs.Node, error) {
	if !d.fs.allowWrite {
		return nil, errReadOnly
	}
	p := path.Join(d.r.FullPath(d.id), req.Name)
	f, id, err := d.r.Mkdir(p)
	if err != nil {
		return nil, err
	}
	return &Dir{
		fs: d.fs,
		r:  d.r,
		id: id,
		n:  f.AsDir(),
	}, nil
}

func (d *Dir) Create(ctx context.Context, req *fuse.CreateRequest, resp *fuse.CreateResponse) (fs.Node, fs.Handle, error) {
	if !d.fs.allowWrite {
		return nil, nil, errReadOnly
	}
	p := path.Join(d.r.FullPath(d.id), req.Name)
	// log.Printf("Create() -> %s", p)
	n := vfile.NewBuffer()
	id, err := d.r.TouchFile(p, n)
	if err != nil {
		return nil, nil, err
	}
	f := &File{
		fs: d.fs,
		id: id,
		n:  n,
	}
	return f, f, nil
}

func (d *Dir) ReadDirAll(ctx context.Context) ([]fuse.Dirent, error) {
	var items []fuse.Dirent
	for _, i := range d.n.Items() {
		it := fuse.Dirent{
			Inode: uint64(i.Id),
			Name:  i.Name,
		}
		if i.IsDir {
			// log.Printf("ReadDirAll: %s dir", i.Name)
			it.Type = fuse.DT_Dir
		} else {
			// log.Printf("ReadDirAll: %s file", i.Name)
			it.Type = fuse.DT_File
		}
		items = append(items, it)
		// log.Printf("ReadDirAll: %s", i.Name)
	}
	// log.Printf("ReadDirAll: %d", len(items))
	return items, nil
}

func (f *File) Attr(ctx context.Context, a *fuse.Attr) error {
	a.Inode = uint64(f.id)
	a.Mode = 0o444
	if m, ok := f.n.(vfs.FileMode); ok && m.IsExecutable() {
		a.Mode = 0o777
	}
	a.Size = uint64(f.n.Size())
	return nil
}

/*
	func (f *File) ReadAll(ctx context.Context) ([]byte, error) {
		log.Printf("File<%s>::ReadAll()[%d]", f.name, f.n.Size())
		r := f.n.Open()
		defer r.Close()
		return io.ReadAll(r)
	}
*/
func (f *File) Open(ctx context.Context, req *fuse.OpenRequest, resp *fuse.OpenResponse) (fs.Handle, error) {
	// log.Printf("File<%d:%s>::Open(...)", f.id, f.name)
	if req.Flags.IsWriteOnly() {
		if b, ok := f.n.(*vfile.Buffer); ok {
			b.Truncate()
		}
	}
	return f, nil
}

// func (f *File) Setattr(ctx context.Context, req *fuse.SetattrRequest, resp *fuse.SetattrResponse) error {
// 	log.Printf("File<%s>::Setattr(...)", f.name)
// 	if b, ok := f.n.(*vfile.Buffer); ok {
// 		if req.Size == 0 {
// 			b.Truncate()
// 		}
// 	}
// 	return nil
// }

func (f *File) Read(ctx context.Context, req *fuse.ReadRequest, resp *fuse.ReadResponse) error {
	// log.Printf("File<%s>::Read(%d, [%d/%d])", f.name, req.Size, len(resp.Data), cap(resp.Data)) // 131072
	buf := resp.Data[:req.Size]
	n, err := f.n.ReadAt(buf, req.Offset)
	resp.Data = buf[:n]
	if err == io.EOF {
		return nil
	}
	if err != nil {
		f.fs.log.Printf("File::Read(offset=%d): %v", req.Offset, err)
	}
	return err
}

func (f *File) Write(ctx context.Context, req *fuse.WriteRequest, resp *fuse.WriteResponse) error {
	// log.Printf("File<%d:%s>::write(%d[%d])", f.id, f.name, req.Offset, len(req.Data))
	if b, ok := f.n.(*vfile.Buffer); ok {
		b.WriteAt(req.Data, req.Offset)
		resp.Size = len(req.Data)
	}
	return nil
}

func Umount(root string) {
	t0 := time.Now()
	c := exec.Command(`fusermount`, `-u`, root)
	c.Start()
	c.Wait()
	log.Printf("umounted %s, took %s", root, time.Since(t0))
}

const mi = 1 << 20

func Start(root string, r *vfs.Tree, super bool) {
	opts := []fuse.MountOption{
		fuse.FSName("elastique"),
		fuse.Subtype("tfrecord"),
		fuse.MaxReadahead(16 * mi),
	}
	if super {
		opts = append(opts, fuse.AllowOther())
	}
	c, err := fuse.Mount(root, opts...)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	defer Umount(root)
	if err = fs.Serve(c, New(r)); err != nil {
		log.Fatal(err)
	}
}
