package vfs

func InitExample(r *Tree) {
	r.Mkdir(`/`)
	r.Mkdir(`/a`)
	r.Mkdir(`/a/b`)
	r.TouchText(`/a/b/c.txt`, "hello world\n")
}
