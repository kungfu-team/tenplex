package vfs

type Item struct {
	IsDir bool
	Name  string
	Id    int
}

type dir struct {
	items []Item
}

func (d *dir) IsDir() bool { return true }

func (d *dir) AsFile() FileNode { return nil }

func (d *dir) AsDir() DirNode { return d }

func (d *dir) Items() []Item { return d.items }

func (d *dir) Add(name string, id int, isdir bool) {
	d.items = append(d.items, Item{IsDir: isdir, Id: id, Name: name})
}

func (d *dir) Del(id int) {
	var j int
	for i := range d.items {
		if d.items[i].Id != id {
			d.items[j] = d.items[i]
			j++
		}
	}
	d.items = d.items[:j]
}
