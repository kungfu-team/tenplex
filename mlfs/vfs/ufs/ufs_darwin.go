package ufs

import (
	"log"

	"github.com/kungfu-team/tenplex/mlfs/vfs"
)

func Umount(mnt string) {
	log.Printf("TODO: support FUSE Umount on darwin")
}

func Start(mnt string, r *vfs.Tree, super bool) {
	log.Printf("TODO: support FUSE Mount on darwin")
}
