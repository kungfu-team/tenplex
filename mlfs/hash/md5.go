package hash

import (
	"crypto/md5"
	"fmt"
	"io"
	"os"

	"github.com/kungfu-team/mlfs/iotrace"
)

type md5db struct {
	hashToPath map[string]string
	pathToHash map[string]string
}

func NewMD5DB() *md5db {
	db := &md5db{
		hashToPath: make(map[string]string),
		pathToHash: make(map[string]string),
	}
	return db
}

// func (db*md5db)

func FileMD5(c *iotrace.Counter, filename string) (string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return md5sum(f, c)
}

func md5sum(r io.Reader, c *iotrace.Counter) (string, error) {
	h := md5.New()
	if _, err := io.Copy(h, iotrace.TraceReader(r, c)); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
