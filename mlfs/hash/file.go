package hash

import (
	"fmt"
	"log"

	"github.com/kungfu-team/tenplex/mlfs/uri"
)

type HashedFile struct {
	MD5  string
	URLs []string
}

func (f *HashedFile) Check() {
	for _, u := range f.URLs {
		ok, got, err := md5Check(f.MD5, u)
		if err != nil {
			log.Printf("%v", err)
			continue
		}
		if !ok {
			fmt.Printf("failed: %s != md5(%s) = %s\n", f.MD5, u, got)
			continue
		}
		fmt.Printf("OK: %s = md5(%s)\n", f.MD5, u)
	}
}

func md5Check(sum string, url string) (bool, string, error) {
	f, err := uri.Open(url)
	if err != nil {
		return false, "", err
	}
	defer f.Close()
	got, err := md5sum(f, nil)
	if err != nil {
		return false, got, err
	}
	return sum == got, got, nil
}
