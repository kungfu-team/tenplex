package ds

import "github.com/kungfu-team/mlfs/hash"

var (
	ImagenetIndex = hash.HashedFile{
		MD5:  `dfe57e9541f8cb7affedefd3c633326e`,
		URLs: []string{`https://minddata.blob.core.windows.net/data/imagenet.idx.txt`},
	}

	ImagenetMd5 = hash.HashedFile{
		MD5:  `91d0846314a61c32f42726aaa05ea9e7`,
		URLs: []string{`https://minddata.blob.core.windows.net/data/imagenet/md5sum.txt`},
	}
)
