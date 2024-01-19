package ds

import "github.com/kungfu-team/tenplex/mlfs/hash"

// https://minddata.blob.core.windows.net/data/squad1/squad1.md5.txt

var (
	Squad1Index = hash.HashedFile{
		MD5:  `57015fef3d187f14a57a55ff04166e0c`,
		URLs: []string{`https://minddata.blob.core.windows.net/data/squad1/squad1.idx.txt`},
	}

	Squad1MD5 = hash.HashedFile{
		MD5:  `9e1ed608ed476e8fed2fbf84ff378884`,
		URLs: []string{`https://minddata.blob.core.windows.net/data/squad1/squad1.md5.txt`},
	}
)
