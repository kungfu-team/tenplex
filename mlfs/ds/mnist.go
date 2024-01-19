package ds

import "github.com/kungfu-team/tenplex/mlfs/hash"

var (
	MnistTrainImages = hash.HashedFile{
		MD5:  `f68b3c2dcbeaaa9fbdd348bbdeb94873`,
		URLs: []string{`https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz`},
	}
	MnistTrainLabels = hash.HashedFile{
		MD5:  `d53e105ee54ea40749a09fcbcd1e9432`,
		URLs: []string{`https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz`},
	}
	MnistTestImages = hash.HashedFile{
		MD5:  `9fb629c4189551a2d022fa330f9573f3`,
		URLs: []string{`https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz`},
	}
	MnistTestLabels = hash.HashedFile{
		MD5:  `ec29112dd5afa0611ce80d1b7f02629c`,
		URLs: []string{`https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz`},
	}
)
