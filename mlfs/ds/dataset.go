package ds

type Dataset struct {
	Name     string `json:"name"`
	IndexURL string `json:"index-url"`
	Size     int    `json:"size"` // Total number of samples
}

var (
	SQuAD1Test = Dataset{
		Name:     `squad1-test`,
		IndexURL: `https://minddata.blob.core.windows.net/data/squad1/squad1.idx.txt`,
	}

	Imagenet = Dataset{
		Name:     `imagenet`,
		IndexURL: `https://minddata.blob.core.windows.net/data/imagenet.idx.txt`,
	}
)
