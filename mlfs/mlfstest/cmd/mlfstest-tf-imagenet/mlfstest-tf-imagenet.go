package main

import (
	"flag"
	"log"
	"os"

	"github.com/kungfu-team/mlfs/ds"
	"github.com/kungfu-team/mlfs/vfs/vfile"
	"github.com/lgarithm/proc"
)

var home = os.Getenv(`HOME`)

const name = `mlfs`

var imagenet = ds.Dataset{
	Name:     `imagenet`,
	IndexURL: `https://minddata.blob.core.windows.net/data/imagenet.idx.txt`,
}

var (
	resume       = flag.Bool(`continue`, false, ``)
	image        = flag.String(`image`, `kungfu.azurecr.io/tf1.13.2:snapshot`, ``)
	trainSamples = flag.Int(`n`, 0, ``)
	batchSize    = flag.Int(`batch-size`, 23, ``)
	dpSize       = flag.Int(`dp`, 4, ``)
)

func main() {
	flag.Parse()
	if err := GetTotalSize(&imagenet); err != nil {
		log.Fatal(err)
	}
	prepare := seq(
		ignore(stop()),
		echo(`old container stopped`),
		start(),
		echo(`systemd started`),
		addIndex(),
		echo(`index added`),
	)
	p := seq(
		If(!*resume, prepare),
		echo(`running`),
		dockerCp(`../benchmarks/tf_read.py`, `/src/tf_read.py`),
		dockerExec(`python3`, `tf_read.py`),
		echo(`done`),
	)
	proc.Main(p)
}

func stop() P { return pc(`docker`, `rm`, `-f`, name) }

func start() P {
	return pc(
		`docker`, `run`,
		`--cap-add`, `SYS_ADMIN`,
		`--device`, `/dev/fuse`,
		`--security-opt`, `apparmor:unconfine`,
		`-v`, `/sys/fs/cgroup/:/sys/fs/cgroup:ro`,
		`-v`, home+`/.az:/etc/mlfs/azure:ro`,
		`--gpus`, `"device=0"`,
		`--rm`,
		`--name`, name,
		`-d`,
		`-t`, *image,
		`/sbin/init`,
	)
}

func addIndex() P {
	n := *trainSamples
	var progress int
	if n > 0 {
		progress = relu(imagenet.Size - n)
	}
	return seq(
		dockerExec(`mlfs-cli`,
			`-ctrl-port`, str(20000),
			`-idx-name`, imagenet.Name,
			`-index-url`, imagenet.IndexURL,
			`-progress`, str(progress),
			`-global-batch-size`, str(*batchSize),
			`-cluster-size`, str(*dpSize),
		),
	)
}

func GetTotalSize(ds *ds.Dataset) error {
	vf, err := vfile.LoadIdxFile(ds.IndexURL)
	if err != nil {
		return err
	}
	size := vf.NumRange()
	if ds.Size > 0 && ds.Size != size {
		log.Printf("inconsistent Dataset Size detected %d !- %d", ds.Size, size)
	}
	ds.Size = size
	return nil
}

func relu(x int) int { return max(0, x) }

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}
