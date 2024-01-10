package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"path"
	"strings"
	"time"

	"github.com/kungfu-team/mlfs/bimap"
	"github.com/kungfu-team/mlfs/mlfs"
	"github.com/kungfu-team/mlfs/par"
	"github.com/kungfu-team/mlfs/uri"
	"github.com/kungfu-team/mlfs/utils"
)

var (
	idxName  = flag.String("idx-name", "", "")
	idxFile  = flag.String("idx-file", "", "") // TODO: change to index-url
	host     = flag.String("host", "", "")
	ctrlPort = flag.Int("ctrl-port", mlfs.DefaultCtrlPort, "")

	pause = flag.Bool("pause", false, "")

	jobID           = flag.String("job", "0", "")
	progress        = flag.Int("progress", 0, "")
	globalBatchSize = flag.Int("global-batch-size", 1, "")
	clusterSize     = flag.Int("cluster-size", 1, "")
	seed            = flag.Int("seed", 0, "") // seed=0 means no shuffle

	fetch     = flag.Bool("fetch", false, "")
	async     = flag.Bool("async", false, "")
	parallism = flag.Int("m", 2, "")

	md5File = flag.String(`md5-file`, ``, ``)

	sas = flag.String(`sas`, ``, ``)
)

func main() {
	flag.Parse()
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	if !mlfs.WaitTCP(*host, *ctrlPort) {
		log.Fatalf("wait timeout")
	}
	cli, err := mlfs.NewClient(*ctrlPort)
	if err != nil {
		utils.ExitErr(err)
	}
	if *pause {
		cli.Pause()
		return
	}
	if len(*sas) > 0 {
		if parts := strings.SplitN(*sas, `:`, 2); len(parts) == 2 {
			cli.SetSAS(parts[0], parts[1])
		} else {
			log.Fatalf("invalid -sas: %q", *sas)
		}
		return
	}
	if err := mount(cli); err != nil {
		log.Fatalf("%v", err)
	}
	if *fetch {
		if len(*md5File) > 0 {
			fetchWithMD5(cli)
		} else {
			if err := cli.FetchAll(*idxName, *async); err != nil {
				utils.ExitErr(err)
			}
		}
	}
}

func mount(cli *mlfs.Client) error {
	if err := cli.AddIndex(*idxName, *idxFile); err != nil {
		return err
	}
	if err := cli.Mount(*jobID, *idxName, int64(*progress), *globalBatchSize, *clusterSize, *seed); err != nil {
		return err
	}
	var s string
	if err := cli.GetRoot(&s); err != nil {
		return err
	}
	log.Printf("root: %s", s)
	return nil
}

func fetchWithMD5(cli *mlfs.Client) {
	// md5File := `https://minddata.blob.core.windows.net/data/imagenet/md5sum.txt`
	m, err := getMD5(*md5File)
	if err != nil {
		utils.ExitErr(err)
	}
	bs, err := cli.File(`/files.txt`)
	if err != nil {
		return
	}
	p := par.New(*parallism)
	for _, url := range strings.Split(strings.TrimSpace(string(bs)), "\n") {
		func(url string) {
			p.Do(func() {
				h, _ := m.RGet(path.Base(url))
				log.Printf("%s | md5: %s", url, h)
				cli.Fetch(url, h)
			})
		}(url)
	}
	p.Wait()
}

func getMD5(md5File string) (*bimap.BiMap, error) {
	f, err := uri.Open(md5File)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReader(f)
	m := bimap.New()
	for {
		line, _, err := br.ReadLine()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		parts := strings.Fields(string(line))
		if len(parts) == 2 {
			if ok := m.Add(parts[0], parts[1]); !ok {
				return nil, fmt.Errorf("invalid entry: %s: %s", parts[0], parts[1])
			}
		}
	}
	return m, nil
}
