package mlfs

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds/trds"
	"github.com/kungfu-team/tenplex/mlfs/iotrace"
	"github.com/kungfu-team/tenplex/mlfs/par"
	"github.com/kungfu-team/tenplex/mlfs/tfrecord"
	"github.com/kungfu-team/tenplex/mlfs/uri"
	"github.com/kungfu-team/tenplex/mlfs/utils"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

type Test struct {
	IsTFRecord bool
	Host       string
	Port       int
	Mount      string
	JobID      string
	Rank       *int
}

func (t *Test) RegisterFlags(flag *flag.FlagSet) {
	flag.StringVar(&t.Host, `host`, `127.0.0.1`, ``)
	flag.IntVar(&t.Port, `port`, 0, ``)
	flag.StringVar(&t.Mount, `mnt`, ``, ``)
	flag.StringVar(&t.JobID, `job`, `0`, ``)
	flag.BoolVar(&t.IsTFRecord, `tf-record`, false, ``)
}

func (t Test) Run() error {
	flag.Parse()
	if len(t.Mount) > 0 {
		if err := t.testVFS(t.Mount, t.JobID); err != nil {
			return err
		}
	}
	if t.Port > 0 {
		u := url.URL{
			Scheme: `http`,
			Host:   net.JoinHostPort(t.Host, strconv.Itoa(t.Port)),
			Path:   `/`,
		}
		if err := t.testVFS(u.String(), t.JobID); err != nil {
			return err
		}
	}
	return nil
}

var errFailed = errors.New("failed")

func (t Test) testVFS(root string, jobID string) error {
	log.Printf("testing VFS(%s, %q)...", root, jobID)
	t0 := time.Now()
	defer func() { log.Printf("took %s", time.Since(t0)) }()
	heads, err := utils.Readlines(appendPath(root, fmt.Sprintf(`/job/%s/head.txt`, jobID)))
	if err != nil {
		return err
	}
	if len(heads) < 1 {
		return trds.ErrInvalidFile
	}
	head := heads[0]
	prefixes, err := utils.Readlines(appendPath(root, head))
	if err != nil {
		return err
	}
	c := iotrace.NewCounter()
	defer iotrace.Reporter(c, ``).Stop()
	p := par.New(len(prefixes))
	var tot int64
	var failed int32
	for i, prefix := range prefixes {
		if t.Rank != nil && *t.Rank != i {
			continue
		}
		func(prefix string) {
			p.Do(func() {
				n, err := t.testWorkerPrefix(c, root, prefix)
				atomic.AddInt64(&tot, int64(n))
				if err != nil {
					atomic.AddInt32(&failed, 1)
					log.Printf("%s failed: %v", prefix, err)
				}
			})
		}(prefix)
	}
	p.Wait()
	log.Printf("%d records", tot)
	if failed > 0 {
		utils.ExitErr(errFailed)
	}
	return nil
}

func (t Test) testWorkerPrefix(c *iotrace.Counter, root, prefix string) (int, error) {
	var readfile = readRanges
	if t.IsTFRecord {
		readfile = readTFRecords
	}
	batchSizes, err := trds.ParseBatchSizes(appendPath(root, prefix+`/batch-sizes.txt`))
	if err != nil {
		return 0, err
	}
	if len(batchSizes) <= 0 {
		return 0, trds.ErrInvalidFile
	}
	list, err := utils.Readlines(appendPath(root, prefix+`/list.txt`))
	if err != nil {
		return 0, err
	}
	var n int
	for _, name := range list {
		filepath := appendPath(root, name)
		log.Printf("testWorkerPrefix: %s", filepath)
		rs, err := readfile(c, filepath)
		n += len(rs)
		if err != nil {
			log.Printf("error after read %d records: %v", len(rs), err)
			return n, err
		}
	}
	batchSize, count := batchSizes[0].First, batchSizes[0].Second
	if n != batchSize*count {
		log.Printf("%d != %d x %d", n, batchSize, count)
		return n, trds.ErrInvalidFile
	}
	return n, nil
}

func readRanges(c *iotrace.Counter, filename string) (vfile.Ranges, error) {
	log.Printf("readRanges(%s)", filename)
	idxFilename := filename + `.idx`
	indices, err := vfile.LoadIdxFile(idxFilename)
	if err != nil {
		return nil, err
	}
	if len(indices) != 1 {
		return nil, errors.New("indices should contain exactly 1 file")
	}
	index := indices[0]
	f, err := uri.Open(filename)
	if err != nil {
		return nil, err
	}
	r := iotrace.TraceReader(f, c)
	for _, rg := range index.Ranges {
		buf := make([]byte, rg.Len())
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
	}
	return index.Ranges, nil
}

func readTFRecords(c *iotrace.Counter, filename string) (vfile.Ranges, error) {
	log.Printf("readTFRecords(%s)", filename)
	f, err := uri.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rs vfile.Ranges
	var off uint64
	for i := 0; ; i++ {
		info, _, err := tfrecord.ReadTFRecord(iotrace.TraceReader(f, c))
		if err == io.EOF {
			break
		}
		if err != nil {
			return rs, err
		}
		r := vfile.Range{
			Begin: off,
			End:   off + uint64(info.Len) + 16,
		}
		off = r.End
		rs = append(rs, r)
	}
	return rs, nil
}

var appendPath = uri.AppendPath
