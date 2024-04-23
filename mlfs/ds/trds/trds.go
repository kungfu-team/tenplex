package trds

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"path"
	"strings"

	"github.com/kungfu-team/tenplex/mlfs/cache"
	"github.com/kungfu-team/tenplex/mlfs/iseq"
	"github.com/kungfu-team/tenplex/mlfs/state"
	"github.com/kungfu-team/tenplex/mlfs/uri"
	"github.com/kungfu-team/tenplex/mlfs/vfs"
	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

// TFRecords Dataset
type TRDS struct {
	idx    vfile.IndexedFiles
	ranges vfile.NamedRanges
	seq    []int

	cache *cache.Cache
}

func New(idx vfile.IndexedFiles, shuffle bool, seed int) *TRDS {
	n := idx.NumRange()
	seq := iseq.Iota(n)
	if shuffle {
		iseq.Shuffle(seq, seed)
	}
	t := &TRDS{
		idx:    idx,
		ranges: idx.NamedRanges(),
		seq:    seq,
	}
	return t
}

func (ds *TRDS) SetCache(cache *cache.Cache) {
	ds.cache = cache
}

func (ds *TRDS) Mount(r *vfs.Tree, jobID string, initProgres int64, globalBatchSize, clusterSize int, minSamplesPerFile int) error {
	// log.Printf("TRDS::Mount(*, %d, %d, %d, %d)", initProgres, globalBatchSize, clusterSize, minSamplesPerFile)
	r.MkdirAll(`/`)
	jobDir := path.Join(`/job`, jobID)
	r.TouchText(path.Join(jobDir, `order.txt`), func() string {
		bs := &bytes.Buffer{}
		for _, x := range ds.seq {
			fmt.Fprintf(bs, "%d\n", x)
		}
		return bs.String()
	}())
	r.TouchText(path.Join(jobDir, `cluster-prefix.txt`), clusterPrefix(jobID, initProgres, clusterSize)+"\n")
	var prefixes []string
	for i := 0; i < clusterSize; i++ {
		es := &state.ElasticState{
			InitProgres: initProgres,
			ClusterSize: clusterSize,
			Rank:        i,
		}
		prefixes = append(prefixes, ds.MountShard(r, jobID, es, globalBatchSize, minSamplesPerFile))
	}
	r.TouchText(clusterPrefix(jobID, initProgres, clusterSize)+`/global-batch-size.txt`, func() string {
		bs := &bytes.Buffer{}
		fmt.Fprintf(bs, "%d\n", globalBatchSize)
		return bs.String()
	}())
	name := fmt.Sprintf("progress-%d-cluster-of-%d.txt", initProgres, clusterSize)
	if _, err := r.TouchOrReplaceText(path.Join(jobDir, `head.txt`), fmt.Sprintf("/job/%s/%s\n", jobID, name)); err != nil {
		return err
	}
	r.TouchText(path.Join(jobDir, name), func() string {
		bs := &bytes.Buffer{}
		for _, prefix := range prefixes {
			fmt.Fprintf(bs, "%s\n", prefix)
		}
		return bs.String()
	}())
	return nil
}

func (ds *TRDS) MountShard(r *vfs.Tree, jobID string, es *state.ElasticState, globalBatchSize int, minSamplesPerFile int) string {
	prefix := workerPrefix(jobID, es.InitProgres, es.ClusterSize, es.Rank)
	r.MkdirAll(prefix)
	filenames, batchSizes, dropped := ds.createShards(r, prefix, es, globalBatchSize, ds.seq[es.InitProgres:], minSamplesPerFile)
	r.TouchText(path.Join(prefix, `list.txt`), func() string {
		bs := &bytes.Buffer{}
		for _, filename := range filenames {
			fmt.Fprintf(bs, "%s/%s\n", prefix, filename)
		}
		return bs.String()
	}())
	r.TouchText(path.Join(prefix, `batch-sizes.txt`), func() string {
		bs := &bytes.Buffer{}
		for _, p := range groupIntList(batchSizes) {
			fmt.Fprintf(bs, "%d %d\n", p.First, p.Second)
		}
		return bs.String()
	}())
	r.TouchText(path.Join(prefix, `dropped.txt`), func() string {
		bs := &bytes.Buffer{}
		fmt.Fprintf(bs, "%d\n", len(dropped))
		return bs.String()
	}())
	r.TouchText(path.Join(prefix, `dropped-ids.txt`), func() string {
		bs := &bytes.Buffer{}
		for _, id := range dropped {
			fmt.Fprintf(bs, "%d\n", id)
		}
		return bs.String()
	}())
	return prefix
}

func clusterPrefix(jobID string, progress int64, np int) string {
	return fmt.Sprintf("/job/%s/progress/%d/cluster-of-%d", jobID, progress, np)
}

func workerPrefix(jobID string, progress int64, np int, rank int) string {
	name := fmt.Sprintf("rank-%03d", rank)
	return path.Join(clusterPrefix(jobID, progress, np), name)
}

func shardFileName(i int) string {
	return fmt.Sprintf("%04d.tf_record", i)
}

func (ds *TRDS) createShards(r *vfs.Tree, prefix string, es *state.ElasticState, globalBatchSize int, order []int, minSamplesPerFile int) ([]string, []int, []int) {
	drop := true
	if globalBatchSize <= 0 {
		log.Printf("globalBatchSize(%d) <= 0", globalBatchSize)
		return nil, nil, nil
	}
	var dropped []int
	var batchSizes []int
	var buf []int
	var filenames []string
	flush := func(threshold int) {
		if len(buf) < threshold {
			return
		}
		filename := shardFileName(len(filenames))
		filenames = append(filenames, filename)
		vf := vfile.VFile(ds.ranges.Select(buf), ds.cache)
		r.TouchFile(path.Join(prefix, filename), cache.Memcache(vf))
		r.TouchText(path.Join(prefix, filename+".meta"), func() string {
			bs := &bytes.Buffer{}
			for _, x := range buf {
				fmt.Fprintf(bs, "%d\n", x)
			}
			return bs.String()
		}())
		r.TouchText(path.Join(prefix, filename+".idx"), func() string {
			bs := &bytes.Buffer{}
			fmt.Fprintf(bs, "%d\n", 1)
			fmt.Fprintf(bs, "%s %d\n", `./`+filename, len(buf))
			var off uint64
			for _, x := range buf {
				next := off + ds.ranges[x].Range.Len()
				fmt.Fprintf(bs, "%d %d\n", off, next)
				off = next
			}
			return bs.String()
		}())
		buf = nil
	}
	s := iseq.Seq(order)
	for !s.Empty() {
		b := s.Take(globalBatchSize)
		if drop && b.Len() < globalBatchSize {
			dropped = b.Get()
			break
		}
		b = b.Shard(es.Rank, es.ClusterSize)
		batchSizes = append(batchSizes, b.Len())
		buf = append(buf, b.Get()...)
		flush(minSamplesPerFile)
	}
	flush(0)
	return filenames, batchSizes, dropped
}

type ipair struct{ First, Second int }

func groupIntList(is []int) []ipair {
	if len(is) == 0 {
		return nil
	}
	var ps []ipair
	x, c := is[0], 1
	for _, y := range is[1:] {
		if y != x {
			ps = append(ps, ipair{x, c})
			x, c = y, 1
		} else {
			c++
		}
	}
	ps = append(ps, ipair{x, c})
	return ps
}

var ErrInvalidFile = errors.New("invalid file")

func ParseBatchSizes(filename string) ([]ipair, error) {
	f, err := uri.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	bs, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	var ps []ipair
	for _, line := range strings.Split(string(bs), "\n") {
		if len(line) == 0 {
			continue
		}
		var p ipair
		if _, err := fmt.Sscanf(line, "%d%d", &p.First, &p.Second); err != nil {
			return nil, err
		}
		ps = append(ps, p)
	}
	return ps, nil
}
