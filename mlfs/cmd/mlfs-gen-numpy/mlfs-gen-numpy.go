package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/kungfu-team/tenplex/mlfs/vfs/vfile"
)

var (
	idxFile  = flag.String("index-file", ``, `path to index file`)
	dataFile = flag.String("data-file", ``, `path to data file`)

	dpSize        = flag.Int("dp-size", 1, ``)
	globalBatsize = flag.Int("global-batch-size", 1, ``)
)

type Pair = vfile.Range

type Index = vfile.Ranges

func main() {
	flag.Parse()
	idx, err := LoadIndex(*idxFile)
	if err != nil {
		panic(err)
	}
	log.Printf("got %d ranges from the index", len(idx))

	if err := AllShards(idx, *dataFile, *dpSize, *globalBatsize); err != nil {
		panic(err)
	}
}

func main1() {
	flag.Parse()
	indices, err := vfile.LoadIdxFile(*idxFile)
	if err != nil {
		panic(err)
	}
	log.Printf("index contains %d files", len(indices))
	for i, idx := range indices {
		log.Printf("got %d ranges from %d-th file (%s)", len(idx.Ranges), i, idx.Filepath)
	}
}

func AllShards(idx Index, filepath string, dpSize /* size of data parallel*/ int, globalBatsize int) error {
	for rank := 0; rank < dpSize; rank++ {
		dir := path.Join(path.Dir(filepath), strconv.Itoa(rank))
		if err := os.MkdirAll(dir, os.ModePerm); err != nil {
			return err
		}
		buf, err := os.Create(path.Join(dir, path.Base(filepath)))
		if err != nil {
			return err
		}
		subIdx, err := ShardFile(idx, filepath, rank, dpSize, globalBatsize, buf)
		buf.Close()
		if err != nil {
			return err
		}
		if err := WriteIndex(path.Join(dir, `indices.txt`), subIdx); err != nil {
			return err
		}
	}
	return nil
}

/*
e.g.
16 samples, global BS = 8, dpSize = 4

=> 2 logical batches,
       i   j
sample 0 | worker 0
sample 1 | worker 0
--
sample 2
sample 3
--
sample 4
sample 5
--
sample 6
sample 7   7

------------
       i   j
sample 8 | 0 worker 0
sample 9 | 1 worker 0
sample 10 | 2
sample 11 | 3
sample 12 | 4
sample 13 | 5
sample 14 | 6
sample 16 | 7


GBS=8, DP=4
j | 0 1 2 3 4 5 6 7
-------------------
w | 0 0 1 1 2 2 3 3

w = f(j), f in terms of GBS and DP

GBS=7, DP=4, LBS=2
j | 0 1 2 3 4 5 6
-----------------
w | 0 0 1 1 2 2 3


not this
// GBS=8, DP=4
// j | 0 1 2 3 4 5 6
// -----------------
// w | 0 1 2 3 0 1 2

// w =

=>
shard file 0 ::
sample 0
sample 1
sample 8
sample 9
*/

func ceilDiv(a, b int) int {
	if a%b == 0 {
		return a / b
	}
	return a/b + 1
}

func ShardFile(idx Index, filename string, rank, dpSize /* size of data parallel*/ int, globalBatsize int, buf io.Writer) (Index, error) {
	log.Printf("ShardFile(idx, %s, %d, %d)", filename, rank, dpSize)
	localBatchSize := ceilDiv(globalBatsize, dpSize)
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var subIdx []Pair
	var off int64
	for i, p := range idx {
		if i%globalBatsize/localBatchSize == rank {
			f.Seek(int64(p.Begin), io.SeekStart)
			r := io.LimitReader(f, int64(p.End-p.Begin))
			n, err := io.Copy(buf, r)
			if err != nil {
				return nil, err
			}
			p := Pair{
				Begin: uint64(off),
				End:   uint64(off + n),
			}
			off += n
			subIdx = append(subIdx, p)
		}
	}
	return subIdx, nil
}

func WriteIndex(filename string, idx Index) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	for _, p := range idx {
		fmt.Fprintf(f, "%d %d\n", p.Begin, p.End)
	}
	return nil
}

func LoadIndex(filename string) (Index, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReader(f)
	var ps []Pair
	for {
		bs, _, err := br.ReadLine()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		splits := strings.Split(string(bs), " ")
		begin, err := strconv.ParseInt(splits[0], 10, 64)
		if err != nil {
			return nil, err
		}
		end, err := strconv.ParseInt(splits[1], 10, 64)
		if err != nil {
			return nil, err
		}
		p := Pair{
			Begin: uint64(begin),
			End:   uint64(end),
		}
		ps = append(ps, p)
	}
	return ps, nil
}
