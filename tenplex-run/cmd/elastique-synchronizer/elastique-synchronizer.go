// runs one instance per replica
package main

import (
	"flag"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex/tenplex-run/mp"
)

var (
	hosts = flag.String("hosts", "", "")
	rank  = flag.Int("rank", -1, "")
	port  = flag.Int("port", 30000, "")

	prefix = flag.String("prefix", "", "")
	ckpt   = flag.String("ckpt", "", "")
	action = flag.String("action", "", "gather | scatter")
)

func main() {
	flag.Parse()
	log.SetFlags(0)
	t0 := time.Now()
	defer func() { log.Printf("%s took %s", os.Args[0], time.Since(t0)) }()
	log.Printf("running %s", strings.Join(os.Args, ` `))
	hs := split(*hosts)
	s := mp.Synchronizer{
		Port:          *port,
		Rank:          *rank,
		WorkerPrefix:  path.Join(*prefix, str(*rank), *ckpt),
		GatherPrefix:  path.Join(*prefix, `gather`, *ckpt),
		ScatterPrefix: path.Join(*prefix, `split`, *ckpt),
	}
	switch *action {
	case "gather":
		s.RunGather(*rank, hs)
	case "scatter":
		s.RunScatter(*rank, hs)
	default:
		log.Panicf("invalid action: %q", *action)
	}
}

var str = strconv.Itoa

func split(s string) []string {
	var ss []string
	for _, s := range strings.Split(s, `,`) {
		s = strings.TrimSpace(s)
		if len(s) > 0 {
			ss = append(ss, s)
		}
	}
	return ss
}
