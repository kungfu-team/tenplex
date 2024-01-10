package main

import (
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex-run/job"
	"github.com/kungfu-team/tenplex-run/runop"
)

var (
	mlfsGitCommit = flag.String("mlfs-git-commit", ``, ``)
)

func parseFlags() *job.JobConfig {
	var j job.JobConfig
	j.RegisterFlags(flag.CommandLine)
	flag.Parse()
	j.ParseSchedule()
	return &j
}

func main() {
	j := parseFlags()
	t0 := time.Now()
	defer func() { log.Printf("%s took %s", strings.Join(os.Args, ` `), time.Since(t0)) }()
	log.Printf("%+v", j)
	if len(*mlfsGitCommit) > 0 {
		if failed := checkDaemons(*mlfsGitCommit, j.Cluster.Hosts, j.MLFSPort); failed > 0 {
			log.Printf("%d daemons are using wrong version", failed)
			return
		}
		log.Printf("all daemons are using same version: %q", *mlfsGitCommit)
	}
	runop.Main(j)
}

func checkDaemons(git string, hosts []string, port int) int {
	var fail int
	for _, h := range hosts {
		commit := getDaemonGitCommit(h, port)
		if commit != git {
			log.Printf("%s is not using right git commit %q, using wrong version %q", h, git, commit)
			fail++
		}
	}
	return fail
}

func getDaemonGitCommit(h string, port int) string {
	u := url.URL{
		Scheme: `http`,
		Host:   net.JoinHostPort(h, strconv.Itoa(port)),
		Path:   `/debug`,
	}
	resp, err := http.DefaultClient.Get(u.String())
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	bs, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(bs), "\n") {
		const prefix = `git commit:`
		if strings.Contains(line, prefix) {
			line = strings.TrimPrefix(line, prefix)
			line = strings.TrimSpace(line)
			return line
		}
	}
	return ""
}
