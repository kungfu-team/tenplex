package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex/ipv4"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/runop"
	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

type TenplexRunFlags struct {
	job.JobConfig
	runop.Options
	DetectIPv4    string `flag:"detect-self-ip"`
	mlfsGitCommit string `flag:"mlfs-git-commit"`
	logfile       string `flag:"logfile"`
}

func (d *TenplexRunFlags) RegisterFlags(flag *flag.FlagSet) {
	d.JobConfig.RegisterFlags(flag)
	d.Options.RegisterFlags(flag)
	structflag.RegisterFlags(d, flag)
}

func main() {
	log.SetPrefix(fmt.Sprintf(`[%s] `, os.Args[0]))
	log.SetFlags(0)
	var d TenplexRunFlags
	d.RegisterFlags(flag.CommandLine)
	flag.Parse()
	d.ParseSchedule()
	d.ParseParaConfig()
	d.Validate()
	if len(d.logfile) > 0 {
		if lf, err := os.Create(d.logfile); err == nil {
			log.Printf("log into %s", d.logfile)
			log.SetOutput(io.MultiWriter(lf, os.Stderr))
			defer lf.Close()
		}
	}
	t0 := time.Now()
	defer func() { log.Printf("%s took %s", strings.Join(os.Args, ` `), time.Since(t0)) }()
	if len(d.DetectIPv4) > 0 {
		if ip := ipv4.Detect(d.DetectIPv4); len(ip) == 0 {
			log.Panicf("failed to detect ipv4 from nic %s", d.DetectIPv4)
		} else {
			d.SchedulerEndpoint = fmt.Sprintf("http://%s:%d", ip, runop.DefaultSchedulerPort)
		}
	}
	log.Printf("%+v", d.JobConfig)
	if len(d.mlfsGitCommit) > 0 {
		if failed := checkDaemons(d.mlfsGitCommit, d.JobConfig.Cluster.Hosts, d.JobConfig.MLFSPort); failed > 0 {
			log.Printf("%d daemons are using wrong version", failed)
			return
		}
		log.Printf("all daemons are using same version: %q", d.mlfsGitCommit)
	}
	runop.Main(&d.JobConfig, d.Options)
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
