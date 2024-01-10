package buildinfo

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

var t0 = time.Now()

type BuildInfo struct {
	BuildTimestamp string
	BuildTime      time.Time
	BuildHost      string
	GitCommit      string
	GitBranch      string
	GitRev         string
}

func (i *BuildInfo) Parse() {
	if n, err := strconv.Atoi(i.BuildTimestamp); err == nil {
		i.BuildTime = time.Unix(int64(n), 0)
	}
}

func (i *BuildInfo) Show(w io.Writer) {
	fmt.Fprintf(w, "git branch: %s\n", i.GitBranch)
	fmt.Fprintf(w, "git commit: %s\n", i.GitCommit)
	fmt.Fprintf(w, "git rev: %s\n", i.GitRev)
	fmt.Fprintf(w, "build host: %s\n", i.BuildHost)
	if i.BuildTime.Unix() > 0 {
		fmt.Fprintf(w, "build age: %s\n", time.Since(i.BuildTime))
	} else {
		fmt.Fprintf(w, "build age: %s (%q)\n", `?`, i.BuildTimestamp)
	}
	fmt.Fprintf(w, "run age: %s\n", time.Since(t0))
}

func (i *BuildInfo) ServeHTTP(w http.ResponseWriter, r *http.Request) { i.Show(w) }

var Default BuildInfo

// func Set(i BuildInfo) {
// 	Default = i
// 	Default.Parse()
// }

var (
	BuildHost      string
	BuildTimestamp string
	GitCommit      string
	GitBranch      string
	GitRev         string
)

func init() {
	Default = BuildInfo{
		BuildHost:      BuildHost,
		BuildTimestamp: BuildTimestamp,
		GitCommit:      GitCommit,
		GitBranch:      GitBranch,
		GitRev:         GitRev,
	}
	Default.Parse()
}
