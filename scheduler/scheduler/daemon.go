package scheduler

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os/user"

	"github.com/kungfu-team/tenplex/tenplex-run/para_config"
	"github.com/kungfu-team/tenplex/tenplex-run/web"
)

type Daemon struct {
	Port           int
	ParaConfigFile string
	SelfIP         string
	DetectIPv4     string
	User           string
	ReInstall      bool
	StateMigrator  string
}

func defaultUser() string {
	if u, err := user.Current(); err == nil {
		return u.Username
	}
	return ""
}

func (d *Daemon) RegisterFlags(flag *flag.FlagSet) {
	flag.IntVar(&d.Port, `port`, DefaultSchedulerPort, ``)
	flag.StringVar(&d.ParaConfigFile, `para-config`, ``, ``)
	flag.StringVar(&d.SelfIP, `self-ip`, ``, ``)
	flag.StringVar(&d.DetectIPv4, `detect-self-ip`, ``, ``)
	flag.StringVar(&d.User, `u`, defaultUser(), `cluster user`)
	flag.BoolVar(&d.ReInstall, `reinstall`, false, ``)
	flag.StringVar(&d.StateMigrator, `tenplex-state-transformer`, ``, `path to tenplex-state-transformer`)
}

func (d Daemon) Run() {
	devAllos, err := para_config.LoadFile(d.ParaConfigFile)
	if err != nil {
		log.Panic(err)
	}
	para_config.Log(devAllos)
	s := NewScheduler(devAllos)
	s.SelfIP = d.SelfIP
	s.SelfPort = d.Port
	s.Admin = d.User
	s.reinstallMLFS = d.ReInstall
	s.StateMigrator = d.StateMigrator
	defer s.Shutdown()

	mux := http.NewServeMux()
	mux.HandleFunc("/addjobs", s.AddJobs)
	mux.HandleFunc("/setcluster", s.SetCluster)
	mux.HandleFunc("/stop", s.GetStop)
	hs := http.Server{
		Addr:    fmt.Sprintf(":%d", d.Port),
		Handler: withLogReq(mux),
	}
	go func() {
		s.WaitFinish()
		log.Printf("all job finished")
		hs.Shutdown(context.TODO())
	}()
	if err := hs.ListenAndServe(); err != nil {
		log.Printf("ListenAndServe: %v", err) // TODO: ignore shutdown
	}
	log.Printf("stopped")
}

var withLogReq = web.WithLogReq
