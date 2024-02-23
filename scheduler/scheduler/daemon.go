package scheduler

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/user"

	"github.com/kungfu-team/tenplex/scheduler/deviceallocation"
)

type Daemon struct {
	Port                 int
	DeviceAllocationFile string
	SelfIP               string
	DetectIPv4           string
	User                 string
	ReInstall            bool
	StateMigrator        string
}

func defaultUser() string {
	if u, err := user.Current(); err == nil {
		return u.Username
	}
	return ""
}

func (d *Daemon) RegisterFlags(flag *flag.FlagSet) {
	flag.IntVar(&d.Port, `port`, DefaultSchedulerPort, ``)
	flag.StringVar(&d.DeviceAllocationFile, `device-allocation`, ``, ``)
	flag.StringVar(&d.SelfIP, `self-ip`, ``, ``)
	flag.StringVar(&d.DetectIPv4, `detect-self-ip`, ``, ``)
	flag.StringVar(&d.User, `u`, defaultUser(), `cluster user`)
	flag.BoolVar(&d.ReInstall, `reinstall`, false, ``)
	flag.StringVar(&d.StateMigrator, `tenplex-state-transformer`, ``, `path to tenplex-state-transformer`)
}

func (d Daemon) Run() {
	devAllos := deviceallocation.GenDevicAllocations()
	if len(d.DeviceAllocationFile) > 0 {
		var err error
		if devAllos, err = deviceallocation.LoadFile(d.DeviceAllocationFile); err != nil {
			log.Panic(err)
		}
	}
	deviceallocation.Log(devAllos)

	s := NewScheduler(devAllos)
	s.SelfIP = d.SelfIP
	s.SelfPort = d.Port
	s.Admin = d.User
	s.reinstallMLFS = d.ReInstall
	s.StateMigrator = d.StateMigrator
	defer s.Shutdown()

	mux := http.NewServeMux()
	mux.HandleFunc("/addjobs", s.AddJobs)
	mux.HandleFunc("/addtimedjob", s.AddTimedJob)
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

func withLogReq(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		LogRequest(req)
		h.ServeHTTP(w, req)
	})
}

var LogRequest = func(r *http.Request) {
	accessLog.Printf("%s %s | %s %s", r.Method, r.URL, r.RemoteAddr, r.UserAgent())
}

var accessLog = logger{l: log.New(os.Stderr, "[access] ", 0)}

type logger struct{ l *log.Logger }

func (l *logger) Printf(format string, v ...interface{}) {
	l.l.Output(2, fmt.Sprintf(format, v...))
}
