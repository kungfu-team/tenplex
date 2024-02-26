package main

import (
	"flag"
	"log"
	"net"
	"os"
	"path"
	"time"

	"github.com/kungfu-team/tenplex/scheduler/experiments/fakeuser"
	"github.com/kungfu-team/tenplex/scheduler/logging"
	"github.com/lgarithm/go/tr"
)

func main() {
	prog := path.Base(os.Args[0])
	logging.SetupLogger(prog)
	defer tr.Patient(prog, 30*time.Second).Done()
	var u fakeuser.User
	u.RegisterFlags(flag.CommandLine)
	flag.Parse()
	// u.Hosts = resolveHosts(*hosts) // TODO: make it work
	if len(u.PlansFile) > 0 {
		if u.SingleTimedJob {
			if err := u.RunSingleJob(); err != nil {
				log.Panic(err)
			}
		} else {
			if err := u.RunPlans(); err != nil {
				log.Panic(err)
			}
		}
	} else {
		log.Printf("! using deprecated Run")
		u.Run()
	}
}

func resolveHosts(hosts []string) []string {
	var ips []string
	for i, h := range hosts {
		ip := resolve(h)
		log.Printf("#%d : %s -> %s", i, h, ip)
		ips = append(ips, ip)
	}
	return hosts
}

func resolve(h string) string { // TODO: does't work for self
	addrs, err := net.LookupHost(h)
	if err != nil {
		return h
	}
	for _, a := range addrs {
		log.Printf("%s", a)
		return a
	}
	return h
}
