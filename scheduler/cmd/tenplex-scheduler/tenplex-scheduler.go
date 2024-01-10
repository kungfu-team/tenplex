package main

import (
	"flag"
	"log"
	"net"
	"os"
	"os/user"
	"path"
	"time"

	"github.com/kungfu-team/scheduler/logging"
	"github.com/kungfu-team/scheduler/scheduler"
	"github.com/lgarithm/go/tr"
)

func main() {
	prog := path.Base(os.Args[0])
	logging.SetupLogger(prog)
	defer tr.Patient(prog, 300*time.Second).Done()
	var runDir = `/run/tenplex`
	user, err := user.Current()
	if err != nil {
		log.Panic(err)
	}
	log.Printf("user: %s", user.Username)
	if user.Username != `root` {
		runDir = user.HomeDir
	}
	if pwd, _ := os.Getwd(); pwd == `/` {
		if err := setupWorkDir(runDir); err != nil {
			log.Panic(err)
		}
	}
	logDirs()
	var d scheduler.Daemon
	d.RegisterFlags(flag.CommandLine)
	flag.Parse()
	if len(d.DetectIPv4) > 0 || len(d.SelfIP) == 0 {
		d.SelfIP = detectIP(d.DetectIPv4)
	}
	log.Printf("using self ip: %s", d.SelfIP)
	d.Run()
}

func detectIP(nicName string) string {
	nics, err := net.Interfaces()
	if err != nil {
		return ""
	}
	for _, nic := range nics {
		if len(nicName) > 0 && nicName != nic.Name {
			continue
		}
		addrs, err := nic.Addrs()
		if err != nil {
			continue
		}
		for _, addr := range addrs {
			var ip net.IP
			switch v := addr.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip != nil {
				ip = ip.To4()
			}
			if ip != nil {
				// fmt.Printf("%s %s\n", nic.Name, ip.String())
				return ip.String()
			}
		}
	}
	return ""
}

func setupWorkDir(dir string) error {
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return err
	}
	if err := os.Chdir(dir); err != nil {
		return err
	}
	return nil
}

func logDirs() {
	pwd, _ := os.Getwd()
	log.Printf("pwd: %s", pwd)
	home, _ := os.UserHomeDir()
	log.Printf("home: %s", home)
}
