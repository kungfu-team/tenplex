package main

import (
	"flag"
	"log"
	"os"
	"os/user"
	"path"
	"time"

	"github.com/kungfu-team/tenplex/ipv4"
	"github.com/kungfu-team/tenplex/scheduler/logging"
	"github.com/kungfu-team/tenplex/scheduler/scheduler"
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
		if d.SelfIP = detectIP(d.DetectIPv4); len(d.SelfIP) == 0 {
			log.Panic("self IP is empty")
		}
	}
	log.Printf("using self ip: %s", d.SelfIP)
	d.Run()
}

var detectIP = ipv4.Detect

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
