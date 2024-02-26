package scheduler

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strings"
	"sync/atomic"

	"github.com/kungfu-team/tenplex/scheduler/experiments"
	"github.com/kungfu-team/tenplex/scheduler/job"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
	"github.com/lgarithm/proc"
	"github.com/lgarithm/proc/experimental"
	"github.com/lgarithm/proc/iostream"
)

func (sch *Scheduler) AddTimedJob(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		log.Printf("method must be post")
		http.Error(w, "method must be post", http.StatusMethodNotAllowed)
		return
	}
	data, err := io.ReadAll(req.Body)
	if err != nil {
		log.Printf("cannot read body")
		http.Error(w, "cannot read body", http.StatusInternalServerError)
		return
	}
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	var job job.TimedJob
	err = dec.Decode(&job)
	if err != nil {
		log.Printf("cannot decode body")
		http.Error(w, "cannot decode body", http.StatusInternalServerError)
		return
	}

	sch.runTimedJob(job)
}

func (sch *Scheduler) AddJobs(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		log.Printf("method must be post")
		http.Error(w, "method must be post", http.StatusMethodNotAllowed)
		return
	}
	data, err := io.ReadAll(req.Body)
	if err != nil {
		log.Printf("cannot read body")
		http.Error(w, "cannot read body", http.StatusInternalServerError)
		return
	}
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	var jobs []job.Job
	err = dec.Decode(&jobs)
	if err != nil {
		log.Printf("cannot decode body")
		http.Error(w, "cannot decode body", http.StatusInternalServerError)
		return
	}

	sch.scale(jobs)
}

func (sch *Scheduler) SetCluster(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		log.Printf("method must be post")
		http.Error(w, "method must be post", http.StatusMethodNotAllowed)
		return
	}
	data, err := io.ReadAll(req.Body)
	if err != nil {
		log.Printf("cannot read body")
		http.Error(w, "cannot read body", http.StatusInternalServerError)
		return
	}
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	var clu cluster.Cluster
	err = dec.Decode(&clu)
	if err != nil {
		log.Printf("cannot decode body")
		http.Error(w, "cannot decode body", http.StatusInternalServerError)
		return
	}
	if sch.Cluster != nil {
		log.Printf("change cluster to %v", clu)
	}
	sch.Cluster = &clu

	log.Printf("schduler cluster %#v", sch.Cluster)
	proc.Main(parmap(func(h string) P { return experimental.WaitSSH(at(sch.Admin, h)) }, sch.Cluster.Hosts...))
	log.Printf("all workers are up: %#v", sch.Cluster)

	// proc.Main(term(`setup swarm % `, experiments.SetupSwarm(sch.Admin, sch.Cluster.Hosts, sch.Cluster.Hosts)))
	if sch.reinstallMLFS {
		log.Printf("reinstall mlfsd on %d workers", len(sch.Cluster.Hosts))
		proc.Main(parmap(func(h string) P { return experiments.ReInstallMLFS(at(sch.Admin, h)) }, sch.Cluster.Hosts...))
	}
	log.Printf("restarting mlfsd on %d workers", len(sch.Cluster.Hosts))
	proc.Main(parmap(sch.restartMLFS, sch.Cluster.Hosts...))
	log.Printf("restarted mlfsd ...")

	log.Printf("clean training dir")
	proc.Main(ignore(parmap(sch.cleanTrainingDir, sch.Cluster.Hosts...)))
	log.Printf("cleaned training dir")

	// log.Printf("upload state migrator")
	// proc.Main(parmap(sch.sendStateMigrator, sch.Cluster.Hosts...))
	// log.Printf("uploaded state migrator")

	log.Printf("clone transformer-checkpoint")
	proc.Main(ignore(parmap(sch.cloneTransformerCheckpoint, sch.Cluster.Hosts...)))
	log.Printf("cloned transformer-checkpoint")
}

func (sch *Scheduler) GetStop(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		log.Printf("method must be get")
		http.Error(w, "method must be get", http.StatusMethodNotAllowed)
		return
	}
	if atomic.LoadInt32(&sch.stopped) > 0 {
		w.Write([]byte("stop"))
	} else {
		w.Write([]byte("run"))
	}
}

func (sch *Scheduler) restartMLFS(h string) P {
	pc := at(sch.Admin, h).PC
	pc = proc.WithTerm(pc)
	pc = experimental.WithLog(pc)
	p := seq(
		proc.Echo(`starting mlfs `+h),
		pc(`sudo`, `systemctl`, `stop`, `mlfs`),
		pc(`sudo`, `systemctl`, `start`, `mlfs`),
		pc(`sudo`, `systemctl`, `status`, `mlfs`),
		proc.Echo(`started mlfs `+h),
		proc.If(false, proc.Lambda(func() P {
			sas, err := loadSAS()
			if err != nil {
				log.Panic(err)
			}
			return pc(`mlfs-cli`, `-sas`, strings.ReplaceAll("tenplex:"+sas, `&`, `\&`))
		})),
		pc(`mlfs`, `info`),
	)
	ps1 := h + ` % `
	return term(ps1, p)
}

func (sch *Scheduler) cleanTrainingDir(h string) P {
	p := Proc{
		Prog: `sudo`,
		Args: []string{`rm -r ~/.tenplex/training/*`},
		Host: h,
		User: sch.Admin,
	}
	return ssh(p)
}

func (sch *Scheduler) sendStateMigrator(h string) P {
	rpc := at(sch.Admin, h).PC
	rpc = proc.WithTerm(rpc)
	rpc = experimental.WithLog(rpc)
	pc := proc.PC
	pc = proc.WithTerm(pc)
	pc = experimental.WithLog(pc)
	stateMigrator := sch.StateMigrator
	if len(stateMigrator) == 0 {
		stateMigrator = "tenplex-state-transformer"
	}
	remoteDir := path.Join(tenplexPrefixRemote, `bin`)
	log.Printf("%s %s %s", `scp`, stateMigrator, sch.Admin+`@`+h+`:`+remoteDir)
	return seq(
		rpc(`rm`, `-fr`, remoteDir),
		rpc(`mkdir`, `-p`, remoteDir),
		pc(`scp`, `-o`, `StrictHostKeyChecking=no`, stateMigrator, sch.Admin+`@`+h+`:`+remoteDir),
	)
}

func (sch *Scheduler) sendTransformerCheckpoint(h string) P {
	pc := proc.PC
	pc = proc.WithTerm(pc)
	pc = experimental.WithLog(pc)
	return seq(
		at(sch.Admin, h).PC(`rm`, `-fr`, path.Join(tenplexPrefixRemote, `transformer-checkpoint`)),
		term(`sendTransformerCheckpoint % `,
			pc(`scp`, `-o`, `StrictHostKeyChecking=no`, `-r`, `./transformer-checkpoint`, sch.Admin+`@`+h+`:`+tenplexPrefixRemote)),
	)
}

func (sch *Scheduler) cloneTransformerCheckpoint(h string) P {
	ckptPath := path.Join(tenplexPrefixRemote, `transformer-checkpoint`)
	return seq(
		at(sch.Admin, h).PC(`rm`, `-fr`, ckptPath),
		at(sch.Admin, h).PC(`git`, `clone`, `https://github.com/kungfu-team/transformer-checkpoint.git`, path.Join(tenplexPrefixRemote, ckptPath)),
	)
}

type (
	P    = proc.P
	Proc = proc.Proc
	At   = proc.UserHost
)

var (
	at     = proc.At
	par    = proc.Par
	seq    = proc.Seq
	term   = proc.Term
	ignore = proc.Ignore
	ssh    = proc.SSH
)

func parmap(f func(string) P, hs ...string) P {
	round := getParmapRunID()
	var ps []P
	for i, h := range hs {
		p := f(h)
		p = tee2Files(fmt.Sprintf("logs/round-%d-worker-%d-%s", round, i, h), p)
		ps1 := fmt.Sprintf("#%d<%s> %% ", i, h)
		ps = append(ps, term(ps1, p))
	}
	return par(ps...)
}

func tee2Files(name string, p P) P {
	o := iostream.Open2Lazy(name+`.out.log`, name+`.err.log`)
	return proc.Tee(p, o.StdWriters())
}

func genCounter() func() int {
	var id int
	return func() int { x := id; id++; return x }
}

var getParmapRunID = genCounter()

func loadSAS() (string, error) {
	sas, err := os.ReadFile(path.Join(home, `.az/tenplex.full.sas`))
	if err != nil {
		return "", err
	}
	return strings.TrimPrefix(string(sas), "?"), nil
}
