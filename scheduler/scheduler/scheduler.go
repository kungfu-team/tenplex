package scheduler

import (
	"fmt"
	"log"
	"math"
	"os"
	"path"
	"sync"
	"sync/atomic"

	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/scheduler/job"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
	"github.com/kungfu-team/tenplex/tenplex-run/para_config"
)

var (
	home                = os.Getenv(`HOME`)
	tenplexPrefix       = path.Join(home, `.tenplex`)
	tenplexPrefixRemote = `.tenplex`
)

const DefaultSchedulerPort = 22222

type Scheduler struct {
	SelfIP         string
	SelfPort       int
	Cluster        *cluster.Cluster
	Jobs           []job.Job
	stopped        int32
	FinishWG       *sync.WaitGroup
	DeleteLock     *sync.RWMutex
	DevAllocations para_config.ParaConfig
	Runners        []*job.Runner
	FinishChannel  chan string
	Admin          string
	reinstallMLFS  bool
	StateMigrator  string
}

func NewScheduler(devAllos para_config.ParaConfig) *Scheduler {
	return &Scheduler{
		SelfPort:       DefaultSchedulerPort,
		FinishWG:       &sync.WaitGroup{},
		DeleteLock:     &sync.RWMutex{},
		DevAllocations: devAllos,
		FinishChannel:  make(chan string),

		Admin: defaultUser(),
		// TODO: infer SelfIP
	}
}

func nextLowerPowTwo(num int) int {
	x := math.Floor(math.Log2(float64(num)))
	return int(math.Pow(2, x))
}

func (sch *Scheduler) selfIP() string {
	if len(sch.SelfIP) > 0 {
		return sch.SelfIP
	}
	return sch.Cluster.Hosts[0] // TODO: double check
}

func (sch *Scheduler) selfAddr() string {
	return fmt.Sprintf("http://%s:%d", sch.selfIP(), sch.SelfPort)
}

func (sch *Scheduler) WaitFinish() {
	for {
		n := <-sch.FinishChannel
		log.Printf("waitFinish read from channel %s", n)
		sch.scale(nil)
	}
}

func (sch *Scheduler) scale(newJobs []job.Job) {
	log.Printf("scale to %d new jobs: %s", len(newJobs), job.ShowJobIds(newJobs))
	if len(sch.Jobs) > 0 { // there are old jobs
		atomic.StoreInt32(&sch.stopped, 1)
		sch.FinishWG.Wait()
		atomic.StoreInt32(&sch.stopped, 0)

		// TODO: remove finished jobs
		for _, r := range sch.Runners {
			if r.Finished {
				sch.removeJob(r.Job.ID)
				sch.removeRunner(r)
			}
		}
	}

	numJobs := len(sch.Jobs) + len(newJobs)
	log.Printf("scale to %d total jobs: %s", numJobs, job.ShowJobIds(sch.Jobs, newJobs))
	if numJobs == 0 {
		return
	}
	numJobDevices := len(sch.Cluster.Hosts) * sch.Cluster.GPUsPerHost / numJobs
	numJobDevices = nextLowerPowTwo(numJobDevices)
	// TODO: allocate devices and not hosts
	numJobHosts := numJobDevices / sch.Cluster.GPUsPerHost

	if len(sch.Jobs) > 0 {
		for i, ru := range sch.Runners {
			newSubClu := cluster.NewCluster(
				sch.Cluster.GPUsPerHost,
				sch.Cluster.GPUsPerContainer,
				sch.Cluster.Hosts[i*numJobHosts:(i+1)*numJobHosts]...,
			)
			newParaConf := sch.DevAllocations[numJobDevices]
			sch.FinishWG.Add(1)
			go ru.TransformAndRun(&newParaConf, newSubClu, sch.FinishWG, &sch.FinishChannel, sch.selfAddr())
		}
	}

	i := len(sch.Jobs)
	for _, jo := range newJobs {
		log.Printf("new job %s %d steps", jo.ID, jo.Steps)
		subClu := cluster.NewCluster(
			sch.Cluster.GPUsPerHost,
			sch.Cluster.GPUsPerContainer,
			sch.Cluster.Hosts[i*numJobHosts:(i+1)*numJobHosts]...,
		)
		paraConf := sch.DevAllocations[numJobDevices]
		runner := job.Runner{
			Job:           jo,
			Cluster:       subClu,
			MDP:           &paraConf,
			MLFSPort:      mlfs.DefaultCtrlPort,
			TenplexPrefix: tenplexPrefix,
		}
		sch.Runners = append(sch.Runners, &runner)
		sch.Jobs = append(sch.Jobs, jo)

		sch.FinishWG.Add(1)
		go runner.RunTraining(sch.FinishWG, &sch.FinishChannel, sch.selfAddr())

		i++
	}
}

func (sch *Scheduler) removeJob(jid string) {
	sch.DeleteLock.Lock()
	defer sch.DeleteLock.Unlock()

	var idx int
	for i, j := range sch.Jobs {
		if jid == j.ID {
			idx = i
			break
		}
	}
	sch.Jobs = append(sch.Jobs[:idx], sch.Jobs[idx+1:]...)
}

func (sch *Scheduler) removeRunner(ru *job.Runner) {
	sch.DeleteLock.Lock()
	defer sch.DeleteLock.Unlock()

	var idx int
	for i, r := range sch.Runners {
		if ru == r {
			idx = i
			break
		}
	}
	sch.Runners = append(sch.Runners[:idx], sch.Runners[idx+1:]...)
}

func (sch *Scheduler) Shutdown() {
	// noop
}
