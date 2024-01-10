package scheduler

import (
	"fmt"
	"log"
	"math"
	"os"
	"path"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kungfu-team/mlfs/mlfs"
	"github.com/kungfu-team/scheduler/deviceallocation"
	"github.com/kungfu-team/scheduler/job"
	"github.com/kungfu-team/tenplex-run/cluster"
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
	DevAllocations deviceallocation.DeviceAllocations
	Runners        []*job.Runner
	FinishChannel  chan string
	Admin          string
	reinstallMLFS  bool
	StateMigrator  string
}

func NewScheduler(devAllos deviceallocation.DeviceAllocations) *Scheduler {
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
			ParaConfig:    &paraConf,
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

func (sch *Scheduler) runTimedJob(j job.TimedJob) {
	for i, sp := range j.Timing {
		duration := sp.Time
		size := sp.Size

		unixTime := time.Now().Unix()
		if size == 0 && duration == 0 {
			log.Printf("stop training at %d", unixTime)
			return
		}

		if size == 0 {
			log.Printf("It is unix time %d and training should pause for %d minutes", unixTime, duration)
			continue
		}

		if size%sch.Cluster.GPUsPerContainer != 0 {
			log.Panicf("size %d is not dividible GPUs per container %d", size, sch.Cluster.GPUsPerHost)
		}

		log.Printf("start interval %d of %d for timed job", i, duration)
		numHosts := size / sch.Cluster.GPUsPerHost

		subClu := cluster.NewCluster(
			sch.Cluster.GPUsPerHost,
			sch.Cluster.GPUsPerContainer,
			sch.Cluster.Hosts[:numHosts]...,
		)
		paraConf := sch.DevAllocations[size]
		if i == 0 {
			runner := job.Runner{
				Job:           j.Job,
				Cluster:       subClu,
				ParaConfig:    &paraConf,
				MLFSPort:      mlfs.DefaultCtrlPort,
				TenplexPrefix: tenplexPrefix,
			}
			sch.Runners = []*job.Runner{&runner}
			sch.Jobs = []job.Job{j.Job}
		}
		sch.FinishWG.Add(1)
		if i == 0 {
			go sch.Runners[0].RunTraining(sch.FinishWG, &sch.FinishChannel, sch.selfAddr())
		} else {
			go sch.Runners[0].TransformAndRun(&paraConf, subClu, sch.FinishWG, &sch.FinishChannel, sch.selfAddr())
		}

		if i == len(j.Timing)-1 {
			sch.FinishWG.Wait()
		} else {
			log.Printf("sleep for %d minutes", duration)
			time.Sleep(time.Duration(duration) * time.Minute)
			log.Printf("woke up again")

			// stop training
			atomic.StoreInt32(&sch.stopped, 1)
			sch.FinishWG.Wait()
			atomic.StoreInt32(&sch.stopped, 0)
		}
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
