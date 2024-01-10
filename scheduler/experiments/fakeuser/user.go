package fakeuser

import (
	"bytes"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/scheduler/job"
	"github.com/kungfu-team/tenplex/scheduler/scalepoint"
	"github.com/kungfu-team/tenplex/scheduler/scheduler"
	"github.com/kungfu-team/tenplex/scheduler/stringlist"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
)

var openwebtext = ds.Dataset{
	IndexURL: `https://tenplex.blob.core.windows.net/tenplexcontainer/indices.txt`,
	Name:     `openwebtext`,
}

//	var enwiki = ds.Dataset{
//	        IndexURL: `https://tenplex.blob.core.windows.net/tenplexcontainer/gpt_enwiki_indices.txt`,
//	        Name:     `enwiki`,
//	}
var enwiki = ds.Dataset{
	IndexURL: `/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt`,
	Name:     `enwiki`,
}

func (u User) Add(url string, ob interface{}) {
	buf := bytes.NewBuffer([]byte{})
	enc := gob.NewEncoder(buf)
	err := enc.Encode(ob)
	if err != nil {
		log.Panic(err)
	}
	resp, err := http.Post(url, "application/gob", buf)
	if err != nil {
		log.Panic(err)
	}
	if resp.StatusCode != http.StatusOK {
		log.Panicf("add job post failed, status: %s", resp.Status)
	}
}

func (u User) AddTimedJob(ip string, job job.TimedJob) {
	url := fmt.Sprintf("http://%s:%d/addtimedjob", ip, u.SchedulerPort)
	u.Add(url, job)
}

func (u User) AddJobs(ip string, jobs ...job.Job) {
	url := fmt.Sprintf("http://%s:%d/addjobs", ip, u.SchedulerPort)
	u.Add(url, jobs)
}

func (u User) SetCluster(ip string) {
	clu := cluster.NewCluster(u.GpuPerHost, u.GpuPerContainer, u.Hosts...)

	url := fmt.Sprintf("http://%s:%d/setcluster", ip, u.SchedulerPort)

	buf := bytes.NewBuffer([]byte{})
	enc := gob.NewEncoder(buf)
	err := enc.Encode(clu)
	if err != nil {
		log.Panic(err)
	}
	resp, err := http.Post(url, "application/gob", buf)
	if err != nil {
		log.Panicf("error %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		log.Panicf("set cluster post failed, status: %s", resp.Status)
	}
}

func genJobID() string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("hello world %d", time.Now().Unix())))
	byt := h.Sum(nil)
	he := hex.EncodeToString(byt)
	return he[0:10]
}

func (u User) NewJob(jobID string, pj PlannedJob) job.Job {
	return u.NewSingleJob(jobID, pj.Dataset, pj.Steps)
}

func bool2int(x bool) int {
	if x {
		return 1
	}
	return 0
}

func (u User) NewSingleJob(jobID string, dataset *ds.Dataset, steps int) job.Job {
	log.Printf("new job %s for %d steps", jobID, steps)
	job := job.Job{
		Framework:      "megatron-lm",
		Precision:      "fp16",
		BatchSize:      128,
		MicroBatchSize: 8,
		SequenceLength: 1024,
		Dataset:        enwiki,
		Image:          u.Image,
		Model:          "gpt",
		ID:             jobID,
		Steps:          steps,
		ModelSize:      "xl",
		NumLayers:      24,
		VocabSize:      30524,
		Failure:        bool2int(u.SimulateFailure),
	}
	return job
}

const localhost = "127.0.0.1"

type User struct {
	GpuPerHost      int
	GpuPerContainer int
	Hosts           stringlist.Value
	SchedulerPort   int
	Image           string
	PlansFile       string
	SingleTimedJob  bool
	SimulateFailure bool // FIXME: change it to int

	// old flags
	AddJobB bool
	Delay   time.Duration
}

func (u *User) RegisterFlags(flag *flag.FlagSet) {
	flag.IntVar(&u.GpuPerHost, "gpu-per-host", 4, ``)
	flag.IntVar(&u.GpuPerContainer, "gpu-per-container", 4, ``)
	flag.IntVar(&u.SchedulerPort, "scheduler-port", scheduler.DefaultSchedulerPort, ``)
	flag.StringVar(&u.Image, `image`, ``, ``)
	flag.StringVar(&u.PlansFile, `plan`, ``, `path to json file`)
	flag.BoolVar(&u.SingleTimedJob, `timed-job`, false, `only run a single timed job`)
	flag.BoolVar(&u.SimulateFailure, `failure`, false, `simulate failure`)

	// old flags
	flag.BoolVar(&u.AddJobB, `B`, false, ``)
	flag.DurationVar(&u.Delay, "delay", 10*time.Minute, ``)
}

type Plan struct {
	Jobs []PlannedJob `json:"jobs"`
}

type PlannedJob struct {
	Steps   int         `json:"steps"`
	Delay   int         `json:"delay"`
	Dataset *ds.Dataset `json:"dataset,omitempty"`
}

func (u User) RunSingleJob() error {
	f, err := os.Open(u.PlansFile)
	if err != nil {
		return err
	}
	defer f.Close()
	var sp []scalepoint.ScalePoint
	log.Printf("call Decode")
	if err := json.NewDecoder(f).Decode(&sp); err != nil {
		return err
	}
	u.RunJob(sp)
	return nil
}

func (u User) RunJob(sp []scalepoint.ScalePoint) {
	WaitTCP(localhost, u.SchedulerPort)
	u.SetCluster(localhost)
	log.Printf("set cluster done")

	jid := `job-single`
	var daSe *ds.Dataset // dataset default
	steps := 10000       // default number of steps
	jDesc := u.NewSingleJob(jid, daSe, steps)
	j := job.TimedJob{Job: jDesc, Timing: sp}
	u.AddTimedJob(localhost, j)
}

func (u User) RunPlans() error {
	f, err := os.Open(u.PlansFile)
	if err != nil {
		return err
	}
	defer f.Close()
	var p Plan
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return err
	}
	u.Run(p.Jobs...)
	return nil
}

func (u User) Run(ps ...PlannedJob) {
	WaitTCP(localhost, u.SchedulerPort)
	u.SetCluster(localhost)
	log.Printf("set cluster done")

	for i, p := range ps {
		jid := `job-` + str(i+1)
		if i > 0 {
			log.Printf("delay %s before adding job %s", u.Delay, jid)
			time.Sleep(time.Duration(p.Delay) * time.Minute)
		}
		j := u.NewJob(jid, p)
		u.AddJobs(localhost, j)
	}
}

var str = strconv.Itoa

func WaitTCP(host string, port int) bool {
	t0 := time.Now()
	for {
		if _, err := net.Dial("tcp", net.JoinHostPort(host, strconv.Itoa(port))); err == nil {
			break
		}
		log.Printf("waiting for %s:%d, took %s", host, port, time.Since(t0))
		time.Sleep(5 * time.Second)
	}
	log.Printf("tcp://%s:%d is up", host, port)
	return true
}
