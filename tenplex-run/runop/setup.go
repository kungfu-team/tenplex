package runop

import (
	"log"
	"os"
	"path"
	"time"

	"github.com/kungfu-team/tenplex/tenplex-run/docker"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/para_config"
	"github.com/lgarithm/proc"
	"github.com/lgarithm/proc/experimental"
)

var (
	seq    = proc.Seq
	ignore = proc.Ignore
	at     = proc.At

	runScript = experimental.RunScript
)

func createCluster(jobConf *job.JobConfig, paraConf *para_config.MDP, hosts []string, maxTrainStep int) *job.ContainerCluster {
	gpusPerContainer := jobConf.Cluster.GPUsPerContainer
	numNodes := int(paraConf.GetTotalSize() / gpusPerContainer)
	if numNodes < 1 {
		numNodes = 1
	}

	cfg := job.TrainingConfig{
		NumNodes:   numNodes,
		GPUPerNode: gpusPerContainer,
		MDP:        *paraConf,

		TrainIters: maxTrainStep,

		LogInterval:  10,
		SaveInterval: 5000,
		EvalInterval: 10000,

		Precision: jobConf.Precision,
	}

	j := job.Job{
		Image:    jobConf.Image,
		HostPath: path.Join(jobConf.TenplexPrefix, `training`),
		Config:   cfg,
	}

	return j.NewCluster(hosts, numNodes, jobConf)
}

func dockerPull(image, user string) func(string) P {
	return func(h string) P {
		pc := at(user, h).PC // TODO: add admin user
		pc = proc.WithTerm(pc)
		pc = experimental.WithLog(pc)
		return pc(`docker`, `pull`, image)
	}
}

func PullImages(jobConf *job.JobConfig) {
	log.Printf("pulling image")
	op := dockerPull(jobConf.Image, jobConf.User)
	for i := 0; i < 10; i++ {
		r := run(par(pmap(op, jobConf.Cluster.Hosts...)...), &stdio)
		if r.Err == nil {
			break
		}

		log.Printf("%v", r.Err)
		time.Sleep(5 * time.Second)
	}
}

func SetupSwarm(jobConf *job.JobConfig) {
	p := docker.SetupSwarm(jobConf.User, jobConf.Cluster.Hosts, jobConf.DockerNetwork)
	if r := run(p, &stdio); r.Err != nil {
		panic(r.Err)
	}
}

func cloneTransformerCkpts(rpc proc.CreatePFn) P {
	dir := "~/.tenplex/transformer-checkpoint"
	return seq(
		ignore(rpc(`rm`, `-rf`, dir)),
		rpc(`git`, `clone`, `git@github.com:kungfu-team/transformer-checkpoint.git`, dir),
	)
}

func PrepareVMs(jobConf *job.JobConfig) {
	const tenplexDir = "~/.tenplex"

	tenplexBinDir := path.Join(tenplexDir, "bin")
	var ps []P
	for _, host := range jobConf.Cluster.Hosts {
		rpc := at(jobConf.User, host).PC
		s := seq(
			ignore(rpc(`rm`, `-r`, tenplexBinDir)),
			rpc(`mkdir`, `-p`, tenplexBinDir),
			term(`clone: `, cloneTransformerCkpts(rpc)),
		)
		ps = append(ps, term(ps1(`PrepareVMs`, host), s))
	}

	if r := run(par(ps...), &stdio); r.Err != nil {
		panic(r.Err)
	}
}

func StopContainers(hosts []string, user string) {
	var ps []P
	for _, host := range hosts {
		script := `docker ps -f 'name=trainer' -q | xargs docker stop`
		p := runScript(at(user, host), script, `stop-container.sh`, false)
		ps = append(ps, term(ps1(`StopContainers`, host), ignore(p)))
	}
	if r := run(par(ps...), &stdio); r.Err != nil {
		panic(r.Err)
	}
}

func CleanMachines(jobConf *job.JobConfig) {
	StopContainers(jobConf.Cluster.Hosts, jobConf.User)
	var ps []P
	for _, host := range jobConf.Cluster.Hosts {
		prefix := ps1(`CleanMachines`, host)
		rpc := at(jobConf.User, host).PC
		ps = append(ps,
			// clean training directory
			term(prefix, ignore(rpc(`sudo`, `rm`, `-r`, `~/.tenplex/training/*`))),
			// restart MLFSd
			term(prefix, ignore(rpc(`sudo`, `systemctl`, `restart`, `mlfs`))),
		)
	}
	if r := run(par(ps...), &stdio); r.Err != nil {
		panic(r.Err)
	}
}

func collectLogs(jobConf *job.JobConfig) {
	var ps []P
	for _, h := range jobConf.Cluster.Hosts {
		remote := h + `:` + path.Join(`.tenplex/training`, jobConf.ID)
		local := path.Join(`training`, jobConf.ID)
		if err := os.MkdirAll(path.Dir(local), os.ModePerm); err != nil {
			log.Printf("`mkdir -p %s` failed: %v", local, err)
		}
		log.Printf("will collect log %s -> %s", remote, local)
		p := proc.PC(`scp`, `-r`, remote, local)
		ps = append(ps, ignore(p))
	}
	if r := run(par(ps...), &proc.Stdio); r.Err != nil {
		log.Printf("collect logs failed")
	}
}

func Main(jobConf *job.JobConfig) {
	RoundID.Reset()
	job.Stage.Reset()
	CleanMachines(jobConf)
	// SetupSwarm(jobConf)
	PrepareVMs(jobConf)
	PullImages(jobConf)
	ScalingTraining(jobConf)
	collectLogs(jobConf)
}
