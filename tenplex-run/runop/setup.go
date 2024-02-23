package runop

import (
	"fmt"
	"log"
	"path"
	"time"

	"github.com/kungfu-team/tenplex/tenplex-run/docker"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/lgarithm/proc"
	"github.com/lgarithm/proc/experimental"
)

var (
	seq    = proc.Seq
	echo   = proc.Echo
	psh    = proc.Psh
	ignore = proc.Ignore
	at     = proc.At

	runScript = experimental.RunScript
)

func createCluster(jobConf *job.JobConfig, paraConf *job.ParallelismConfig, hosts []string, maxTrainStep int) *job.ContainerCluster {
	gpusPerContainer := jobConf.Cluster.GPUsPerContainer
	numNodes := int(paraConf.Size / gpusPerContainer)
	if numNodes < 1 {
		numNodes = 1
	}

	cfg := job.MDPConfig{
		NumNodes:             numNodes,
		GPUPerNode:           gpusPerContainer,
		PipelineParallelSize: paraConf.PPSize,
		ModelParallelSize:    paraConf.MPSize,

		TrainIters: maxTrainStep,

		LogInterval:  5,
		SaveInterval: 500,
		EvalInterval: 500,

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
		pc := proc.At(user, h).PC // TODO: add admin user
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

func copyDir(uh proc.UserHost, local, remote string) P {
	p0 := Proc{
		Prog: `rsync`,
		Args: []string{`-r`, `--exclude=.git`, local, uh.User + `@` + uh.Host + ":" + remote},
	}
	return seq(
		psh(p0),
		echo("done rsync: "+local+" to "+uh.Host+":"+remote),
	)
}

func cloneTransformerCkpts(host, user string) P {
	dir := "~/.tenplex/transformer-checkpoint"
	pRm := Proc{
		Prog: `rm`,
		Args: []string{`-rf`, dir},
		Host: host,
		User: user,
	}
	pClone := Proc{
		Prog: `git`,
		Args: []string{`clone`, `git@github.com:kungfu-team/transformer-checkpoint.git`, dir},
		Host: host,
		User: user,
	}
	return seq(ignore(ssh(pRm)), ssh(pClone))
}

func PrepareVMs(jobConf *job.JobConfig) {
	const tenplexDir = "~/.tenplex"

	tenplexBinDir := path.Join(tenplexDir, "bin")
	var ps []P
	for _, host := range jobConf.Cluster.Hosts {
		prefix := fmt.Sprintf("[%s]/%s ", host, `parepare--machines`)
		rm := Proc{
			Prog: `rm`,
			Args: []string{`-r`, tenplexBinDir},
			Host: host,
			User: jobConf.User,
		}
		mk := Proc{
			Prog: `mkdir`,
			Args: []string{`-p`, tenplexBinDir},
			Host: host,
			User: jobConf.User,
		}
		s := seq(
			ignore(ssh(rm)),
			ssh(mk),
			term(`clone: `, cloneTransformerCkpts(host, jobConf.User)),
		)
		ps = append(ps, term(prefix, s))
	}

	if r := run(par(ps...), &stdio); r.Err != nil {
		panic(r.Err)
	}
}

func StopContainers(hosts []string, user string) {
	var ps []P
	for _, host := range hosts {
		prefix := fmt.Sprintf("[%s]/%s ", host, `stop-containers`)
		script := `docker ps -f 'name=trainer' -q | xargs docker stop`
		p := runScript(at(user, host), script, `stop-container.sh`, false)
		ps = append(ps, term(prefix, ignore(p)))
	}
	if r := run(par(ps...), &stdio); r.Err != nil {
		panic(r.Err)
	}
}

func CleanMachines(jobConf *job.JobConfig) {
	StopContainers(jobConf.Cluster.Hosts, jobConf.User)
	var ps []P
	for _, host := range jobConf.Cluster.Hosts {
		prefix := fmt.Sprintf("[%s]/%s ", host, `clean-machines`)

		// clean training directory
		p := Proc{
			Prog: `sudo`,
			Args: []string{`rm -r ~/.tenplex/training/*`},
			Host: host,
			User: jobConf.User,
		}
		ps = append(ps, term(prefix, ignore(ssh(p))))

		// restart MLFSd
		p = Proc{
			Prog: `sudo`,
			Args: []string{`systemctl restart mlfs`},
			Host: host,
			User: jobConf.User,
		}
		ps = append(ps, term(prefix, ignore(ssh(p))))
	}

	if r := run(par(ps...), &stdio); r.Err != nil {
		panic(r.Err)
	}
}

func Main(jobConf *job.JobConfig) {
	RoundID.Reset()
	CleanMachines(jobConf)
	// SetupSwarm(jobConf)
	PrepareVMs(jobConf)
	PullImages(jobConf)
	ScalingTraining(jobConf)
}
