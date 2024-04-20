package job

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path"

	"github.com/kungfu-team/tenplex/tenplex-run/cancelgroup"
	"github.com/kungfu-team/tenplex/tenplex-run/counter"
	"github.com/lgarithm/proc"
	"github.com/lgarithm/proc/experimental"
	"github.com/lgarithm/proc/iostream"
)

type PathMap struct {
	HostPath      string
	ContainerPath string
}

type Container struct {
	Name     string
	IP       string // private IP
	GPUs     []string
	Host     string // Docker Host IP
	Cmd      []string
	Rank     int
	PathMaps []PathMap
}

func (c Container) MapFlags() []string {
	var args []string
	for _, m := range c.PathMaps {
		args = append(args, `-v`, fmt.Sprintf("%s:%s", m.HostPath, m.ContainerPath))
	}
	return args
}

func (c Container) GetCkptDir() string {
	for _, m := range c.PathMaps {
		if m.ContainerPath == `/data/ckpt` {
			return m.HostPath
		}
	}
	return ``
}

type ContainerCluster struct {
	Image     string
	Workers   []Container
	trainJob  Job
	Framework string
	User      string
}

var (
	startSSHD = []string{`/usr/sbin/sshd`, `-e`, `-D`}
)

func (c *ContainerCluster) Check() P {
	hosts := []string{komodo01, komodo02}
	return Seq(
		Par(Pmap(nvidiaSmi, hosts...)...),
		Par(Pmap(dockerPs, hosts...)...),
	)
}

func (c *ContainerCluster) Init() P {
	hosts := []string{komodo01, komodo02}
	return Seq(
		Par(Pmap(nvidiaSmi, hosts...)...),
		Par(Pmap(dockerPs, hosts...)...),
	)
}

func (cluster *ContainerCluster) CreateHostfile() error {
	f, err := os.Create(`hostfile.txt`)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, c := range cluster.Workers {
		// fmt.Fprintf(f, "%s slots=%d\n", c.IP, len(c.GPUs))
		fmt.Fprintf(f, "%s slots=%d\n", c.Name, len(c.GPUs))
	}
	return nil
}

func (cluster *ContainerCluster) SendHostFile(c Container) P {
	return experimental.Scp(proc.At(``, c.Host), `hostfile.txt`, path.Join(c.GetCkptDir(), `hostfile.txt`))
}

func (c *ContainerCluster) RunTrain(jConf *JobConfig) P {
	log.Printf("%s(...)", `RunTrain`)
	if jConf.Framework == "megatron-lm" {
		return c.RunTrainMegatronLM()
	} else if jConf.Framework == "deepspeed" {
		return c.RunTrainDeepspeed(jConf)
	} else if jConf.Framework == "deepspeed-new-repo" {
		return c.RunTrainMegatronLM()
	}
	return nil
}

var (
	Stage      = counter.New()
	GetStageId = Stage.Next
)

func (c *ContainerCluster) RunTrainMegatronLM() P {
	ctx := context.TODO()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stageID := GetStageId()
	workers := c.Workers
	var runs []P
	for i, w := range workers {
		p := c.RunCtx(w, ctx)
		p = Tee2Files(fmt.Sprintf("logs/stage-%02d-worker-%02d", stageID, i), p)
		var err error = errors.New("worker failed")
		i := i
		log.Printf("adding worker %d", i)
		runs = append(runs,
			Seq(
				proc.FnOk(func() {
					log.Printf("RUNNING: %d", i)
				}),
				proc.Ignore(
					Seq(
						p,
						proc.FnOk(func() { err = nil }),
					),
				),
				proc.Fn(func() error {
					log.Printf("one worker (%d) finished with %v", i, err)
					if err != nil {
						cancel()
					}
					return err
				}),
			))
	}
	var cmds []P
	cmds = append(cmds, Par(Cmap(c.MkMountDirs, workers...)...))
	// cmds = append(cmds, Par(cmap(CopyBERTVocab, workers...)...))
	cmds = append(cmds,
		Term(`[*] `, Echo(`starting containers ...`)),
		cancelgroup.CancelGroup(runs, errors.New("worker failed"), cancel),
		Term(`[*] `, Echo(`started containers`)),
	)
	return Seq(cmds...)
}

func (c *ContainerCluster) MkMountDirs(con Container) P {
	return MkMountDirs(con, c.User)
}

func (c *ContainerCluster) RunTrainDeepspeed(jConf *JobConfig) P {
	c.CreateHostfile()
	workers := c.Workers
	cmd := GenDeepspeedCommand(c.trainJob.Config, jConf)
	return Seq(
		Par(Cmap(c.MkMountDirs, workers...)...),
		Par(Cmap(CopyEnWikiData, workers...)...),
		Par(Cmap(c.SendHostFile, workers...)...),
		Par(Cmap(c.Run, workers...)...),
		DockerExec(workers[0], cmd...),
	)
}

func (c *ContainerCluster) Repartition(globalStep, srcMPSize, mpSize int) P {
	c.CreateHostfile()
	workers := c.Workers
	delay := Proc{
		Prog: `sleep`, Args: []string{`5`},
	}

	newWorkers := workers[:mpSize]

	return Seq(
		Term(`[*] `, Echo(`mkdirs ...`)),
		Par(Cmap(c.MkMountDirs, workers...)...),
		Term(`[*] `, Echo(`send hostfiles ...`)),
		Par(Cmap(c.SendHostFile, workers...)...),
		Term(`[*] `, Echo(`starting containers ...`)),
		Par(Cmap(c.Run, workers...)...), // starts daemon
		Term(`[*] `, Echo(`started containers`)),
		Ignore(DockerExec(workers[0], `chown`, `-R`, `root:root`, `/usr/`)), // FIXME: fix the Dockerfile
		Term(`[*] `, Echo(`chown done`)),
		Par(
			Seq(
				Par(Cmap(ServeRepartition, workers...)...),
				Echo(`repartion server stopped.`),
			),
			Seq(
				Shell(delay.CmdCtx(context.TODO())), // TODO: pull server in the client
				Par(cfor(Repartition(globalStep, srcMPSize, mpSize), newWorkers...)...),
				Echo(`repartion finished.`),
				Par(Cmap(StopRepartitionServer, workers...)...),
				Echo(`requested repartion servers to stop.`),
			),
		),
	)
}

// map P to a list of Containers
func Cmap(f func(Container) P, cs ...Container) []P {
	var ps []P
	for _, h := range cs {
		ps = append(ps, f(h))
	}
	return ps
}

// map P to a list of Containers, with index
func cfor(f func(int, Container) P, cs ...Container) []P {
	var ps []P
	for i, h := range cs {
		ps = append(ps, f(i, h))
	}
	return ps
}

func (c *ContainerCluster) Stop() P {
	cs := c.Workers
	userStop := func(con Container) P {
		return Stop(con, c.User)
	}
	return Par(Cmap(userStop, cs...)...)
}

func Tee2Files(name string, p P) P {
	o := iostream.Open2Lazy(name+`.out.log`, name+`.err.log`)
	return proc.Tee(p, o.StdWriters())
}
