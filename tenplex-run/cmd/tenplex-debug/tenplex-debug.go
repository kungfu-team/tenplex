package main

import (
	"bytes"
	"flag"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/runop"
	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
	"log"
	"os"
	"os/exec"
	"path"
	"strings"
)

type TenplexRunFlags struct {
	job.JobConfig
}

func (d *TenplexRunFlags) RegisterFlags(flag *flag.FlagSet) {
	d.JobConfig.RegisterFlags(flag)
	structflag.RegisterFlags(d, flag)
}

func main() {
	var d TenplexRunFlags
	d.RegisterFlags(flag.CommandLine)
	flag.Parse()
	d.ParseSchedule()
	d.ParseParaConfig()
	d.Validate()

	id := `tenplexdeb`
	d.ID = id
	d.JobConfig.ID = id
	log.Printf("JOB CONFIG %+v", d.JobConfig)

	cfg := job.MDPConfig{
		NumNodes:             1,
		GPUPerNode:           d.Cluster.GPUsPerContainer,
		PipelineParallelSize: 1,
		ModelParallelSize:    1,

		TrainIters: 10_000,

		LogInterval:  50_000,
		SaveInterval: 50_000,
		EvalInterval: 50_000,

		Precision: d.JobConfig.Precision,
	}

	j := job.Job{
		Image:    d.JobConfig.Image,
		HostPath: path.Join(d.JobConfig.TenplexPrefix, `training`),
		Config:   cfg,
	}

	if !d.NoTenplex {
		err := runop.AddDataset(1, 0, &d.JobConfig)
		if err != nil {
			log.Fatal(err)

		}
	}

	cmd := j.GenCmd(0, &d.JobConfig, d.JobConfig.Cluster.Hosts[0])
	for i, arg := range cmd {
		cmd[i] = strings.Replace(arg, "pretrain_gpt.py", "hash_batch.py", 1)
	}

	pathMaps := []job.PathMap{
		{
			HostPath:      path.Join(j.HostPath, d.JobConfig.ID, `0`, `ckpt`),
			ContainerPath: `/data/ckpt`,
		},
	}
	if d.NoTenplex {
		pathMaps = append(pathMaps,
			job.PathMap{
				HostPath:      `/mnt/k1d2/megatron-lm`,
				ContainerPath: `/data/dataset`,
			})
	} else {
		pathMaps = append(pathMaps,
			job.PathMap{
				HostPath:      "/mnt/mlfs",
				ContainerPath: `/data/mlfs`,
			})
	}

	dockerCmd := []string{`docker`, `run`}
	for _, pm := range pathMaps {
		dockerCmd = append(dockerCmd, `-v`, pm.HostPath+`:`+pm.ContainerPath)
	}
	dockerCmd = append(dockerCmd, []string{
		`--gpus`, `'"device=0"'`,
		`--network`, `host`,
		`--name`, d.ID,
		`--rm`,
		`--env`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
		`--env`, `PYTHONUNBUFFERED=1`,
		`--ulimit`, `memlock=-1`,
		`--shm-size`, `1g`,
		`--expose`, `6000`,
		`--device`, `/dev/infiniband`,
		`-t`, d.JobConfig.Image,
	}...)
	dockerCmd = append(dockerCmd, cmd...)

	execCmd := exec.Command(`bash`, `-c`, strings.Join(dockerCmd, ` `))
	log.Printf(`EXEC CMD """%v"""`, execCmd)

	if err := os.WriteFile("exec.txt", []byte(execCmd.String()), 0666); err != nil {
		log.Fatal(err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	execCmd.Stderr = &stderr
	execCmd.Stdout = &stdout
	err := execCmd.Run()
	log.Printf("Stdout: %s", stdout.String())
	if err != nil {
		log.Printf("Stderr: %s", stderr.String())
		log.Fatal(err)
	}
}
