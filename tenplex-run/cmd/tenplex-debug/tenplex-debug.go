package main

import (
	"bytes"
	"flag"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
	"log"
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

	cfg := job.MDPConfig{
		NumNodes:             1,
		GPUPerNode:           4,
		PipelineParallelSize: 1,
		ModelParallelSize:    1,

		TrainIters: 10_000,

		LogInterval:  10,
		SaveInterval: 5000,
		EvalInterval: 10000,

		Precision: d.JobConfig.Precision,
	}

	// log.Printf("JOB CONFIG %+v", d.JobConfig)

	j := job.Job{
		Image:    d.JobConfig.Image,
		HostPath: path.Join(d.JobConfig.TenplexPrefix, `training`),
		Config:   cfg,
	}

	cmd := j.GenCmd(0, &d.JobConfig, d.JobConfig.Cluster.Hosts[0])
	for i, arg := range cmd {
		cmd[i] = strings.Replace(arg, "pretrain_gpt.py", "hash_samples.py", 1)
	}

	pathMaps := []job.PathMap{
		{
			HostPath:      "/mnt/mlfs",
			ContainerPath: `/data/mlfs`,
		},
		{
			HostPath:      path.Join(j.HostPath, d.JobConfig.ID, `0`, `ckpt`),
			ContainerPath: `/data/ckpt`,
		},
	}

	dockerCmd := []string{`docker`, `run`}
	for _, pm := range pathMaps {
		dockerCmd = append(dockerCmd, `-v`, pm.HostPath+`:`+pm.ContainerPath)
	}
	dockerCmd = append(dockerCmd, []string{
		`--gpus`, `'"device=0"'`,
		// `--network`, `tenplex`,
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

	// log.Printf("CMD %v", cmd)
	// log.Printf("DOCKER CMD %v", dockerCmd)
	// log.Printf("PATH MAPS %v", pathMaps)

	execCmd := exec.Command(`bash`, `-c`, strings.Join(dockerCmd, ` `))
	// execCmd := exec.Command(`bash`, `-c`, `docker run --rm -t kungfu.azurecr.io/mw-megatron-lm-23.06-debug:latest python --version`)
	log.Printf("EXEC CMD %v", execCmd)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	execCmd.Stderr = &stderr
	execCmd.Stdout = &stdout
	err := execCmd.Run()
	if err != nil {
		log.Printf("Stderr: %s", stderr.String())
		log.Fatal(err)
	}
	log.Printf("Stdout: %s", stdout.String())
}
