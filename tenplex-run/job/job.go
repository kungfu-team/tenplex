package job

import (
	"fmt"
	"log"
	"path"
)

// Job defines a elastique training job
type Job struct {
	Image    string
	HostPath string
	Config   MDPConfig
}

func (j Job) createWorkers(jConf *JobConfig, numContainers int, hosts []string, jobID string) []Container {
	var workers []Container
	containersPerHost := jConf.Cluster.GPUsPerHost / jConf.Cluster.GPUsPerContainer
	for i := 0; i < numContainers; i++ {
		hostIdx := i / containersPerHost
		host := hosts[hostIdx]
		l := i % containersPerHost
		var gpus []string
		for k := l * jConf.Cluster.GPUsPerContainer; k < (l+1)*jConf.Cluster.GPUsPerContainer; k++ {
			gpus = append(gpus, str(k))
		}
		workers = append(workers, j.newWorker(i, jConf, jobID, host, hosts[0], gpus...))
	}
	return workers
}

func (j Job) newWorker(i int, jConf *JobConfig, jobID string, host string, masterAddr string, gpus ...string) Container {
	var cmd []string
	if jConf.Framework == "megatron-lm" {
		pyHost := OverwriteHost(host, jConf)
		if jConf.Model == "bert" {
			cmd = GenMegatronLMBERTCmd(j.Config, i, jobID, pyHost, jConf)
		} else if jConf.Model == "gpt" {
			cmd = GenMegatronLMGPTCmd(j.Config, i, jobID, pyHost, jConf, masterAddr)
		}
	} else if jConf.Framework == "deepspeed" {
		cmd = startSSHD
	} else if jConf.Framework == "deepspeed-new-repo" {
		cmd = GenMegatronDeepspeedCommand(j.Config, i, jobID, jConf)
	}
	dockerName := fmt.Sprintf(`trainer-%s-%02d`, jobID, i)
	return Container{
		Name: dockerName,
		IP:   dockerName,
		// IP:   host,
		GPUs: gpus,
		Host: host,
		Cmd:  cmd,
		Rank: i,

		PathMaps: []PathMap{
			// {
			//         HostPath:      path.Join(j.HostPath, jobID, str(i), `data`),
			//         ContainerPath: `/data/dataset`,
			// },
			{
				HostPath:      `/data/megatron-lm/gpt-2`,
				ContainerPath: `/data/dataset`,
			},
			{
				HostPath:      path.Join(j.HostPath, jobID, str(i), `ckpt`),
				ContainerPath: `/data/ckpt`,
			},
			{
				HostPath:      path.Join("/mnt/mlfs/job", jobID),
				ContainerPath: `/data/mlfs`,
			},
		},
	}
}

func (j Job) NewCluster(hosts []string, size int, jConf *JobConfig, jobID string) *ContainerCluster {
	if size > len(hosts)*jConf.Cluster.GPUsPerHost {
		log.Panicf("size %d > num hosts %d * %d gpus per host", size, len(hosts), jConf.Cluster.GPUsPerHost)
	}
	return &ContainerCluster{
		Image:     j.Image,
		trainJob:  j,
		Workers:   j.createWorkers(jConf, size, hosts, jobID),
		Framework: jConf.Framework,
		User:      jConf.User,
	}
}
