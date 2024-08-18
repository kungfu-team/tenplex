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
	Config   TrainingConfig
}

func (j Job) createWorkers(jConf *JobConfig, numContainers int, hosts []string) []Container {
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
		workers = append(workers, j.newWorker(i, jConf, host, gpus))
	}
	return workers
}

func (j Job) GenCmd(i int, jConf *JobConfig, host string) []string {
	if jConf.Framework == "megatron-lm" {
		pyHost := OverwriteHost(host, jConf)
		gf := map[string]GenCmdFunc{
			`bert`: GenMegatronLMBERTCmd,
			`gpt`:  GenMegatronLMGPTCmd,
		}
		if g := gf[jConf.Model]; g != nil {
			return g(j.Config, i, jConf.ID, pyHost, jConf)
		}
	} else if jConf.Framework == "deepspeed" {
		return startSSHD
	} else if jConf.Framework == "deepspeed-new-repo" {
		return GenMegatronDeepspeedCommand(j.Config, i, jConf)
	}
	log.Fatalf("invalid framework or model: (%s, %s)", jConf.Framework, jConf.Model)
	return nil
}

func (j Job) newWorker(i int, jConf *JobConfig, host string, gpus []string) Container {
	dockerName := fmt.Sprintf(`trainer-%s-%02d`, jConf.ID, i)
	var pathMaps []PathMap
	if jConf.NoTenplex {
		pathMaps = append(pathMaps,
			PathMap{
				HostPath:      `/mnt/k1d2/megatron-lm`,
				ContainerPath: `/data/dataset`,
			},
			PathMap{
				HostPath:      `/mnt/k1d2/ckpt`,
				ContainerPath: `/data/ckpt`,
			},
		)
	} else {
		pathMaps = append(pathMaps,
			PathMap{
				HostPath:      "/mnt/mlfs",
				ContainerPath: `/data/mlfs`,
			},
			PathMap{
				HostPath:      path.Join(j.HostPath, jConf.ID, str(i), `ckpt`),
				ContainerPath: `/data/ckpt`,
			},
		)
	}

	return Container{
		Name:             dockerName,
		IP:               dockerName,
		GPUs:             gpus,
		Host:             host,
		Cmd:              j.GenCmd(i, jConf, host),
		Rank:             i,
		PathMaps:         pathMaps,
		NetworkInterface: jConf.NetworkInterface,
	}
}

func (j Job) NewCluster(hosts []string, size int, jConf *JobConfig) *ContainerCluster {
	c := jConf.Cluster
	if size*c.GPUsPerContainer > len(hosts)*c.GPUsPerHost {
		log.Panicf("#Nodes %d * GPUs per Container %d > #Hosts %d * %d GPUs per host", size, c.GPUsPerContainer, len(hosts), c.GPUsPerHost)
	}
	return &ContainerCluster{
		Image:     j.Image,
		trainJob:  j,
		Workers:   j.createWorkers(jConf, size, hosts),
		Framework: jConf.Framework,
		User:      jConf.User,
	}
}
