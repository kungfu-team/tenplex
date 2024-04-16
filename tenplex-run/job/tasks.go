package job

import (
	"fmt"
	"path"
	"strings"
)

const network = `host`

// const network = `tenplex`

func mkdir(h string, user string, d string) P {
	p := Proc{
		Prog: `mkdir`,
		Args: []string{`-p`, d},
		Host: h,
		User: user,
	}
	return Ssh(p)
}

// Options graveyard
// `--env`, `NCCL_DEBUG=INFO`,
// `--env`, `NCCL_DEBUG_SUBSYS=ALL`,
// `--env`, `NCCL_P2P_LEVEL=PXB`,
// `--env`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`,
// `--env`, `TORCH_CPP_LOG_LEVEL=INFO`,
// `--env`, `GLOO_SOCKET_IFNAME=eth0`,
// `--ulimit`, `stack=67108864`,
// `--ipc`, `host`,
// `--cap-add`, `IPC_LOCK`,

func (cluster *ContainerCluster) Run(c Container) P {
	args := []string{`run`}
	args = append(args, c.MapFlags()...)
	if cluster.Framework == "deepspeed" {
		args = append(args, `-d`)
	}
	args = append(args, []string{
		`--gpus`, `'"` + fmt.Sprintf("device=%s", strings.Join(c.GPUs, `,`)) + `"'`,
		`--network`, network,
		`--name`, c.Name,
		`--rm`,
		`--env`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
		`--env`, `PYTHONUNBUFFERED=1`,
		`--ulimit`, `memlock=-1`,
		`--shm-size`, `1g`,
		`--expose`, `6000`,
		`--device`, `/dev/infiniband`,
		`-t`, cluster.Image,
	}...)
	args = append(args, c.Cmd...)

	p := Proc{
		Prog: `docker`,
		Args: args,
		Host: c.Host,
		User: cluster.User,
	}
	ps1 := fmt.Sprintf("%s %s ", c.Host, c.IP)
	return Term(ps1, Seq(
		Echo(fmt.Sprintf("docker %s", strings.Join(args, ` `))),
		Ssh(p),
	))
}

func Sleep(t int) func(c Container) P {
	return func(c Container) P {
		command := []string{
			`sleep`,
			str(t),
		}
		return DockerExec(c, command...)
	}
}

func ServeRepartition(c Container) P {
	command := []string{
		`python`,
		`/workspace/training-state/training_state/serve_ckpt.py`,
		`--port`, `21234`,
		`--data`, `/data/ckpt`,
	}
	ps1 := fmt.Sprintf("[s@%s] ", c.IP)
	return Term(ps1, DockerExec(c, command...))
}

func Repartition(globalStep, srcMPSize, targetMPSize int) func(int, Container) P {
	return func(i int, c Container) P {
		prefix := `/data/ckpt/`
		inDir := path.Join(prefix, `global_step`+str(globalStep))
		outDir := path.Join(prefix, `repartition`)
		downloadCmd := []string{
			`python`,
			`/workspace/training-state/training_state/repartition.py`,
			`--output-dir`, outDir,
			`--shapes-dir`, `/workspace/training-state/scripts/shapes`,
			`--source-mp-size`, str(srcMPSize),
			`--target-mp-size`, str(targetMPSize),
			`--rank`, str(i),
			`--global-step`, str(globalStep),
			`--hostfile`, `/data/ckpt/hostfile.txt`,
			`--port 21234`,
		}

		ps1 := fmt.Sprintf("[c@%s] ", c.IP)
		return Term(ps1, Seq(
			DockerExec(c, `mkdir`, `-p`, outDir),
			DockerExec(c, downloadCmd...),
			DockerExec(c, `mv`, path.Join(outDir, `/latest`), prefix),
			DockerExec(c, `mv`, path.Join(outDir, `/latest_checkpointed_iteration.txt`), prefix),
			DockerExec(c, `rm`, `-rf`, inDir),
			DockerExec(c, `mv`, outDir, inDir),
			Echo(`finished moving state`),
		))
	}
}

func StopRepartitionServer(c Container) P {
	command := []string{
		`curl`,
		fmt.Sprintf(`http://127.0.0.1:%d/quit`, 21234),
	}
	ps1 := fmt.Sprintf("[stop@%s] ", c.IP)
	return Term(ps1, Seq(
		DockerExec(c, command...),
	))
}

func DockerExec(c Container, cmd ...string) P {
	args := []string{`exec`, `-t`, c.Name}
	args = append(args, cmd...)
	p := Proc{
		Prog: `docker`,
		Args: args,
		Host: c.Host,
	}
	ps1 := fmt.Sprintf("%s ", c.Host)
	return Term(ps1, Seq(
		Echo(fmt.Sprintf("docker %s", strings.Join(args, ` `))),
		Ssh(p),
	))
}

func MkMountDirs(c Container, user string) P {
	var ps []P
	for _, m := range c.PathMaps {
		if strings.Contains(m.HostPath, "mnt/mlfs") {
			continue
		}
		ps = append(ps,
			Seq(
				Echo(fmt.Sprintf("mkdir -p %s:%s", c.Host, m.HostPath)),
				mkdir(c.Host, user, m.HostPath),
			),
		)
	}
	return Seq(ps...)
}

func CopyFile(host, source, destination string) P {
	p := Proc{
		Prog: `cp`,
		Args: []string{`-u`, source, destination},
		Host: host,
	}

	return Ssh(p)
}

func CopyEnWikiData(c Container) P {
	dataPathMap := c.PathMaps[0]
	dataHostPath := `/data/megatron-lm/gpt-2`
	ps := []P{
		CopyFile(c.Host, path.Join(dataHostPath, `gpt2-merges.txt`), dataPathMap.HostPath),
		CopyFile(c.Host, path.Join(dataHostPath, `gpt2-vocab.json`), dataPathMap.HostPath),
		CopyFile(c.Host, path.Join(dataHostPath, `my-gpt2_text_document.bin`), dataPathMap.HostPath),
		CopyFile(c.Host, path.Join(dataHostPath, `my-gpt2_text_document.idx`), dataPathMap.HostPath),
	}
	return Seq(ps...)
}

func CopyBERTWikiData(c Container) P {
	dataPathMap := c.PathMaps[0]
	dataHostPath := `/data/megatron-lm/bert`
	ps := []P{
		CopyFile(c.Host, path.Join(dataHostPath, `bert-large-uncased-vocab.txt`), dataPathMap.HostPath),
		CopyFile(c.Host, path.Join(dataHostPath, `bert_text_sentence.bin`), dataPathMap.HostPath),
		CopyFile(c.Host, path.Join(dataHostPath, `bert_text_sentence.idx`), dataPathMap.HostPath),
	}
	return Seq(ps...)
}

func CopyBERTVocab(c Container) P {
	dataPathMap := c.PathMaps[0]
	dataHostPath := `/data/megatron-lm/bert`
	ps := []P{
		CopyFile(c.Host, path.Join(dataHostPath, `bert-large-uncased-vocab.txt`), dataPathMap.HostPath),
	}
	return Seq(ps...)
}

func Stop(c Container, user string) P {
	p := Proc{
		Prog: `docker`,
		Args: []string{
			`rm`,
			`-f`,
			c.Name,
		},
		Host: c.Host,
		User: user,
	}
	ps1 := fmt.Sprintf("%s ", c.Host)
	return Term(ps1, Ssh(p))
}

func nvidiaSmi(h string) P {
	p := Proc{
		Prog: `nvidia-smi`,
		Host: h,
	}
	ps1 := fmt.Sprintf("%s ", h)
	return Term(ps1, Ssh(p))
}

func dockerPs(h string) P {
	p := Proc{
		Prog: `docker`,
		Args: []string{`ps`},
		Host: h,
	}
	ps1 := fmt.Sprintf("%s ", h)
	return Term(ps1, Ssh(p))
}
