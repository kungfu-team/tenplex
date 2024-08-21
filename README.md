# Tenplex
Tenplex is a state management library for DL systems that enables jobs to change their parallelism dynamically after the GPU allocation changes at runtime.

## Install

### Prerequisites
- [Go](https://go.dev/doc/install)
- [Docker](https://docs.docker.com/desktop/install/linux-install)

### Install tenplex-run
```bash
git clone https://github.com/kungfu-team/tenplex
cd tenplex
make install
```

### Install Tensor Store (mlfs)
```bash
echo "deb https://europe-west2-apt.pkg.dev/projects/tenplex tenplex main" | sudo tee /etc/apt/sources.list.d/tenplex.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/packages-cloud-google-apt.gpg >/dev/null
sudo apt update
sudo apt install -y mlfs
```
## Examples
Examples are in the `benchmark` directory. For instance, to run the dynamic resources benchmark in `benchmark/dynamic_resources`, just execute `./run.sh` in the directory.

## Citation
If you use Tenplex for your research, please cite our [paper](https://arxiv.org/abs/2312.05181):

@inproceedings{wagenlander2024tenplex,
  title={Tenplex: Dynamic Parallelism for Deep Learning using Parallelizable Tensor Collections},
  author={Marcel Wagenlander, Guo Li, Bo Zhao, Luo Mai, Peter Pietzuch},
  booktitle={Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles},
  year={2024}
}
