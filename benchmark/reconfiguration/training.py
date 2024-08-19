import argparse
import subprocess
import time


def create_cmd(worker: int, image: str, model_size: str):
    docker_cmd = (
        "docker run "
        "--network host "
        "-v /mnt/LSDSDataShare/projects/tenplex/megatron-lm/gpt:/data/gpt "
        "-v /mnt/k1d2/ckpt:/data/ckpt "
        "--device /dev/infiniband/uverbs0 "
        "--expose 29400 "
        f"--gpus all "
        f"--name worker{worker} "
        "--ulimit memlock=-1 "
        "--shm-size=1gb "
        "--env GLOO_SOCKET_IFNAME=eth0 "
        "-d "
        f"-t {image} "
    )
    if model_size == "large":
        docker_cmd += "./run_gpt.sh"
    elif model_size == "xl":
        docker_cmd += "./run_gpt_xl.sh"
    else:
        raise NotImplementedError

    return docker_cmd


def main():
    hosts = ["komodo01", "komodo02", "komodo03", "komodo04"]
    num_hosts = len(hosts)
    image = "kungfu.azurecr.io/mw-megatron-deepspeed-update:latest"
    model_size = "xl"
    parser = argparse.ArgumentParser(description="Deepspeed")
    parser.add_argument(
        "--scale-up",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    scale_up = args.scale_up

    # Pull image
    for host in hosts:
        subprocess.run(f"ssh {host} docker pull {image}".split(" "), check=True)

    if scale_up:
        # Start worker
        for i in range(num_hosts // 2):
            docker_cmd = create_cmd(i, image, model_size)
            host = hosts[i]
            subprocess.run(f"ssh {host} {docker_cmd}".split(" "), check=True)

        # Wait
        time.sleep(4 * 60)

        # Scale up
        for i in range(num_hosts // 2, num_hosts):
            docker_cmd = create_cmd(i, image, model_size)
            host = hosts[i]
            subprocess.run(f"ssh {host} {docker_cmd}".split(" "), check=True)
        print(f"finished docker run -d {time.time()}")
    else:
        # Start worker
        for i in range(num_hosts):
            docker_cmd = create_cmd(i, image, model_size)
            host = hosts[i]
            subprocess.run(f"ssh {host} {docker_cmd}".split(" "), check=True)

        # Wait
        time.sleep(4 * 60)

        # Scale down
        for i in range(num_hosts // 2, num_hosts):
            docker_cmd = create_cmd(i, image, model_size)
            host = hosts[i]
            subprocess.run(
                f"ssh {host} docker ps -aq -f name='worker' | xargs docker stop".split(
                    " "
                ),
                check=False,
            )
        print(f"finished docker stop {time.time()}")


if __name__ == "__main__":
    main()
