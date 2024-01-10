#!/bin/sh
set -e

tag=$(cat $(dirname $0)/ubuntu/1804/tf.tag.txt)

docker_run_flags() {
    # echo --privileged
    echo --cap-add SYS_ADMIN
    echo --device /dev/fuse

    # https://forum.rclone.org/t/fusermount-permission-denied-in-docker-rclone/13914/6
    echo --security-opt apparmor:unconfine # For FUSE

    # https://medium.com/swlh/docker-and-systemd-381dfd7e4628
    echo -v /sys/fs/cgroup/:/sys/fs/cgroup:ro # For systemd

    echo -v $PWD/benchmarks:/benchmarks
    echo --rm
    echo -d
    echo --name mlfs
}

docker rm -f mlfs
docker run $(docker_run_flags) --gpus "device=0" -it $tag /sbin/init
