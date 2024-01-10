#!/bin/sh

# https://tenplex.blob.core.windows.net/public/deb/Release
# https://tenplex.blob.core.windows.net/public/deb/Packages
# https://tenplex.blob.core.windows.net/public/deb/mlfs_0.0.1-git-main-rev1-97718f7_amd64.deb

install_mlfs() {
    echo 'deb https://tenplex.blob.core.windows.net/public/deb ./' | sudo tee /etc/apt/sources.list.d/tenplex.list
    curl -s https://tenplex.blob.core.windows.net/public/deb/tenplex.gpg | sudo apt-key add -
    sudo apt update
    sudo apt remove -y mlfs # TODO: fix deb package version number
    sudo apt reinstall -y mlfs
    sudo systemctl stop mlfs
    sudo systemctl start mlfs
    mlfs-admin
}

install_mlfs
