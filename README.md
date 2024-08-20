# Tenplex
Dynamic resources changes for multi-dimensional parallelism training

## Install

```
pip3 install tenplex -i https://pkgs.dev.azure.com/gli7/releases/_packaging/nightly/pypi/simple -U
```

```
echo "deb https://europe-west2-apt.pkg.dev/projects/tenplex tenplex main" | sudo tee /etc/apt/sources.list.d/tenplex.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/packages-cloud-google-apt.gpg >/dev/null
sudo apt update
sudo apt install -y mlfs
```
