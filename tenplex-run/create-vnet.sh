#!/bin/sh
set -e

name=elastique

docker network create --driver overlay --scope swarm --attachable elastique

# TODO: extract Subnet from JSON
docker network inspect $name
