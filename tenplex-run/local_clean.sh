#!/bin/bash

docker ps -f "name=trainer" -q | xargs docker stop
sudo rm -r ~/.tenplex/training/*
