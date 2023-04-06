#!/usr/bin/env bash

./build.sh

# Change the name of your docker algo. or compressed algo.
docker save ocelot23algo | gzip -c > ocelot23algo.tar.gz
