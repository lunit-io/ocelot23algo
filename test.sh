#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="4g"

docker volume create ocelot23algo-output

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/input:/input \
        -v ocelot23algo-output:/output/ \
        ocelot23algo

echo "Done initializing container"

docker run --rm \
        -v ocelot23algo-output:/output/ \
        python:3.9-slim cat /output/cell_predictions.json | python -m json.tool

echo "Done running the processing script"

docker run --rm \
        -v ocelot23algo-output:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.9-slim python -c "import json, sys; f = json.load(open('/output/cell_predictions.json')); sys.exit(not len(f)>0);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm ocelot23algo-output
