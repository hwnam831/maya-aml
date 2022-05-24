#!/usr/bin/bash
for i in {1..500}
do
    sudo -E --preserve-env=PATH env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" bash Scripts/Launch.sh --rundir Dist/Release --options "--mode Baseline" --logdir /home/hwnam/maya-aml/logs --tag $1_$i -a $1
done