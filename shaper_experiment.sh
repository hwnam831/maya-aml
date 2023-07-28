#!/usr/bin/bash

for (( i=$2; i<$3; i++ ))
do
    sudo -E --preserve-env=PATH env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" bash Scripts/Launch.sh --rundir Dist/Release --options "--mode Mask --mask Shaper --ctldir ../../Controller --ctlfile mayaRobust" --logdir /home/hwnam/maya-aml/shaper_logs --tag $1_$i -a $1
done
