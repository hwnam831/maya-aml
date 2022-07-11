#!/usr/bin/bash

for i in {$2..$3}
do
    sudo -E --preserve-env=PATH env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" bash Scripts/Launch.sh --rundir Dist/Release --options "--mode Mask --mask AML --ctldir ../../Controller --ctlfile mayaRobust" --logdir /home/hwnam/maya-aml/aml_logs --tag $1_$i -a $1
done
