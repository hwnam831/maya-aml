#!/bin/bash

for (( i=$2; i<$3; i++ ))
do
    sudo -E --preserve-env=PATH env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" bash Scripts/Launch.sh --rundir Dist/Release --options "--mode Mask --mask AML --ctldir ../../Controller --ctlfile mayaRobust" --logdir /mydata/maya-aml/aml_logs --tag $1_$i -a $1
done
