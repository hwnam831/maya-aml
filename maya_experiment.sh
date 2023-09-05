#!/usr/bin/bash
for (( i=$2; i<$3; i++ ))
do
    sudo -E --preserve-env=PATH env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" bash Scripts/Launch.sh --rundir Dist/Release --options "--mode Mask --mask GaussSine --ctldir ../../Controller --ctlfile ssvFast2" --logdir /home/hwnam/maya-aml/maya_logs --tag $1_$i -a $1
done