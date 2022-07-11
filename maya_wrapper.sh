mkdir -p maya_logs
./maya_experiment.sh canneal $1 $2; 
./maya_experiment.sh freqmine $1 $2; 
./maya_experiment.sh vips $1 $2; 
./maya_experiment.sh streamcluster $1 $2;
./maya_experiment.sh blackscholes $1 $2;
./maya_experiment.sh bodytrack $1 $2;
./maya_experiment.sh splash2x.radiosity $1 $2;
./maya_experiment.sh splash2x.volrend $1 $2;
./maya_experiment.sh splash2x.water_nsquared $1 $2;
./maya_experiment.sh splash2x.water_spatial $1 $2;