mkdir -p defender_logs
bash setuserspace.sh
./defender_experiment.sh canneal $1 $2; 
./defender_experiment.sh freqmine $1 $2; 
./defender_experiment.sh vips $1 $2; 
./defender_experiment.sh streamcluster $1 $2;
./defender_experiment.sh blackscholes $1 $2;
./defender_experiment.sh bodytrack $1 $2;
./defender_experiment.sh splash2x.radiosity $1 $2;
./defender_experiment.sh splash2x.volrend $1 $2;
./defender_experiment.sh splash2x.water_nsquared $1 $2;
./defender_experiment.sh splash2x.water_spatial $1 $2;