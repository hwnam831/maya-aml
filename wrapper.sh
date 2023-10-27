mkdir -p logs
bash setondemand.sh
./experiment.sh canneal $1 $2; 
./experiment.sh freqmine $1 $2; 
./experiment.sh vips $1 $2; 
./experiment.sh streamcluster $1 $2;
./experiment.sh blackscholes $1 $2;
./experiment.sh bodytrack $1 $2;
./experiment.sh splash2x.radiosity $1 $2;
./experiment.sh splash2x.volrend $1 $2;
./experiment.sh splash2x.water_nsquared $1 $2;
./experiment.sh splash2x.water_spatial $1 $2;