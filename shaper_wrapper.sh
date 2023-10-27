mkdir -p shaper_logs
bash setuserspace.sh
./shaper_experiment.sh canneal $1 $2; 
./shaper_experiment.sh freqmine $1 $2; 
./shaper_experiment.sh vips $1 $2; 
./shaper_experiment.sh streamcluster $1 $2;
./shaper_experiment.sh blackscholes $1 $2;
./shaper_experiment.sh bodytrack $1 $2;
./shaper_experiment.sh splash2x.radiosity $1 $2;
./shaper_experiment.sh splash2x.volrend $1 $2;
./shaper_experiment.sh splash2x.water_nsquared $1 $2;
./shaper_experiment.sh splash2x.water_spatial $1 $2;