mkdir -p aml_logs
./aml_experiment.sh canneal $1 $2; 
./aml_experiment.sh freqmine $1 $2; 
./aml_experiment.sh vips $1 $2; 
./aml_experiment.sh streamcluster $1 $2;
./aml_experiment.sh blackscholes $1 $2;
./aml_experiment.sh bodytrack $1 $2;
./aml_experiment.sh splash2x.radiosity $1 $2;
./aml_experiment.sh splash2x.volrend $1 $2;
./aml_experiment.sh splash2x.water_nsquared $1 $2;
./aml_experiment.sh splash2x.water_spatial $1 $2;