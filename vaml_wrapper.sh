mkdir -p traces/aml_video_60ms_2
./video_aml.sh parkrun $1 $2; 
./video_aml.sh riverbed $1 $2; 
./video_aml.sh sunflower $1 $2; 
./video_aml.sh tractor $1 $2;
./video_aml.sh wind $1 $2;