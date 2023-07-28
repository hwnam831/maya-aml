mkdir -p traces/raw_video
./video_raw.sh parkrun $1 $2; 
./video_raw.sh riverbed $1 $2; 
./video_raw.sh sunflower $1 $2; 
./video_raw.sh tractor $1 $2;
./video_raw.sh wind $1 $2;