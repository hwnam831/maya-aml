numCores="$(nproc --all)"

for ((core = 0; core < numCores; core++)); do
  sudo cpufreq-set -c ${core} -g ondemand
done
