#!/bin/bash

# Settings
startid=0
use_cpu_number=1
binary="./cpp/build/cpp"
pid_log_file="running_pids.txt"
endid=startid+use_cpu_number

# Make if updated
cpucores=$(($(cat /proc/cpuinfo | grep processor | wc -l) - 2))
echo $cpucores
mkdir -p cpp/build
cd cpp/build
cmake ..
make -j $cpucores
cd ../..

# Kill running processes and create log files
mkdir -p logs
if [ ! -f $pid_log_file ]; then
    echo "No already runnning programs."
else
    echo "Killing already runnning programs..."
	# Stop all the running programs
	while IFS= read line
	do
		if ps -p $line -o comm= | grep "cpp"; then
			kill -9 $line
		fi
	done <"$pid_log_file"
	rm -f $pid_log_file
fi

# Run programs
echo "Starting programs..."
for ((i=$startid;i<$endid;i+=1))
do
    $binary $i > "logs/${i}_output.txt" &
    echo $! >> $pid_log_file
done
# wait
