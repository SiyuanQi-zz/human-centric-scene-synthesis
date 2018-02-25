#!/bin/bash

pid_log_file="running_pids.txt"
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