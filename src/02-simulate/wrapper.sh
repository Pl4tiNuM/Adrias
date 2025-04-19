#!/bin/bash

for i in "5-30","3600" "5-25","3600" "5-20","3600"
do 
	IFS="," 
	set -- $i
       	./01-run-simulation.sh $1 $2
done
