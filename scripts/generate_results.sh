#!/bin/bash
# Assumes a file structure like:
# [root data dir]
#   [camera0]
#     test.det
#     training_arm.det 
#     validation.det
#     training_manual_[1-10].det
#   [camera1]
#     etc.
#
# We will iterate through the folders and store results in:
# [root results dir]
#	[camera0]
#     ur_validation_arm_n#_i#.log
#     ur_validation_manual_[1-10]_n#_i#.log
#     etc..

if [ $# -ne 2 ]; then
	echo "Please specify data_dir and results_dir"
	exit -1
fi

dataDir=${1%/}
resultsDir=${2%/}

budgets=`seq 10 10 50`
iters=`seq 1 10`
methods=( "ur" "ss" "gc" )

for d in $dataDir/*/
do
	name=$(basename "$d")
	
	echo "Processing data in $name..."
	if [ ! -d "${resultsDir}/${name}" ]; then
		mkdir "${resultsDir}/${name}"
	fi
	
	for method in ${methods[@]};
	do
		for n in $budgets;
		do
			for i in $iters;
			do
			
				trap "echo Exited!; exit;" SIGINT SIGTERM
				
				echo "Running ${method} for b=${n} iter=${i} validations"
				# Arm training validation
				~/Software/calotypes/bin/holdout_calibration \
							-t "${dataDir}/${name}/validation.det" \
							-d "${dataDir}/${name}/training_arm.det" \
							-o "${resultsDir}/${name}/${method}_validation_arm_n${n}_i${i}.log" \
							-n $n -m $method -b 0.05  > /dev/null &
				
				# Manual training 1-5 validation
				for s in `seq 1 4`; do
					~/Software/calotypes/bin/holdout_calibration \
							-t "${dataDir}/${name}/validation.det" \
							-d "${dataDir}/${name}/training_manual_${s}.det" \
							-o "${resultsDir}/${name}/${method}_validation_manual_${s}_n${n}_i${i}.log" \
							-n $n -m $method -b 0.05  > /dev/null &
				done
				~/Software/calotypes/bin/holdout_calibration \
							-t "${dataDir}/${name}/validation.det" \
							-d "${dataDir}/${name}/training_manual_5.det" \
							-o "${resultsDir}/${name}/${method}_validation_manual_5_n${n}_i${i}.log" \
							-n $n -m $method -b 0.05  > /dev/null
				
				# Manual training 5-10 validation
				for s in `seq 6 9`; do
					~/Software/calotypes/bin/holdout_calibration \
							-t "${dataDir}/${name}/validation.det" \
							-d "${dataDir}/${name}/training_manual_${s}.det" \
							-o "${resultsDir}/${name}/${method}_validation_manual_${s}_n${n}_i${i}.log" \
							-n $n -m $method -b 0.05  > /dev/null &
				done
				~/Software/calotypes/bin/holdout_calibration \
							-t "${dataDir}/${name}/validation.det" \
							-d "${dataDir}/${name}/training_manual_10.det" \
							-o "${resultsDir}/${name}/${method}_validation_manual_10_n${n}_i${i}.log" \
							-n $n -m $method -b 0.05  > /dev/null
				
			done
		done
	done
	
done

echo "All tests complete!"
date

# for n in `seq 10 10 50`;
# do
# 	
# 	for i in `seq 1 10`;
# 	do
# 		echo "Running trial $i for n=$n SS"
# 		log="$logDir/results_ss_n"$n"_i"$i".txt"
# 		~/Software/calotypes/bin/holdout_calibration -t $testdata -d $traindata -n $n -m ss -o $log -b 0.05
# 		echo "Running trial $i for n=$n UR"
# 		log="$logDir/results_ur_n"$n"_i"$i".txt"
# 		~/Software/calotypes/bin/holdout_calibration -t $testdata -d $traindata -n $n -m ur -o $log -b 0.05
# 		echo "Running trial $i for n=$n GC"
# 		log="$logDir/results_gc_n"$n"_i"$i".txt"
# 		~/Software/calotypes/bin/holdout_calibration -t $testdata -d $traindata -n $n -m gc -o $log -b 0.05
# 	done
# 	
# done
