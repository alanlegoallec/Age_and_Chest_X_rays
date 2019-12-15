#!/bin/bash
segments=( "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" )
images_sizes=( "224" "299" "331" )
images_sizes=( "331" )
memory=3G
n_cores=1
time=60
for i in "${segments[@]}"
do
for images_size in "${images_sizes[@]}"
do
job_name="S01-$i-$images_size.job"
out_file="../eo/S01-$i-$images_size.out"
err_file="../eo/S01-$i-$images_size.err"
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time CXR_S01_preprocessing_segments.sh $i $images_size
done
done


