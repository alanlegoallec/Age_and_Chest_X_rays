#!/bin/bash
models=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "NASNetLarge" "Xception" "InceptionV3" "InceptionResNetV2" )
models=( "NASNetLarge" )
positions=( "PA" "AP" "BOTH" )
positions=( "BOTH" )
memory=50G
n_cores=1
time=3000
for model in "${models[@]}"
do
for position in "${positions[@]}"
do
job_name="S05-$model-$position.job"
out_file="../eo/S05-$model-$position.out"
err_file="../eo/S05-$model-$position.err"
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time CXR_S05_tuning.sh $model $position
done
done

