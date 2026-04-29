#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 5000 9000 10000 15000 20000 25000 35000 40000 
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak --model_name GaussianImage_RS --num_points $num_points --iterations 50000 --save_imgs && \

CUDA_VISIBLE_DEVICES=0 python train_quantize.py -d $data_path \
--data_name kodak --model_name GaussianImage_RS --num_points $num_points --iterations 10000 --save_imgs \
--model_path ./checkpoints/kodak/GaussianImage_RS_50000_$num_points
done

# for num_points in 20000
# do
# CUDA_VISIBLE_DEVICES=0 python train_quantize.py -d $data_path \
# --data_name kodak --model_name Cluster --num_points $num_points --iterations 11002 --save_imgs \
# --model_path ./checkpoints/kodak/Cluster_50000_$num_points
# done