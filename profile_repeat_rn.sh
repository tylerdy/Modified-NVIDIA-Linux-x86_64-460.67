#!/usr/bin/env bash

# bash profile_repeat.sh 5000 bank_part  bank_part_warp  bc_part  bc_part_warp  cache_part_warp  module_part  module_part_warp
# (first three may have gotten mixed up in results oops)

#export CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export PATH="/usr/local/cuda-11.1/bin:$PATH"

samples="$1" 
start=860
res_file="/playpen/leochanj/bench/results/results_$(date +"%H%M%S")"
mkdir $res_file

app_res_dir="$res_file/RN"
mkdir $app_res_dir

cd ResNet

for ((rep=start; rep<=samples; rep++)); do
  echo $rep
  sudo CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb CUDA_DEVICE_ORDER=PCI_BUS_ID PATH="/usr/local/cuda-11.1/bin:$PATH" nvprof --print-gpu-trace --concurrent-kernels off --csv --log-file $app_res_dir/trace$rep.csv ../build/RN 1
done






