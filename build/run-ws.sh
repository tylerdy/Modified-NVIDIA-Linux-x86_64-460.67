#!/usr/bin/env bash

# bash profile_repeat.sh 5000 bank_part  bank_part_warp  bc_part  bc_part_warp  cache_part_warp  module_part  module_part_warp
# (first three may have gotten mixed up in results oops)

#export CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export PATH="/usr/local/cuda-11.1/bin:$PATH"

samples="$1" 
start=1
res_file="/playpen/leochanj/bench/results/ws-misses-warps"
mkdir $res_file

for ((w=1; w<=16; w=w*2)); do
    echo $w
    mkdir $res_file/$w
    for ((rep=start; rep<=samples; rep++)); do
        echo $rep
        sudo CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb CUDA_DEVICE_ORDER=PCI_BUS_ID PATH="/usr/local/cuda-11.1/bin:$PATH" nvprof --print-gpu-trace --concurrent-kernels on --csv --log-file $res_file/$w/$rep.csv ./ws $w
    done
    
done
for ((w=32; w<=64; w=w+8)); do
    echo $w
    mkdir $res_file/$w
    for ((rep=start; rep<=samples; rep++)); do
        echo $rep
        sudo CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb CUDA_DEVICE_ORDER=PCI_BUS_ID PATH="/usr/local/cuda-11.1/bin:$PATH" nvprof --print-gpu-trace --concurrent-kernels on --csv --log-file $res_file/$w/$rep.csv ./ws $w
    done
    
done












