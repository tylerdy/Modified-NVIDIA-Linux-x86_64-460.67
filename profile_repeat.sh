#!/usr/bin/env bash

# bash profile_repeat.sh 5000 bank_part  bank_part_warp  bc_part  bc_part_warp  cache_part_warp  module_part  module_part_warp
# (first three may have gotten mixed up in results oops)

export CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda-11.1/bin:$PATH"

samples="$1" 
start=1 #"$2"
shift #2
apps=("$@")
res_file="/playpen/leochanj/bench/results/results_$(date +"%H%M%S")"
mkdir $res_file
i=1
for part in "${apps[@]}";
  do
    app_res_dir="$res_file/$part"
    mkdir $app_res_dir
    i=$((i + 1))

    #adjust becuase some programs use relative paths when referring to files
    #name=${prog##*/}
    #location=${prog%/*}
    cd build
    # prog_to_exec="./$name"

    for ((rep=start; rep<=samples; rep++)); do
      echo $rep

      # $app_res_dir/../../../../fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/stress
      nvprof --print-gpu-trace --concurrent-kernels off --csv --log-file $app_res_dir/trace$rep.csv ./synth < /playpen/leochanj/partitions/$part
    done

    #adjust back 
    cd /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67
  done




