#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda-11.1/bin:$PATH"

declare -a progs=("convolutionFFT2D")
declare -a execs=("./convolutionFFT2D")
declare -a etimes=(16)
declare -a sleeps=(10)

# ("AlexNet" "bilateralFilter" "boxFilter" "CifarNet" "convolutionFFT2D" "convolutionSeparable" "dct8x8" "dwtHaar1D" "dxtc" "fastWalshTransform" "GRU" "histogram" "HSOpticalFlow" "LSTM" "matrixMul" "NV12toBGRandResize" "ResNet" "scalarProd")
# ("./AN 1" "./bilateralFilter" "./boxFilter" "./CN 1" "./convolutionFFT2D" "./convolutionSeparable" "./dct8x8" "./dwtHaar1D" "./dxtc" "./fastWalshTransform" "./GRU 1" "./histogram" "./HSOpticalFlow" "./LSTM 1" "./matrixMul" "./NV12toBGRandResize" "./RN 1" "./scalarProd")
# (25 8 8 8 16 8 8 8 8 8 8 8 8 8 8 8 14 8)
# (5 4 6 6 10 12 6 6 6 6 6 6 6 6 6 6 8 6)

#declare -a progs=("convolutionFFT2D" "convolutionSeparable" "histogram" "NV12toBGRandResize" "ResNet" "SqueezeNet")
#declare -a execs=("./convolutionFFT2D" "./convolutionSeparable" "./histogram" "./NV12toBGRandResize" "./RN 1" "./SN 1")

#declare -a etimes=(16 8 8 8 18 10)
#declare -a sleeps=(8 9 6 8 9 9)

#declare -a concurrent=("on" "on" "on" "on" "on" "on") # on kernels 

declare -a scens=("-b 16 -s 8") # "-b 8 -s 4" "-b 4 -s 2" "-b 2 -s 1") #"-b 16 -s 16" "-b 16 -s 8" "-b 8 -s 8" "-b 8 -s 4" "-b 4 -s 2" "-b 2 -s 1" "-b 16 -s 16 -w" "-b 16 -s 8 -w")

samples="$1" 
start=1 #"$2"
res_file="/playpen/leochanj/bench/results/eviction-results_$(date +"%H%M%S")"
mkdir $res_file

cd "/playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/"

for i in "${!progs[@]}"
do
  prog=${progs[$i]}
  cd "../eviction-victims/$prog"
  for j in "${!scens[@]}"
  do
    scen=${scens[$j]}
    print_scen=${scen//[[:space:]]/}
    print_scen=${print_scen//-}
    scen_res_base="$res_file/$prog-$print_scen"
    mkdir "$scen_res_base-m"
    mkdir "$scen_res_base-c"
    for ((rep=start; rep<=samples; rep++)); do
      echo "$prog. $print_scen. $rep."
      echo quit | nvidia-cuda-mps-control
      nvidia-cuda-mps-control -d
      (../../build/stress -t ${etimes[$i]} $scen && echo "-----------------eviction done-----------------") & (sleep 8 && nvprof --print-gpu-trace --concurrent-kernels on --csv --log-file $scen_res_base-m/trace$rep.csv ${execs[$i]})
      sleep ${sleeps[$i]}
      (../../build/stress -t ${etimes[$i]} $scen -c && echo "-----------------eviction done-----------------") & (sleep 8 && nvprof --print-gpu-trace --concurrent-kernels on --csv --log-file $scen_res_base-c/trace$rep.csv ${execs[$i]})
      sleep ${sleeps[$i]}
    done
  done
done
