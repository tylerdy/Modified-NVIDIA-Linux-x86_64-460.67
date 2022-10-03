#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda-11.1/bin:$PATH"

# bash naive-evaluation.sh 5 fastWalshTransform/fastWalshTransform fastWalshTransform; bash naive-evaluation.sh 5 "SqueezeNet/SN 1" SqueezeNet; bash naive-evaluation.sh 5 vectorAdd/vectorAdd vectorAdd

# declare -a progs=("SqueezeNet" "HSOpticalFlow" "convolutionFFT2D")
# declare -a execs=("./SN 1" "./HSOpticalFlow" "./convolutionFFT2D")
# declare -a etimes=(8 8 16)
# declare -a sleeps=(6 6 10)

# standard L1 and L2 experiments:
#declare -a scens=("-m 4 -a 2097152 -s 128" "-m 4 -a 4194304 -s 128" "-m 4 -a 8388608 -s 128" "-m 4 -a 10485760 -s 128" "-m 4 -a 12582912 -s 128" "-m 4 -a 14680064 -s 128" "-m 4 -a 16777216 -s 128" "-m 4 -a 2097152 -s 4" "-m 4 -a 2097152 -s 8" "-m 4 -a 2097152 -s 16" "-m 4 -a 2097152 -s 32" "-m 4 -a 2097152 -s 64" "-c 32 -a 1310720 -s 128 -l" "-c 32 -a 1966080 -s 128 -l" "-c 32 -a 2293760 -s 128 -l" "-c 32 -a 2621440 -s 128 -l" "-c 32 -a 2949120 -s 128 -l" "-c 32 -a 3276800 -s 128 -l" "-c 32 -a 3604480 -s 128 -l" "-c 32 -a 3932160 -s 128 -l" "-c 32 -a 4259840 -s 128 -l" "-c 32 -a 983040 -s 4 -l" "-c 32 -a 983040 -s 8 -l" "-c 32 -a 983040 -s 16 -l" "-c 32 -a 983040 -s 32 -l" "-c 32 -a 983040 -s 64 -l")
# nonstandard
#declare -a scens=("-m 4 -a 2097152 -s 32" "-m 4 -a 4194304 -s 32" "-m 4 -a 8388608 -s 32" "-m 4 -a 10485760 -s 32" "-m 4 -a 12582912 -s 32" "-m 4 -a 14680064 -s 32" "-m 4 -a 16777216 -s 32" "-m 4 -a 12582912 -s 4" "-m 4 -a 12582912 -s 8" "-m 4 -a 12582912 -s 16" "-m 4 -a 12582912 -s 32" "-m 4 -a 12582912 -s 64" "-c 32 -a 1966080 -s 8 -l" "-c 32 -a 2949120 -s 8 -l" "-c 32 -a 3932160 -s 8 -l" "-c 32 -a 3932160 -s 4 -l" "-c 32 -a 3932160 -s 16 -l" "-c 32 -a 3932160 -s 32 -l" "-c 32 -a 3932160 -s 64 -l")

#declare -a scens=("-c 32 -i 0" "-c 32 -i 1" "-c 32 -i 2" "-c 32 -i 3" "-c 32 -i 4" "-c 32 -i 5" "-c 32 -i 6" "-c 32 -i 7" "-c 32 -i 8" "-c 32 -i 9" "-c 32 -i 10" "-c 32 -i 11" "-c 32 -i 12" "-c 32 -i 13" "-c 32 -i 14" "-c 32 -i 15" "-c 32 -i 16" "-m 8 -r 0" "-m 8 -r 1" "-m 8 -r 2" "-m 8 -r 3" "-c 32 -a 983040 -l -r 0" "-c 32 -a 983040 -l -r 1" "-c 32 -a 983040 -l -r 2" "-c 32 -a 983040 -l -r 3" "--smk-hybrid-comp-warps 0" "--smk-hybrid-comp-warps 16" "--smk-hybrid-comp-warps 32");
# declare -a scens=("-c 32 -i 0" "-c 32 -i 1" "-c 32 -i 3" "-c 32 -i 5" "-c 32 -i 14" "-c 32 -i 17") # should do this later
#declare -a scens=("--smk-hybrid-comp-warps 16 -r 2" "--smk-hybrid-comp-warps 32 -r 2" "--smk-hybrid-comp-warps 16 -r 1" "--smk-hybrid-comp-warps 32 -r 1")
declare -a scens=("-m 8 -r 1 -a 12582912")

declare -a progs=("AlexNet" "bilateralFilter" "boxFilter" "CifarNet" "convolutionFFT2D" "convolutionSeparable" "dct8x8" "dwtHaar1D" "dxtc" "fastHOG" "fastWalshTransform" "GRU" "histogram" "HSOpticalFlow" "imageDenoising" "LSTM" "matrixMul" "NV12toBGRandResize" "recursiveGaussian" "ResNet" "scalarProd" "SqueezeNet" "stereoDisparity" "vectorAdd")
declare -a execs=("./AN 1" "./bilateralFilter" "./boxFilter" "./CN 1" "./convolutionFFT2D" "./convolutionSeparable" "./dct8x8" "./dwtHaar1D" "./dxtc" "./fastHOG" "./fastWalshTransform" "./GRU 1" "./histogram" "./HSOpticalFlow" "./imageDenoising" "./LSTM 1" "./matrixMul" "./NV12toBGRandResize" "./recursiveGaussian" "./RN 1" "./scalarProd" "./SN 1" "./stereoDisparity" "./vectorAdd")
declare -a etimes=(25 12 8 8 16 8  8 8 8 8 10 8 12 12 8 8 10 8 8 16 8 10 8 10)
declare -a sleeps=(5  10 6 6 10 12 6 6 6 6 8  6 8  8  6 6 8  6 6 10 6 8  6 8)



samples="$1" 
naive="$2"
naive_short="$3"
echo "Running $naive_short"
start=1 #"$2"
res_file="/playpen/leochanj/bench/results/naive_evaluation-results_$(date +"%H%M%S")"
mkdir $res_file

cd "/playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/"

for i in "${!progs[@]}"
do
  prog=${progs[$i]}
  cd "/playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/eviction-victims/$prog"
  for j in "${!scens[@]}"
  do
    scen=${scens[$j]}

    # if [ "$prog" == "AlexNet" ] || [ "$prog" == "CifarNet" ] || [ "$prog" == "matrixMul" ] || [ "$prog" == "NV12" ] || [ "$prog" == "ResNet" ] || [ # "$prog" == "SqueezeNet" ]
    # then
    #   if [ "$scen" == "-c 48" ]
    #   then
    #     echo "Skipped $prog for $scen"
    #     continue
    #   fi
    # fi
    
    
    if [[ $scen == -c* ]] || [[ $scen == --smk* ]]
    then
      if [ "$prog" == "AlexNet" ] || [ "$prog" == "CifarNet" ] || [ "$prog" == "ResNet" ] 
      then
        echo "Skipped $prog for $scen"
        continue
      fi
    fi

    print_scen=${scen//[[:space:]]/}
    print_scen=${print_scen//-}
    scen_res_base="$res_file/$prog-$naive_short&$print_scen"
    mkdir "$scen_res_base"
    # mkdir "$scen_res_base-b"
    for ((rep=start; rep<=samples; rep++)); do
      echo "$prog. $print_scen. $rep."
      echo quit | nvidia-cuda-mps-control
      nvidia-cuda-mps-control -d
      
      #sleep ${sleeps[$i]}
      #(../../build/stress -t ${etimes[$i]} $scen -b && echo "-----------------eviction done-----------------") & (sleep 8 && nvprof --print-gpu-trace --concurrent-kernels on --csv --log-file $scen_res_base-b/trace$rep.csv ${execs[$i]})
      etime=$((${etimes[$i]} + 1)) 
      sleeptime=$((${sleeps[$i]} + 4)) 
      (../../build/stress -t $etime $scen && echo "-----------------eviction done-----------------") & (sleep 8 && nvprof --print-gpu-trace --normalized-time-unit ms --concurrent-kernels on --csv --log-file $scen_res_base/trace$rep.csv ${execs[$i]})
      sleep $sleeptime

      #etime=$((${etimes[$i]} + 7)) 
      #sleeptime=$((${sleeps[$i]} + 10)) 
      #(../../naive-stressors/$naive -t $etime && echo "-----------------eviction done-----------------") & (sleep 8 && nvprof --print-gpu-trace --normalized-time-unit ms --concurrent-kernels on --csv --log-file $scen_res_base-b/trace$rep.csv ${execs[$i]})
      #sleep $sleeptime
    done
  done
done
