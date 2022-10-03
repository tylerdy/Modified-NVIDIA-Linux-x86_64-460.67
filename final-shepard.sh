#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda-11.1/bin:$PATH"

# RE RUN (BEFORE RUNNING NEED TO MANUALLY CHANGE INSTRUCTION WE USE HERE BASED ON A/B RESULTS!!!!!)
# D: --smk-hybrid-comp-warps 0 16 32. -r 1 -a 983040 -l -w (C EXP. DEP.)
#declare -a scens=("--smk-hybrid-comp-warps 0 -r 1 -a 983040 -l -w" "--smk-hybrid-comp-warps 16 -r 1 -a 983040 -l -w" "--smk-hybrid-comp-warps 32 -r 1 -a 983040 -l -w")

# declare -a scens=("--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 0 --mem-w 32 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 8 --mem-w 24 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 16 --mem-w 16 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 24 --mem-w 8 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 32 --mem-w 0 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 32 -a 983040 -l -w --mem-r 0 --mem-w 0 --comp-fp 0 --comp-int 32" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 8 --comp-int 24" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 16 --comp-int 16" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 24 --comp-int 8" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 32 --comp-int 0")

# 0/32
# declare -a scens=("--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 0 --mem-w 32 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 8 --mem-w 24 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 16 --mem-w 16 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 24 --mem-w 8 --comp-fp 0 --comp-int 0" "--smk-hybrid-comp-warps 0 -a 983040 -l -w --mem-r 32 --mem-w 0 --comp-fp 0 --comp-int 0")

# 32/0
# declare -a scens=("--smk-hybrid-comp-warps 32 -a 983040 -l -w --mem-r 0 --mem-w 0 --comp-fp 0 --comp-int 32" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 8 --comp-int 24" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 16 --comp-int 16" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 24 --comp-int 8" "--smk-hybrid-comp-warps 32 -a 983040 -l -w  --mem-r 0 --mem-w 0 --comp-fp 32 --comp-int 0")

# 16/16
declare -a scens=("--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 4 --mem-w 12 --comp-fp 4 --comp-int 12" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 4 --mem-w 12 --comp-fp 8 --comp-int 8" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 4 --mem-w 12 --comp-fp 12 --comp-int 4" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 8 --mem-w 8 --comp-fp 4 --comp-int 12" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 8 --mem-w 8 --comp-fp 8 --comp-int 8" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 8 --mem-w 8 --comp-fp 12 --comp-int 4" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 12 --mem-w 4 --comp-fp 4 --comp-int 12" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 12 --mem-w 4 --comp-fp 8 --comp-int 8" "--smk-hybrid-comp-warps 16 -a 983040 -l -w --mem-r 12 --mem-w 4 --comp-fp 12 --comp-int 4")

# FINAL DATA EXPERIMENTS (20 SAMPLES). Should order based on results most likely to change from draft. Should run as several different scripts chained together.
# A/B: -i 00 01 02 03 05 08 14 17. -c 32
#declare -a scens=("-i 00 -c 32" "-i 01 -c 32" "-i 02 -c 32" "-i 03 -c 32" "-i 05 -c 32" "-i 08 -c 32" "-i 14 -c 32" "-i 17 -c 32")
# C: -r 0 1 2 3. -c 32 -a 983040 -s 128 -l
#declare -a scens=("-r 0 -c 32 -a 983040 -s 128 -l" "-r 1 -c 32 -a 983040 -s 128 -l" "-r 2 -c 32 -a 983040 -s 128 -l" "-r 3 -c 32 -a 983040 -s 128 -l")
# D: --smk-hybrid-comp-warps 0 16 32. -r 2 -a 983040 -l -w (C EXP. DEP.)
# declare -a scens=("--smk-hybrid-comp-warps 0 -r 2 -a 983040 -l -w" "--smk-hybrid-comp-warps 16 -r 2 -a 983040 -l -w" "--smk-hybrid-comp-warps 32 -r 2 -a 983040 -l -w")
# E/warp*: -w {NO -w}. -m 8 -a 2097152 -s 128 -r 3 (E/rw EXP. DEP.)
#declare -a scens=("-m 8 -a 2097152 -s 128 -r 3 -w" "-m 8 -a 2097152 -s 128 -r 3")
# E/rw: -r 0 1 2 3. -m 8 -a 2097152 -s 128.
#declare -a scens=("-r 0 -m 8 -a 2097152 -s 128" "-r 1 -m 8 -a 2097152 -s 128" "-r 2 -m 8 -a 2097152 -s 128" "-r 3 -m 8 -a 2097152 -s 128")
# G/size: -a 02097152 04194304 08388608 12582912 16777216. -m 8 -s 128 -r 3 (E/rw EXP. DEP.)
#declare -a scens=("-a 02097152 -m 8 -s 128 -r 3" "-a 04194304 -m 8 -s 128 -r 3" "-a 08388608 -m 8 -s 128 -r 3" "-a 12582912 -m 8 -s 128 -r 3" "-a 16777216 -m 8 -s 128 -r 3")
# G/stride*: -s 008 016 032 064 128 256 512. -m 8 -a 02097152 -r 3 (E/rw EXP. DEP.)
#declare -a scens=("-s 008 -m 8 -a 02097152 -r 3" "-s 016 -m 8 -a 02097152 -r 3" "-s 032 -m 8 -a 02097152 -r 3" "-s 064 -m 8 -a 02097152 -r 3" "-s 128 -m 8 -a 02097152 -r 3" "-s 256 -m 8 -a 02097152 -r 3" "-s 512 -m 8 -a 02097152 -r 3")



declare -a progs=("AlexNet" "bilateralFilter" "boxFilter" "CifarNet" "convolutionFFT2D" "convolutionSeparable" "dct8x8" "dwtHaar1D" "dxtc" "fastHOG" "fastWalshTransform" "GRU" "histogram" "HSOpticalFlow" "imageDenoising" "LSTM" "matrixMul" "NV12toBGRandResize" "recursiveGaussian" "ResNet" "scalarProd" "SqueezeNet" "stereoDisparity" "vectorAdd")
declare -a execs=("./AN 1" "./bilateralFilter" "./boxFilter" "./CN 1" "./convolutionFFT2D" "./convolutionSeparable" "./dct8x8" "./dwtHaar1D" "./dxtc" "./fastHOG" "./fastWalshTransform" "./GRU 1" "./histogram" "./HSOpticalFlow" "./imageDenoising" "./LSTM 1" "./matrixMul" "./NV12toBGRandResize" "./recursiveGaussian" "./RN 1" "./scalarProd" "./SN 1" "./stereoDisparity" "./vectorAdd")
declare -a etimes=(25 12 8 8 16 8  8 8 8 8 10 8 12 12 8 8 10 8 8 16 8 10 8 10)
declare -a sleeps=(5  10 6 6 10 12 6 6 6 6 8  6 8  8  6 6 8  6 6 10 6 8  6 8)


samples="$1" 
start=1 #"$2"
res_file="/playpen/leochanj/bench/results/eviction-results-SH_$(date +"%H%M%S")"
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
    # if [ "$prog" == "bilateralFilter" ] 
    # then
    #   if [[ "$scen" == "--smk-hybrid-comp-warps 0 -r 1 -a 983040 -l -w" ]] || [[ "$scen" == "--smk-hybrid-comp-warps 16 -r 1 -a 983040 -l # -w" ]]
    #   then
    #     echo "Skipped $prog for $scen"
    #     continue
    #   fi
    # fi
 
    if [[ $scen == *"-c 32"* ]] || [[ $scen == *"--smk"* ]]
    then
      if [ "$prog" == "AlexNet" ] || [ "$prog" == "CifarNet" ] || [ "$prog" == "ResNet" ] 
      then
        echo "Skipped $prog for $scen"
        continue
      fi
    fi

    print_scen=${scen//[[:space:]]/}
    print_scen=${print_scen//-}
    scen_res_base="$res_file/$prog-$print_scen"
    mkdir "$scen_res_base"
    # mkdir "$scen_res_base-b"
    for ((rep=start; rep<=samples; rep++)); do
      echo "$prog. $print_scen. $rep."
      echo quit | nvidia-cuda-mps-control
      nvidia-cuda-mps-control -d
      etime=$((${etimes[$i]} + 1)) 
      sleeptime=$((${sleeps[$i]} + 4)) 
      (../../build/stress -t $etime $scen && echo "-----------------eviction done-----------------") & (sleep 8 && nvprof --print-gpu-trace --normalized-time-unit ms --concurrent-kernels on --csv --log-file $scen_res_base/trace$rep.csv ${execs[$i]})
      sleep $sleeptime
      #(../../build/stress -t $etime $scen -b && echo "-----------------eviction done-----------------") & (sleep 8 && nvprof --print-gpu-trace --normalized-time-unit ms --concurrent-kernels on --csv --log-file $scen_res_base-b/trace$rep.csv ${execs[$i]})
      #sleep $sleeptime
    done
  done
done