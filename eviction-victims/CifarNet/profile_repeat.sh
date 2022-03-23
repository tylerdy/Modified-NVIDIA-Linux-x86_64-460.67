#!/usr/bin/env bash

# bash profile_repeat.sh 1000 ./fastHOG/fastHOG ./matrixMul/matrixMul ./scalarProd/scalarProd ./vectorAdd/vectorAdd ./CUDA_Imaging_Samples/convolutionFFT2D/convolutionFFT2D ./CUDA_Imaging_Samples/convolutionSeparable/convolutionSeparable ./CUDA_Imaging_Samples/convolutionTexture/convolutionTexture ./CUDA_Imaging_Samples/dct8x8/dct8x8 ./CUDA_Imaging_Samples/dxtc/dxtc ./CUDA_Imaging_Samples/EGLStream_CUDA_CrossGPU/EGLStream_CUDA_CrossGPU ./CUDA_Imaging_Samples/histogram/histogram ./CUDA_Imaging_Samples/HSOpticalFlow/HSOpticalFlow ./CUDA_Imaging_Samples/NV12toBGRandResize/NV12toBGRandResize ./CUDA_Imaging_Samples/stereoDisparity/stereoDisparity
# ^ all but YOLO (15 applications)

export CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda-11.1/bin:$PATH"

samples="$1" 
shift
apps=("$@")
res_file="/playpen/leochanj/bench/results/results_$(date +"%H%M%S")"
mkdir $res_file

app_res_dir="$res_file/CifarNet"
mkdir $app_res_dir

for ((rep=1; rep<=samples; rep++)); do
  echo $rep

  $app_res_dir/../../../../fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/stress
  nvprof --print-gpu-trace --concurrent-kernels off --csv --log-file $app_res_dir/trace$rep.csv ./CN 1
done



