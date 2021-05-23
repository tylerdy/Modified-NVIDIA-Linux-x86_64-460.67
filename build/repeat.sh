#!/usr/bin/env bash
echo quit | nvidia-cuda-mps-control
for i in $(eval echo {1..$1});
do
nvidia-cuda-mps-control -d
./stress -t 5 & (sleep 1 && ./victim)
echo quit | nvidia-cuda-mps-control
done
