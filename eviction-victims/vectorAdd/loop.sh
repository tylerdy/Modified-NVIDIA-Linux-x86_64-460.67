#!/bin/bash
count=1024
for i in $(seq $count); do
 sudo  CUDA_VISIBLE_DEVICES=GPU-5c7929ac-b24c-c43d-8a4e-06b1146defcb CUDA_DEVICE_ORDER=PCI_BUS_ID ./vectorAdd
 echo $i
done

