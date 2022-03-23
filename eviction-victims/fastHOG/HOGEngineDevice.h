#ifndef __CUDA_HOG__
#define __CUDA_HOG__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "HOGDefines.h"

extern void InitHOG(int width, int height, int avSizeX, int avSizeY, int marginX, int marginY,
		    int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
		    int windowSizeX, int windowSizeY, int noOfHistogramBins, float wtscale,
		    float svmBias, float* svmWeights, int svmWeightsCount, bool useGrayscale);

extern void CloseHOG();

extern void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, 
                               float minScale, float maxScale);
extern float* EndHOGProcessing();

extern void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
			     int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
			     int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
			     int *cNumberOfWindowsX, int *cNumberOfWindowsY,
			     int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY);

extern void GetProcessedImage(unsigned char* hostImage, int imageType);

void InitCUDAHOG(int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
		 int windowSizeX, int windowSizeY, int noOfHistogramBins, float wtscale,
		 float svmBias, float* svmWeights, int svmWeightsCount, bool useGrayscale);
void CloseCUDAHOG();

#endif
