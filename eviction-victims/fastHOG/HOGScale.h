#ifndef __HOG_SCALE__
#define __HOG_SCALE__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "HOGDefines.h"

void InitScale(int hPaddedWidth, int hPaddedHeight);
void CloseScale();

void DownscaleImage(int startScaleId, int endScaleId, int scaleId, float scale, 
		    bool useGrayscale, float4* paddedRegisteredImage,
		    float1* resizedPaddedImageF1, float4* resizedPaddedImageF4);

__global__ void resizeFastBicubic1(float1 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale);
__global__ void resizeFastBicubic4(float4 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale);


#endif
