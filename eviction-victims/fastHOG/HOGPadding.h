#ifndef __HOG_PADDING__
#define __HOG_PADDING__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "HOGDefines.h"

void InitPadding(int hPaddedWidth, int hPaddedHeight);
void ClosePadding();

void PadHostImage(uchar4* registeredImage, float4 *paddedRegisteredImage,
		  int minx, int miny, int maxx, int maxy);

#endif
