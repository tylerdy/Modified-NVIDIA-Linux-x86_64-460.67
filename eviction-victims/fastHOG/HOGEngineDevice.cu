#include "HOGEngineDevice.h"
#include "HOGUtils.h"
#include "HOGConvolution.h"
#include "HOGHistogram.h"
#include "HOGSVMSlider.h"
#include "HOGScale.h"
#include "HOGPadding.h"
#include "cutil.h"

int hWidth, hHeight;
int hWidthROI, hHeightROI;
int hPaddedWidth, hPaddedHeight;
int rPaddedWidth, rPaddedHeight;

int minX, minY, maxX, maxY;

int hNoHistogramBins, rNoHistogramBins;

int hPaddingSizeX, hPaddingSizeY;
int hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, hWindowSizeX, hWindowSizeY;
int hNoOfCellsX, hNoOfCellsY, hNoOfBlocksX, hNoOfBlocksY;
int rNoOfCellsX, rNoOfCellsY, rNoOfBlocksX, rNoOfBlocksY;

int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;
int hNumberOfWindowsX, hNumberOfWindowsY;
int rNumberOfWindowsX, rNumberOfWindowsY;

float4 *paddedRegisteredImage;

float1 *resizedPaddedImageF1;
float4 *resizedPaddedImageF4;

float2 *colorGradientsF2;

float1 *blockHistograms;
float1 *cellHistograms;

float1 *svmScores;

bool hUseGrayscale;

uchar1* outputTest1;
uchar4* outputTest4;

float* hResult;

float scaleRatio;
float startScale;
float endScale;
int scaleCount;

int avSizeX, avSizeY, marginX, marginY;

extern uchar4* paddedRegisteredImageU4;
extern pid_t my_pid;
extern int sync_level;
extern int period;
extern int time_length;


//cudaStream_t my_stream;

void InitHOG(int width, int height, int _avSizeX, int _avSizeY,
	     int _marginX, int _marginY, int cellSizeX, int cellSizeY,
	     int blockSizeX, int blockSizeY, int windowSizeX, int windowSizeY,
	     int noOfHistogramBins, float wtscale, float svmBias, float* svmWeights, 
             int svmWeightsCount, bool useGrayscale)
{
    switch (sync_level)
      {
       case 0:
          cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        //   printf("PID %d started > Synch Level is Spin\n", my_pid);
          break;
       case 1:
          cudaSetDeviceFlags(cudaDeviceScheduleYield);
        //   printf("PID %d started > Synch Level is Yield\n", my_pid);
          break;
       case 2:
          cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        //   printf("PID %d started > Synch Level is Block\n", my_pid);
      }

        // follow convention and initialize CUDA/GPU
        // used here to invoke initialization of GPU locking
        cudaFree(0);

        // use device 0, the only one on a TK1
        cudaSetDevice(0);
  
        // create a user-defined stream
        //cudaStreamCreate(&my_stream);

	int i;
	int toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0;

	hWidth = width; hHeight = height;
	avSizeX = _avSizeX; avSizeY = _avSizeY; marginX = _marginX; marginY = _marginY;

	if (avSizeX) { toaddxx = hWidth * marginX / avSizeX; toaddxy = hHeight * marginY / avSizeX; }
	if (avSizeY) { toaddyx = hWidth * marginX / avSizeY; toaddyy = hHeight * marginY / avSizeY; }

	hPaddingSizeX = max(toaddxx, toaddyx); hPaddingSizeY = max(toaddxy, toaddyy);

	hPaddedWidth = hWidth + hPaddingSizeX*2;
	hPaddedHeight = hHeight + hPaddingSizeY*2;

	hUseGrayscale = useGrayscale;

	hNoHistogramBins = noOfHistogramBins;
	hCellSizeX = cellSizeX; hCellSizeY = cellSizeY; hBlockSizeX = blockSizeX; hBlockSizeY = blockSizeY;
	hWindowSizeX = windowSizeX; hWindowSizeY = windowSizeY;

	hNoOfCellsX = hPaddedWidth / cellSizeX;
	hNoOfCellsY = hPaddedHeight / cellSizeY;

	hNoOfBlocksX = hNoOfCellsX - blockSizeX + 1;
	hNoOfBlocksY = hNoOfCellsY - blockSizeY + 1;

	hNumberOfBlockPerWindowX = (windowSizeX - cellSizeX * blockSizeX) / cellSizeX + 1;
	hNumberOfBlockPerWindowY = (windowSizeY - cellSizeY * blockSizeY) / cellSizeY + 1;

	hNumberOfWindowsX = 0;
	for (i=0; i<hNumberOfBlockPerWindowX; i++) hNumberOfWindowsX += (hNoOfBlocksX-i)/hNumberOfBlockPerWindowX;

	hNumberOfWindowsY = 0;
	for (i=0; i<hNumberOfBlockPerWindowY; i++) hNumberOfWindowsY += (hNoOfBlocksY-i)/hNumberOfBlockPerWindowY;

	scaleRatio = 1.05f;
	startScale = 1.0f;
	endScale = min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight / (float) hWindowSizeY);
	scaleCount = (int)floor(logf(endScale/startScale)/logf(scaleRatio)) + 1;

	cutilSafeCall(cudaMalloc((void**) &paddedRegisteredImage, sizeof(float4) * hPaddedWidth * hPaddedHeight));

	if (useGrayscale)
		cutilSafeCall(cudaMalloc((void**) &resizedPaddedImageF1, sizeof(float1) * hPaddedWidth * hPaddedHeight));
	else
		cutilSafeCall(cudaMalloc((void**) &resizedPaddedImageF4, sizeof(float4) * hPaddedWidth * hPaddedHeight));

	cutilSafeCall(cudaMalloc((void**) &colorGradientsF2, sizeof(float2) * hPaddedWidth * hPaddedHeight));
	cutilSafeCall(cudaMalloc((void**) &blockHistograms, sizeof(float1) * hNoOfBlocksX * hNoOfBlocksY * cellSizeX * cellSizeY * hNoHistogramBins));
	cutilSafeCall(cudaMalloc((void**) &cellHistograms, sizeof(float1) * hNoOfCellsX * hNoOfCellsY * hNoHistogramBins));

	cutilSafeCall(cudaMalloc((void**) &svmScores, sizeof(float1) * hNumberOfWindowsX * hNumberOfWindowsY * scaleCount));

	InitConvolution(hPaddedWidth, hPaddedHeight, useGrayscale);
	InitHistograms(cellSizeX, cellSizeY, blockSizeX, blockSizeY, noOfHistogramBins, wtscale);
	InitSVM(svmBias, svmWeights, svmWeightsCount);
	InitScale(hPaddedWidth, hPaddedHeight);
	InitPadding(hPaddedWidth, hPaddedHeight);

	rPaddedWidth = hPaddedWidth;
	rPaddedHeight = hPaddedHeight;

	if (useGrayscale)
		cutilSafeCall(cudaMalloc((void**) &outputTest1, sizeof(uchar1) * hPaddedWidth * hPaddedHeight));
	else
		cutilSafeCall(cudaMalloc((void**) &outputTest4, sizeof(uchar4) * hPaddedWidth * hPaddedHeight));

	cutilSafeCall(cudaMallocHost((void**)&hResult, sizeof(float) * hNumberOfWindowsX * hNumberOfWindowsY * scaleCount));
}

void CloseHOG()
{
	/*cutilSafeCall(cudaFree(paddedRegisteredImage));

	if (hUseGrayscale)
		cutilSafeCall(cudaFree(resizedPaddedImageF1));
	else
		cutilSafeCall(cudaFree(resizedPaddedImageF4));

	cutilSafeCall(cudaFree(colorGradientsF2));
	cutilSafeCall(cudaFree(blockHistograms));
	cutilSafeCall(cudaFree(cellHistograms));

	cutilSafeCall(cudaFree(svmScores));*/

	CloseConvolution();
	CloseHistogram();
	CloseSVM();
	CloseScale();
	ClosePadding();

	if (hUseGrayscale)
		cutilSafeCall(cudaFree(outputTest1));
	else
		cutilSafeCall(cudaFree(outputTest4));

	cutilSafeCall(cudaFreeHost(hResult));

	cudaThreadExit();
}

void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, float minScale, float maxScale)
{
	int i;
	minX = minx; minY = miny; maxX = maxx; maxY = maxy;
	PadHostImage((uchar4*)hostImage, paddedRegisteredImage, minX, minY, maxX, maxY);

	rPaddedWidth = hPaddedWidth; rPaddedHeight = hPaddedHeight;
	scaleRatio = 1.05f;
	startScale = (minScale < 0.0f) ? 1.0f : minScale;
	endScale = (maxScale < 0.0f) ? min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight / (float) hWindowSizeY) : maxScale;
	scaleCount = (int)floor(logf(endScale/startScale)/logf(scaleRatio)) + 1;

	float currentScale = startScale;

	ResetSVMScores(svmScores);

	for (i=0; i<scaleCount; i++)
	{
		DownscaleImage(0, scaleCount, i, currentScale, hUseGrayscale, paddedRegisteredImage, resizedPaddedImageF1, resizedPaddedImageF4);

		SetConvolutionSize(rPaddedWidth, rPaddedHeight);

		if(hUseGrayscale) ComputeColorGradients1to2(resizedPaddedImageF1, colorGradientsF2);
		else ComputeColorGradients4to2(resizedPaddedImageF4, colorGradientsF2);

		ComputeBlockHistogramsWithGauss(colorGradientsF2, blockHistograms, hNoHistogramBins,
			hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, hWindowSizeX, hWindowSizeY,  rPaddedWidth, rPaddedHeight);

		NormalizeBlockHistograms(blockHistograms, hNoHistogramBins, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, rPaddedWidth, rPaddedHeight);

		LinearSVMEvaluation(svmScores, blockHistograms, hNoHistogramBins, hWindowSizeX, hWindowSizeY, hCellSizeX, hCellSizeY,
			hBlockSizeX, hBlockSizeY, rNoOfBlocksX, rNoOfBlocksY, i, rPaddedWidth, rPaddedHeight);

		currentScale *= scaleRatio;
	}
}

float* EndHOGProcessing()
{
	cudaThreadSynchronize();
	cutilSafeCall(cudaMemcpyAsync(hResult, svmScores, sizeof(float) * scaleCount * hNumberOfWindowsX * hNumberOfWindowsY, cudaMemcpyDeviceToHost));
	return hResult;
}


void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
		      int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
		      int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
		      int *cNumberOfWindowsX, int *cNumberOfWindowsY,
		      int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY)
{
	*cStartScale = startScale;
	*cEndScale = endScale;
	*cScaleRatio = scaleRatio;
	*cScaleCount = scaleCount;
	*cPaddingSizeX = hPaddingSizeX;
	*cPaddingSizeY = hPaddingSizeY;
	*cPaddedWidth = hPaddedWidth;
	*cPaddedHeight = hPaddedHeight;
	*cNoOfCellsX = hNoOfCellsX;
	*cNoOfCellsY = hNoOfCellsY;
	*cNoOfBlocksX = hNoOfBlocksX;
	*cNoOfBlocksY = hNoOfBlocksY;
	*cNumberOfWindowsX = hNumberOfWindowsX;
	*cNumberOfWindowsY = hNumberOfWindowsY;
	*cNumberOfBlockPerWindowX = hNumberOfBlockPerWindowX;
	*cNumberOfBlockPerWindowY = hNumberOfBlockPerWindowY;
}

cudaArray *imageArray2 = 0;
texture<float4, 2, cudaReadModeElementType> tex2;
cudaChannelFormatDesc channelDescDownscale2;

__global__ void resizeFastBicubic3(float4 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale)
{
	int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	int i = __umul24(y, width) + x;

	float u = x*scale;
	float v = y*scale;

	if (x < width && y < height)
	{
		float4 cF;

		if (scale == 1.0f)
			cF = paddedRegisteredImage[x + y * width];
		else
			cF = tex2D(tex2, u, v);

		outputFloat[i] = cF;
	}
}

void DownscaleImage2(float scale, float4* paddedRegisteredImage,
		     float4* resizedPaddedImageF4, int width, int height,
		     int &rPaddedWidth, int &rPaddedHeight)
{
	dim3 hThreadSize, hBlockSize;

	hThreadSize = dim3(THREAD_SIZE_W, THREAD_SIZE_H);

	rPaddedWidth = iDivUpF(width, scale);
	rPaddedHeight = iDivUpF(height, scale);

	hBlockSize = dim3(iDivUp(rPaddedWidth, hThreadSize.x), iDivUp(rPaddedHeight, hThreadSize.y));

	cutilSafeCall(cudaMemcpyToArrayAsync(imageArray2, 0, 0, paddedRegisteredImage, sizeof(float4) * width * height, cudaMemcpyDeviceToDevice));

	cutilSafeCall(cudaBindTextureToArray(tex2, imageArray2, channelDescDownscale2));

	cutilSafeCall(cudaMemsetAsync(resizedPaddedImageF4, 0, width * height * sizeof(float4)));

	resizeFastBicubic3<<<hBlockSize, hThreadSize>>>((float4*)resizedPaddedImageF4, (float4*)paddedRegisteredImage, rPaddedWidth, rPaddedHeight, scale);

	cutilSafeCall(cudaUnbindTexture(tex2));
}
