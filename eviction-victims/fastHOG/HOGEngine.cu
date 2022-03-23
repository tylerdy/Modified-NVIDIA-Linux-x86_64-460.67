#include "HOGEngine.h"
#include "HOGDefines.h"
#include "HOGUtils.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

extern void InitHOG(int width, int height, int avSizeX, int avSizeY,
		    int marginX, int marginY, int cellSizeX, int cellSizeY,
		    int blockSizeX, int blockSizeY, int windowSizeX, int windowSizeY,
		    int noOfHistogramBins, float wtscale, float svmBias, float* svmWeights,
		    int svmWeightsCount, bool useGrayscale);

extern void CloseHOG();
extern void ComputeFormattedResults();
extern void SaveResultsToDisk(char* fileName);
extern void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, float minScale, float maxScale);
extern float* EndHOGProcessing();

extern void GetProcessedImage(unsigned char* hostImage, int imageType);
extern void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
			     int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
			     int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
			     int *cNumberOfWindowsX, int *cNumberOfWindowsY,
			     int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY);
extern pid_t my_pid;
extern int sync_level;
extern int period;
extern int time_length;

#define MAX_DELAY      750  // mean of 500 microseconds delay
#define MIN_DELAY      250

//select random delay of microseconds between
//successive kernel launches in a stream sampling
//from a uniform distribution
int get_delay(struct random_data *buf)
{
 int32_t random;
 random_r(buf, &random);
 return (MIN_DELAY + random%(MAX_DELAY - MIN_DELAY + 1));
}
   

double elapsed_ms(struct timespec *te, struct timespec *ts)
{
  double elapsed;

  elapsed = (((double)te->tv_sec)*1e9 + (double)te->tv_nsec) -
            (((double)ts->tv_sec)*1e9 + (double)ts->tv_nsec);
  elapsed /= (double)1e6; // convert to ms
  return(elapsed);
}

void InitializeHOG(int iw, int ih, float svmBias, float* svmWeights, int svmWeightsCount)
{
	HOG.imageWidth = iw;
	HOG.imageHeight = ih;

	HOG.avSizeX = 48; //48
	HOG.avSizeY = 96; //96
	HOG.marginX = 4; // 4
	HOG.marginY = 4; // 4

	HOG.hCellSizeX = 8;
	HOG.hCellSizeY = 8;
	HOG.hBlockSizeX = 2;
	HOG.hBlockSizeY = 2;
	HOG.hWindowSizeX = 64;
	HOG.hWindowSizeY = 128;
	HOG.hNoOfHistogramBins = 9;

	HOG.svmWeightsCount = svmWeightsCount;
	HOG.svmBias = svmBias;
	HOG.svmWeights = svmWeights;

	HOG.wtScale = 2.0f;

	HOG.useGrayscale = false;

	HOG.formattedResultsAvailable = false;

	InitHOG(iw, ih, HOG.avSizeX, HOG.avSizeY, HOG.marginX, HOG.marginY, HOG.hCellSizeX, HOG.hCellSizeY, HOG.hBlockSizeX, HOG.hBlockSizeY,
		HOG.hWindowSizeX, HOG.hWindowSizeY, HOG.hNoOfHistogramBins, HOG.wtScale, HOG.svmBias, HOG.svmWeights, HOG.svmWeightsCount, HOG.useGrayscale);
}



void FinalizeHOG()
{
	CloseHOG();
}

void BeginProcess(HOGImage* hostImage,
		int _minx, int _miny, int _maxx, int _maxy, float minScale, float maxScale)
{
    int i;
    int count = 0;

    struct timespec task_start, job_start; 
    struct timespec delay_t;
    long delay_us;
	
    time_t start_time, now, elapsed;

    // struct random_data buf;
    // char r_state[64];

    // initstate_r((unsigned int)my_pid, r_state, sizeof(r_state), &buf);


	HOG.minX = _minx, HOG.minY = _miny, HOG.maxX = _maxx, HOG.maxY = _maxy;

	if (HOG.minY == -1 && HOG.minY == -1 && HOG.maxX == -1 && HOG.maxY == -1)
	{
		HOG.minX = 0;
		HOG.minY = 0;
		HOG.maxX = HOG.imageWidth;
		HOG.maxY = HOG.imageHeight;
	}

    // all setup and initialization complete, start image processing
    // printf("PID %d Iterating CUDA fastHOG for %d seconds\n", my_pid, time_length);

    now = start_time = time(NULL);
    clock_gettime(CLOCK_MONOTONIC_RAW, &task_start);
    
//  for (i = 0; (now - time_length) < start_time; i++) {

    clock_gettime(CLOCK_MONOTONIC_RAW, &job_start);
   
    BeginHOGProcessing(hostImage->pixels, HOG.minX, HOG.minY, HOG.maxX, HOG.maxY, minScale, maxScale);

    HOG.cppResult = EndHOGProcessing();

    GetHOGParameters(&HOG.startScale, &HOG.endScale, &HOG.scaleRatio, &HOG.scaleCount,
		&HOG.hPaddingSizeX, &HOG.hPaddingSizeY, &HOG.hPaddedWidth, &HOG.hPaddedHeight,
		&HOG.hNoOfCellsX, &HOG.hNoOfCellsY, &HOG.hNoOfBlocksX, &HOG.hNoOfBlocksY, &HOG.hNumberOfWindowsX,
		&HOG.hNumberOfWindowsY, &HOG.hNumberOfBlockPerWindowX, &HOG.hNumberOfBlockPerWindowY);

    ComputeFormattedResults();
    
    // delay_us = get_delay(&buf); //randomly select a loop delay
    // delay_t.tv_sec = 0;
    // delay_t.tv_nsec = delay_us * 1000;
    // nanosleep(&delay_t, NULL);
		     
    now = time(NULL);
//   } //ends for loop 
  elapsed = now - start_time;
  count = i;
//   printf("PID %d completed %d, duration %ld seconds\n", my_pid, count, elapsed);
}

void EndProcess()
{
  char file_name[] = "formattedResults.txt";

  SaveResultsToDisk(file_name);
}


void SaveResultsToDisk(char* fileName)
{
	FILE* f; 
	f = fopen(fileName, "w+");
	fprintf(f, "%d\n", HOG.formattedResultsCount);
	for (int i=0; i<HOG.formattedResultsCount; i++)
	{
		fprintf(f, "%f %f %d %d %d %d %d %d\n",
			HOG.formattedResults[i].scale, HOG.formattedResults[i].score,
			HOG.formattedResults[i].width, HOG.formattedResults[i].height,
			HOG.formattedResults[i].x, HOG.formattedResults[i].y,
			HOG.formattedResults[i].origX, HOG.formattedResults[i].origY);
	}
	fclose(f);
}

void ComputeFormattedResults()
{
	int i, j, k, resultId;
	int leftoverX, leftoverY, currentWidth, currentHeight, rNumberOfWindowsX, rNumberOfWindowsY;

	resultId = 0;
	HOG.formattedResultsCount = 0;

	float* currentScaleWOffset;
	float currentScale = HOG.startScale;

	for (i=0; i<HOG.scaleCount; i++)
	{
		currentScaleWOffset = HOG.cppResult + i * HOG.hNumberOfWindowsX * HOG.hNumberOfWindowsY;

		for (j = 0; j < HOG.hNumberOfWindowsY; j++)
		{
			for (k = 0; k < HOG.hNumberOfWindowsX; k++)
			{
				float score = currentScaleWOffset[k + j * HOG.hNumberOfWindowsX];
				if (score > 0)
					HOG.formattedResultsCount++;
			}
		}
	}

	for (i=0; (i<HOG.scaleCount) && (resultId<MAX_RESULTS); i++)
	{
		currentScaleWOffset = HOG.cppResult + i * HOG.hNumberOfWindowsX * HOG.hNumberOfWindowsY;

		for (j=0; j<HOG.hNumberOfWindowsY; j++)
		{
			for (k=0; k<HOG.hNumberOfWindowsX; k++)
			{
				float score = currentScaleWOffset[k + j * HOG.hNumberOfWindowsX];
				if (score > 0)
				{
					currentWidth = iDivUpF(HOG.hPaddedWidth, currentScale);
					currentHeight = iDivUpF(HOG.hPaddedHeight, currentScale);

					rNumberOfWindowsX = (currentWidth - HOG.hWindowSizeX) / HOG.hCellSizeX + 1;
					rNumberOfWindowsY = (currentHeight - HOG.hWindowSizeY) / HOG.hCellSizeY + 1;

					leftoverX = (currentWidth - HOG.hWindowSizeX - HOG.hCellSizeX * (rNumberOfWindowsX - 1)) / 2;
					leftoverY = (currentHeight - HOG.hWindowSizeY - HOG.hCellSizeY * (rNumberOfWindowsY - 1)) / 2;

					HOG.formattedResults[resultId].origX = k * HOG.hCellSizeX + leftoverX;
					HOG.formattedResults[resultId].origY = j * HOG.hCellSizeY + leftoverY;

					HOG.formattedResults[resultId].width = (int)floorf((float)HOG.hWindowSizeX * currentScale);
					HOG.formattedResults[resultId].height = (int)floorf((float)HOG.hWindowSizeY * currentScale);

					HOG.formattedResults[resultId].x = (int)ceilf(currentScale * 
                                                                       (HOG.formattedResults[resultId].origX + HOG.hWindowSizeX / 2) - 
                                                                       (float) HOG.hWindowSizeX * currentScale / 2) - 
                                                                       HOG.hPaddingSizeX + HOG.minX;
					HOG.formattedResults[resultId].y = (int)ceilf(currentScale * 
                                                                       (HOG.formattedResults[resultId].origY + HOG.hWindowSizeY / 2) - 
                                                                       (float) HOG.hWindowSizeY * currentScale / 2) - 
                                                                       HOG.hPaddingSizeY + HOG.minY;

					HOG.formattedResults[resultId].scale = currentScale;
					HOG.formattedResults[resultId].score = score;

					resultId++;
				}
			}
		}

		currentScale = currentScale * HOG.scaleRatio;
	}
}
