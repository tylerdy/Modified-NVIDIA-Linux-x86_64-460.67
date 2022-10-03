/*
 * fastHog.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */
/*
 * Modified to make fastHOG a periodic task to use as a
 * benchmark/stress test for GPU locking using Cuda call-wrapping
 * functions.  The program's performance is dominated by 
 * the computation on the execution engine (EE) while memory copies 
 * between Host and Device using the copy engine (CE) are significantly
 * less time consuming.
 *
 * This version uses a user allocated stream and asynchronous memory
 * copy operations (cudaMemcpyAsync()).  Cuda kernel invocations on the
 * stream are also asynchronous.  cudaStreamSynchronize() is used to 
 * synchronize with both the copy and kernel executions.  Host pinned
 * memory is used.
 *
 * The program depends on an input file containing the image 
 * representation (see file_name[] declaration below). 
 *
 * Modified by Don Smith, Department of Computer Science,
 * University of North Carolina at Chapel Hill
 * 2016
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>

#include "HOGImage.h"

#include "Others/persondetectorwt.tcc"

extern void InitializeHOG(int iw, int ih, float svmBias, float* svmWeights, int svmWeightsCount);
extern void BeginProcess(HOGImage* hostImage,
			 int _minx, int _miny, int _maxx, int _maxy, float minScale, float maxScale);
extern void EndProcess();
extern void GetImage(HOGImage *imageCUDA, ImageType imageType);
extern void FinalizeHOG();
extern void *HOGImageCUDA(int width, int height);
extern void *HOGImageFile(char* fileName);

HOGImage* image;
HOGImage* imageCUDA;

char file_name[] = "Files/Images/testImage.bmp";
pid_t my_pid;

int sync_level = 0; //default -- spin synchronization
int time_length = 1200; //default 20 minutes/1200 seconds

void Usage(char *s)
{
  fprintf(stderr,"\nUsage:%s \n", s);
  fprintf (stderr,"      [-s sync_level] (0-spin, 1-yield, 2-block)\n");
  fprintf (stderr,"      [-t time_length] (stop after this many seconds)\n");
  fprintf (stderr,"\n");
  exit(-1);
}

int main(int argc, char **argv)
{

  int i;

  my_pid = getpid();

  /* Parse the command line */

  i = 1;
  while (i < argc) {
    if (!strcmp (argv[i], "-d")) {
    //debug = 1;
    }
    else if (!strcmp (argv[i], "-s")) {
      if (++i >= argc) Usage (argv[0]);
      sync_level = atoi(argv[i]);
      // level 0 - spin polling (busy waiting) for GPU to finish
      // level 1 - yield each time through the polling loop to let another thread run
      // level 2 - block process waiting for GPU to finish
    }
    else if (!strcmp (argv[i], "-t")) {
      if (++i >= argc) Usage (argv[0]);
      time_length = atoi(argv[i]);
    }
    else 
      Usage (argv[0]);
    i++;
  }

    image = (HOGImage *)HOGImageFile(file_name);
    if (image->isLoaded == false){
       fprintf(stderr, "Failed to Load Image File.\n");
       exit(EXIT_FAILURE);
    }

    imageCUDA = (HOGImage *)HOGImageCUDA(image->width,image->height);

    InitializeHOG(image->width, image->height,
			PERSON_LINEAR_BIAS, PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC_LENGTH);

    // Pin everything
    if(mlockall(MCL_CURRENT | MCL_FUTURE)) {
       fprintf(stderr, "Failed to lock code pages.\n");
       exit(EXIT_FAILURE);
    }
    
    BeginProcess(image, -1, -1, -1, -1, -1.0f, -1.0f);
    //EndProcess();
        
    //FinalizeHOG();
    cudaStreamSynchronize(0);
    printf("Finished\n");
    return 0;
}
