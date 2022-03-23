#ifndef __HOG_ENGINE__
#define __HOG_ENGINE__

#include "HOGResult.h"
#include "HOGImage.h"
#include "HOGDefines.h"

struct hog {
  int imageWidth, imageHeight;
  int avSizeX, avSizeY, marginX, marginY;
  int scaleCount;
  int hCellSizeX, hCellSizeY;
  int hBlockSizeX, hBlockSizeY;
  int hWindowSizeX, hWindowSizeY;
  int hNoOfHistogramBins;
  int hPaddedWidth, hPaddedHeight;
  int hPaddingSizeX, hPaddingSizeY;

  int minX, minY, maxX, maxY;

  float wtScale;

  float startScale, endScale, scaleRatio;

  int svmWeightsCount;
  float svmBias, *svmWeights;

  int hNoOfCellsX, hNoOfCellsY;
  int hNoOfBlocksX, hNoOfBlocksY;
  int hNumberOfWindowsX, hNumberOfWindowsY;
  int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;

  bool useGrayscale;

  float* cppResult;

  HOGResult formattedResults[MAX_RESULTS];

  bool formattedResultsAvailable;
  int formattedResultsCount;

  // void BeginProcess(HOGImage* hostImage, int _minx = -1, int _miny = -1, int _maxx = -1, int _maxy = -1,
  //		float minScale = -1.0f, float maxScale = -1.0f);
}HOG;

#endif
