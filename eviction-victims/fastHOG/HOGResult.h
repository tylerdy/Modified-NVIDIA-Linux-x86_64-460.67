#ifndef __HOG_RESUL__
#define __HOG_RESUL__


typedef struct hogresult {
  float score;
  float scale;
  int width;
  int height;
  int origX;
  int origY;
  int x;
  int y;
} HOGResult;

#endif

