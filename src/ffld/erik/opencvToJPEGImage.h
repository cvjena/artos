#ifndef OPENCV_TO_JPEGIMAGE_INCLUDE
#define OPENCV_TO_JPEGIMAGE_INCLUDE

#include "JPEGImage.h"

#include <opencv2/imgproc/imgproc.hpp>

FFLD::JPEGImage *opencvToJPEGImage ( const cv::Mat & image )
{
  int depth = 3;
  int w = image.cols;
  int h = image.rows;
  FFLD::JPEGImage *img = new FFLD::JPEGImage ( w, h, depth ); 

  uint8_t *pixels = img->bits();
  cv::Vec3b bgr;
  for ( int y = 0; y < h; y++ )
    for ( int x = 0; x < w; x++ )
    {
      cv::Point p ( x, y );
      bgr = image.at<cv::Vec3b>(p);
      for ( int i = 0; i < depth; i++ )
        pixels[ (y * w + x) * depth + i] = bgr[i];
    }
  
  return img;
}

#endif
