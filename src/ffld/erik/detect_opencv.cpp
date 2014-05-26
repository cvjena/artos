#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "DPMDetection.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencvToJPEGImage.h>

using namespace std;
using namespace FFLD;

int main ( int argc, char **argv )
{
    if ( argc < 3 ) 
    {
      cerr << "usage: " << argv[0] << " <modelfile> [images]" << endl;
      exit(-1);
    }

    string modelfile = argv[1];
    
    DPMDetection dpmdetect ( modelfile, 0.8, true /*verbose*/ );

    for ( int k = 2; k < argc ; k++ )
    {
      string file = argv[k];

      cv::Mat img = cv::imread( file, CV_LOAD_IMAGE_COLOR );
      JPEGImage *image = opencvToJPEGImage ( img ); 

      if (image->empty()) {
        cerr << "\nInvalid image " << file << endl;
        exit(-1);
      }

      std::vector< Detection > detections;
      int errcode = dpmdetect.detect ( *image, detections );

      if ( errcode != 0 ) {
        cerr << "DPM detection failed with error code = " << errcode << endl;
        exit(-1);
      }

      delete image;

      cout << "Number of detections: " << detections.size() << endl;
   
      for (int i = 0; i < std::min(10,(int)detections.size()); i++ )
        cout << file << ' ' << detections[i].classname << ' ' << detections[i].score << ' ' << (detections[i].left() + 1) << ' '
          << (detections[i].top() + 1) << ' ' << (detections[i].right() + 1) << ' '
          << (detections[i].bottom() + 1) << endl;
    }
}
