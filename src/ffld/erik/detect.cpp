#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "DPMDetection.h"

using namespace std;
using namespace FFLD;


std::string trim(string s, const std::string& drop)
{
    std::string r=s.erase(s.find_last_not_of(drop)+1);
    return r.erase(0,r.find_first_not_of(drop));
}

std::string chomp(string s)
{
    return trim(trim(s, "\r"), "\n");
}


int main ( int argc, char **argv )
{
    if ( argc < 4 ) 
    {
      cerr << "usage: " << argv[0] << " <modellist> <imagelist> <outfile>" << endl;
      exit(-1);
    }

    string modellist = argv[1];
    string imagelist = argv[2];
    string outfile   = argv[3];
    
    bool verbose = true;
    DPMDetection dpmdetect ( verbose /*verbose*/ );
    
    dpmdetect.addModels ( modellist );

    ifstream ifs ( imagelist.c_str(), ifstream::in );
    
    if ( ! ifs.good() )
    {
      cerr << "Unable to read imagelist: " << imagelist << endl;
      exit(-1);
    }

    ofstream out ( outfile.c_str(), ofstream::out );
    if ( ! out.good() )
    {
      cerr << "Unable to write to " << outfile << endl; 
      exit(-1);
    }

    while ( ifs.good() )
    {
      string file;
      if ( !(ifs >> file) )
        break;

      file = chomp(file);
      JPEGImage image( file );

      if ( verbose )
        cerr << "Processing image: " << file << endl;

      if (image.empty()) {
        cerr << "Invalid image <" << file << ">" << endl;
      } else {

        std::vector< Detection > detections;
        int errcode = dpmdetect.detect ( image, detections );

        if ( errcode != 0 ) {
          cerr << "DPM detection failed with error code = " << errcode << " for image " << file << endl;
          continue;
        }
        
        if ( verbose )
          cerr << "Number of detections: " << detections.size() << endl;
     
        for (int i = 0; i < std::min(10,(int)detections.size()); i++ )
          out << file << ' ' << detections[i].classname << ' ' << detections[i].score << ' ' << (detections[i].left() + 1) << ' '
            << (detections[i].top() + 1) << ' ' << (detections[i].right() + 1) << ' '
            << (detections[i].bottom() + 1) << endl;
      }
    }

    out.close();
    ifs.close();
}
