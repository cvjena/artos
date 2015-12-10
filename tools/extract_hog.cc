#include <iostream>
#include <fstream>
#include "JPEGImage.h"
#include "FeatureExtractor.h"
#include "Mixture.h"
#include "Model.h"
using namespace std;
using namespace ARTOS;

int main(int argc, char * argv[])
{
    if (argc < 3)
        cout << "Usage: " << argv[0] << " <jpeg-filename> <hog-filename>" << endl << endl
             << "Extracts HOG features from <jpeg-filename> and writes them" << endl
             << "as model file to <hog-filename>." << endl;
    else
    {
        // Load image
        JPEGImage img(argv[1]);
        if (img.empty())
        {
            cout << "Could not read JPEG file: " << argv[1] << endl;
            return 1;
        }
        
        // Compute HOG features
        shared_ptr<FeatureExtractor> fe = FeatureExtractor::create("HOG");
        FeatureMatrix feat;
        fe->extract(img, feat);
        
        if (feat.empty())
        {
            cout << "HOG computation failed." << endl;
            return 2;
        }
        
        // Write features to file
        Mixture mix;
        mix.addModel(Model(feat, 0));
        ofstream outFile(argv[2], ofstream::out | ofstream::trunc);
        if (outFile.is_open())
        {
            outFile << mix;
            outFile.close();
        }
        else
        {
            cout << "Could not open " << argv[2] << " for writing." << endl;
            return 3;
        }
    }
    return 0;
}