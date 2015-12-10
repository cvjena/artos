#include <iostream>
#include <fstream>
#include "StationaryBackground.h"
#include "JPEGImage.h"
#include "FeatureExtractor.h"
#include "Mixture.h"
#include "Model.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
using namespace std;
using namespace ARTOS;

int main(int argc, char * argv[])
{
    if (argc < 4)
        cout << "Usage: " << argv[0] << " <jpeg-filename> <who-filename> <bg-file>" << endl << endl
             << "Extracts HOG features from <jpeg-filename>, whitens them using the" << endl
             << "background statistics in <bg-file> and writes the resulting WHO features" << endl
             << "as model file to <who-filename>." << endl;
    else
    {
        // Load background statistics
        StationaryBackground bg(argv[3]);
        if (bg.empty())
        {
            cout << "Could not read background statistics." << endl;
            return 4;
        }
        
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
        
        // Preparation
        FeatureCell negMean = FeatureCell::Zero(fe->numFeatures());
        negMean.head(bg.getNumFeatures()) = bg.mean;
        Eigen::LLT<ScalarMatrix> llt;
        {
            ScalarMatrix cov = bg.computeFlattenedCovariance(feat.rows(), feat.cols(), fe->numFeatures());
            float lambda = 0.0f;
            ScalarMatrix identity = ScalarMatrix::Identity(cov.rows(), cov.cols());
            do
            {
                lambda += 0.01f;
                llt.compute(cov + identity * lambda);
            }
            while (llt.info() != Eigen::Success);
        }
        int i, j, k, l;
        // Centre
        feat -= negMean;
        // Whiten feature vector
        FeatureCell posVector = feat.asVector();
        llt.solveInPlace(posVector);
        // Normalize and shape back into matrix
        feat.asVector() = posVector / posVector.cwiseAbs().maxCoeff();
        
        // Write WHO features to file
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