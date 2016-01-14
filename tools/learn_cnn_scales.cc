#include <iostream>
#include <fstream>
#include <cstdlib>
#include "FeaturePyramid.h"
#include "ImageRepository.h"
using namespace ARTOS;
using namespace std;

void printHelp(const char *);
void displayProgress(unsigned int, unsigned int, int*);

int main(int argc, char * argv[])
{
    if (argc < 6)
    {
        printHelp(argv[0]);
        return 0;
    }
    
    // Check repository
    if (!ImageRepository::hasRepositoryStructure(argv[5]))
    {
        cout << "Invalid image repository." << endl;
        return 1;
    }
    MixedImageIterator imgIt(argv[5], 1);
    
    // Get parameters
    unsigned int numImages = (argc >= 7) ? strtoul(argv[6], NULL, 0) : 0;
    if (numImages == 0)
        numImages = 1000;
    string netFile(argv[2]), weightsFile(argv[3]), meanFile(argv[4]), layerName = "";
    if (argc >= 8)
        layerName = argv[7];
    
    // Set-up feature extractor
    shared_ptr<FeatureExtractor> fe;
    try
    {
        fe = FeatureExtractor::create("Caffe");
        fe->setParam("netFile", netFile);
        fe->setParam("weightsFile", weightsFile);
        fe->setParam("meanFile", meanFile);
        fe->setParam("maxImgSize", 1024);
        if (!layerName.empty())
            fe->setParam("layerName", layerName);
    }
    catch (const Exception & e)
    {
        cerr << "Could not create feature extractor: " << e.what() << endl;
        return 2;
    }
    
    // Iterate over the images and compute the mean feature vector
    int i, lastProgress = -1;
    FeatureCell maxima = FeatureCell::Zero(fe->numFeatures());
    vector<FeatureMatrix>::const_iterator levelIt;
    for (imgIt.rewind(); imgIt.ready() && (unsigned int) imgIt < numImages; ++imgIt)
    {
        displayProgress((unsigned int) imgIt, numImages, &lastProgress);
        JPEGImage img = (*imgIt).getImage();
        if (!img.empty())
        {
            FeaturePyramid pyra(img, fe);
            // Loop over various scales
            for (levelIt = pyra.levels().begin(); levelIt != pyra.levels().end(); levelIt++)
                for (i = 0; i < levelIt->numCells(); i++)
                    maxima = maxima.cwiseMax(levelIt->cell(i).cwiseAbs());
        }
    }
    displayProgress(numImages, numImages, &lastProgress);
     
    // Save
    ofstream file(argv[1], ofstream::out | ofstream::trunc);
    if (!file.is_open())
    {
        cerr << "Could not open file: " << argv[1] << endl;
        return 3;
    }
    file << maxima;
    file.close();
    
    return 0;
}


void displayProgress(unsigned int current, unsigned int total, int * data)
{
    int * lastProgress = reinterpret_cast<int*>(data);
    if (*lastProgress < 0)
    {
        cout << "....................";
        *lastProgress = 0;
    }
    else
    {
        int progress = (current * 20) / total;
        if (progress > 20)
            progress = 20;
        if (progress > *lastProgress)
        {
            int i;
            for (i = 0; i < 20; i++)
                cout << static_cast<char>(8);
            for (i = 0; i < progress; i++)
                cout << "|";
            for (i = progress; i < 20; i++)
                cout << ".";
            cout << flush;
            if (current >= total)
                cout << endl;
            *lastProgress = progress;
        }
    }
}


void printHelp(const char * progName)
{
    cout << "Determines the maximum absolute value of each channel of a specific layer of a CNN" << endl
         << "based on features extracted from sample images." << endl
         << "The computed maxima may be used for scaling feature values." << endl << endl
         << "Usage: " << progName << " <scales-file> <net-file> <weights-file> <mean-file> <image-repository> <num-images = 1000> [<layer>]" << endl << endl
         << "ARGUMENTS" << endl << endl
         << "    scales-file            Filename where the channel-wise maxima will be written to." << endl
         << endl
         << "    net-file               Path to the the prototxt file specifying the structure of the network." << endl
         << endl
         << "    weights-file           Path to the the protobuf file with the pre-trained weights for the network." << endl
         << endl
         << "    mean-file              Path to the the mean image file." << endl
         << endl
         << "    image-repository       Path to the image repository." << endl
         << endl
         << "    num-images             Number of images from the repository to take into account." << endl
         << endl
         << "    layer                  The layer to extract features from. Defaults to the last layer before" << endl
         << "                           before the first fully-connected layer." << endl;
}
