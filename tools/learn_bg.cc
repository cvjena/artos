#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include "FeatureExtractor.h"
#include "ImageRepository.h"
#include "StationaryBackground.h"
using namespace ARTOS;
using namespace std;
using namespace std::chrono;

void printHelp(const char *);
bool displayProgress(unsigned int, unsigned int, void*);

int main(int argc, char * argv[])
{
    if (argc < 3)
    {
        printHelp(argv[0]);
        return 0;
    }
    
    // Check repository
    if (!ImageRepository::hasRepositoryStructure(argv[2]))
    {
        cout << "Invalid image repository." << endl;
        return 1;
    }
    MixedImageIterator imgIt(argv[2], 1);
    
    // Get parameters
    unsigned int numImages = (argc >= 4) ? strtoul(argv[3], NULL, 0) : 0;
    unsigned int maxOffset = (argc >= 5) ? strtoul(argv[4], NULL, 0) : 0;
    bool accurate = (argc >= 6) ? static_cast<bool>(strtoul(argv[5], NULL, 0)) : false;
    string layerName = "";
    if (numImages == 0)
        numImages = 1000;
    if (maxOffset == 0)
        maxOffset = 19;
    
    // Learn
    StationaryBackground bg;
    int lastProgress = -1;
    cout << "Learning negative mean" << endl;
    auto start = high_resolution_clock::now();
    bg.learnMean(imgIt, numImages, &displayProgress, reinterpret_cast<void*>(&lastProgress));
    auto stop = high_resolution_clock::now();
    cout << "Took " << duration_cast<milliseconds>(stop - start).count() << " ms." << endl << endl;
    lastProgress = -1;
    cout << "Learning autocorrelation function" << endl;
    start = high_resolution_clock::now();
    if (accurate)
        bg.learnCovariance_accurate(imgIt, numImages, maxOffset, &displayProgress, reinterpret_cast<void*>(&lastProgress));
    else
        bg.learnCovariance(imgIt, numImages, maxOffset, &displayProgress, reinterpret_cast<void*>(&lastProgress));
    stop = high_resolution_clock::now();
    cout << "Took " << duration_cast<seconds>(stop - start).count() << " s." << endl << endl;
   
    if (!bg.learnedAllOffsets)
        cout << "Note: Images were not big enough to learn covariance for all offsets." << endl;
     
    // Save
    if (!bg.writeToFile(argv[1]))
        cerr << "Could not write the computed statistics to disk!" << endl;
    
    return 0;
}


bool displayProgress(unsigned int current, unsigned int total, void * data)
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
    return true;
}


void printHelp(const char * progName)
{
    cout << "Learns stationary background statistics which are necessary for learning WHO models." << endl << endl
         << "Usage: " << progName << " <bg-file> <image-repository> <num-images = 1000> <max-offset = 19> [<accurate = 0>]" << endl << endl
         << "ARGUMENTS" << endl << endl
         << "    bg-file                Filename where the learned statistics will be written to." << endl
         << endl
         << "    image-repository       Path to the image repository." << endl
         << endl
         << "    num-images             Number of images from the repository to take into account" << endl
         << "                           for computing statistics." << endl
         << endl
         << "    max-offset             Maximum offset to be learned (will restrict the maximum" << endl
         << "                           possible model size in cells)." << endl
         << endl
         << "    accurate               If set to 1, the accurate, but very slow method for" << endl
         << "                           computing the autocorrelation function will be used." << endl;
}
