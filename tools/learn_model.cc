/**
* @file
* This example illustrates the usage of the `libartos` `C`-API for learning a
* object detection model with data from an image repository.
* 
* It's just a single function call!
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/


#include <iostream>
#include <cstdlib>
#include "libartos.h"
#include "sysutils.h"
using namespace std;


void printHelp(const char *);


int main(int argc, char * argv[])
{
    if (argc < 5)
    {
        printHelp(argv[0]);
        return 0;
    }
    
    unsigned int arClusters = (argc >= 6) ? strtoul(argv[5], 0, 10) : 0;
    unsigned int whoClusters = (argc >= 7) ? strtoul(argv[6], 0, 10) : 0;
    if (arClusters == 0)
        arClusters = 1;
    if (whoClusters == 0)
        whoClusters = 1;
    
    int res = learn_imagenet(
        argv[2], argv[3], argv[1],
        join_path(2, argv[4], (string(argv[3]) + ".txt").c_str()).c_str(),
        false, arClusters, whoClusters, 20, 20, ARTOS_THOPT_LOOCV, 0, true
    );
    
    if (res == ARTOS_RES_OK)
        return EXIT_SUCCESS;
    else
    {
        switch (res)
        {
            case ARTOS_IMGREPO_RES_INVALID_REPOSITORY:
                cout << "Invalid image repository." << endl;
                break;
            case ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND:
                cout << "Synset not found." << endl;
                break;
            case ARTOS_IMGREPO_RES_EXTRACTION_FAILED:
                cout << "Could not extract images from synset." << endl;
                break;
            case ARTOS_LEARN_RES_FAILED:
                cout << "Learning failed." << endl;
                break;
            case ARTOS_LEARN_RES_INVALID_BG_FILE:
                cout << "Invalid background statistics." << endl;
                break;
            case ARTOS_RES_FILE_ACCESS_DENIED:
                cout << "Could not write model file." << endl;
                break;
            default:
                cout << "Failed with error code " << res << "." << endl;
                break;
        }
        return EXIT_FAILURE;
    }
}


void printHelp(const char * progName)
{
    cout << "Learns a model for object detection using libartos." << endl << endl
         << "Usage: " << progName << " <bg-file> <image-repo> <synset-id> <model-directory> <ar-clusters = 1> <who-clusters = 1>" << endl << endl
         << "ARGUMENTS" << endl << endl
         << "    bg-file                Path to the file with the stationary background" << endl
         << "                           statistics (usually bg.dat)." << endl
         << endl
         << "    image-repo             Path to the image repository." << endl
         << endl
         << "    synset-id              ID of the synset to extract positive samples from." << endl
         << endl
         << "    model-directory        Path of the directory where the learned model file" << endl
         << "                           will be written to." << endl
         << endl
         << "    ar-clusters            Number of clusters to form by aspect ratio." << endl
         << endl
         << "    who-clusters           Number of clusters to form by WHO features." << endl;
}
