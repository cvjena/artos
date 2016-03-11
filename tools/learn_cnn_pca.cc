#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <Eigen/Eigenvalues>
#include "FeaturePyramid.h"
#include "ImageRepository.h"
#include "portable_endian.h"
using namespace ARTOS;
using namespace std;

void printHelp(const char *);
void displayProgress(unsigned int, unsigned int, int*);

int main(int argc, char * argv[])
{
    if (argc < 8)
    {
        printHelp(argv[0]);
        return 0;
    }
    
    /* Check repository */
    if (!ImageRepository::hasRepositoryStructure(argv[7]))
    {
        cout << "Invalid image repository." << endl;
        return 1;
    }
    MixedImageIterator imgIt(argv[7], 1);
    
    /* Get parameters */
    unsigned int numDim = strtoul(argv[2], NULL, 0);
    if (numDim == 0)
    {
        cout << "Number of dimensions must be greater than 0." << endl;
        return 2;
    }
    unsigned int numImages = (argc >= 9) ? strtoul(argv[8], NULL, 0) : 0;
    if (numImages == 0)
        numImages = 1000;
    string netFile(argv[3]), weightsFile(argv[4]), meanFile(argv[5]), scalesFile(argv[6]), layerName = "";
    if (argc >= 10)
        layerName = argv[9];
    
    /* Set-up feature extractor */
    shared_ptr<FeatureExtractor> fe;
    try
    {
        fe = FeatureExtractor::create("Caffe");
        fe->setParam("netFile", netFile);
        fe->setParam("weightsFile", weightsFile);
        if (!meanFile.empty())
            fe->setParam("meanFile", meanFile);
        fe->setParam("maxImgSize", 1024);
        if (!layerName.empty())
            fe->setParam("layerName", layerName);
        if (!scalesFile.empty())
            fe->setParam("scalesFile", scalesFile);
    }
    catch (const Exception & e)
    {
        cerr << "Could not create feature extractor: " << e.what() << endl;
        return 3;
    }
    if (fe->numFeatures() < numDim)
    {
        cerr << "Can not reduce feature space to " << numDim << " features, since it only has " << fe->numFeatures() << "." << endl;
        return 4;
    }
    
    /* Iterate over the images and compute the mean feature vector */
    int lastProgress = -1;
    FeatureMatrix_<double>::Cell mean = FeatureMatrix_<double>::Cell::Zero(fe->numFeatures());
    vector<FeatureMatrix>::const_iterator levelIt;
    unsigned long long numCells = 0;
    cout << "Computing mean..." << endl;
    for (imgIt.rewind(); imgIt.ready() && (unsigned int) imgIt < numImages; ++imgIt)
    {
        displayProgress((unsigned int) imgIt, numImages, &lastProgress);
        JPEGImage img = (*imgIt).getImage();
        if (!img.empty())
        {
            FeaturePyramid pyra(img, fe);
            // Loop over various scales
            for (levelIt = pyra.levels().begin(); levelIt != pyra.levels().end(); levelIt++)
            {
                mean += levelIt->asCellMatrix().cast<double>().colwise().sum().transpose();
                numCells += levelIt->numCells();
            }
        }
    }
    displayProgress(numImages, numImages, &lastProgress);
    if (numCells == 0)
    {
        cerr << "Features could not be extracted." << endl;
        return 5;
    }
    mean /= static_cast<double>(numCells);
    
    /* Iterate over the images again and compute the covariance matrix of features */
    lastProgress = -1;
    numCells = 0;
    FeatureMatrix_<double> centered;
    FeatureMatrix_<double>::ScalarMatrix cov = FeatureMatrix_<double>::ScalarMatrix::Zero(fe->numFeatures(), fe->numFeatures());
    cout << "Computing covariance matrix..." << endl;
    for (imgIt.rewind(); imgIt.ready() && (unsigned int) imgIt < numImages; ++imgIt)
    {
        displayProgress((unsigned int) imgIt, numImages, &lastProgress);
        JPEGImage img = (*imgIt).getImage();
        if (!img.empty())
        {
            FeaturePyramid pyra(img, fe);
            // Loop over various scales
            for (levelIt = pyra.levels().begin(); levelIt != pyra.levels().end(); levelIt++)
            {
                centered = *levelIt;
                centered -= mean;
                cov += centered.asCellMatrix().transpose() * centered.asCellMatrix();
                numCells += levelIt->numCells();
            }
        }
    }
    displayProgress(numImages, numImages, &lastProgress);
    cov /= static_cast<double>(numCells);
    
    /* PCA */
    cout << "Performing PCA..." << flush;
    Eigen::SelfAdjointEigenSolver<FeatureMatrix_<double>::ScalarMatrix> eigensolver(cov);
    // Eigenvectors are sorted by eigenvalue in increasing order, so we take the last right columns and reverse their order:
    FeatureMatrix_<double>::ScalarMatrix pca = eigensolver.eigenvectors().rightCols(numDim) * Eigen::PermutationMatrix<Eigen::Dynamic>(Eigen::VectorXi::LinSpaced(numDim, numDim - 1, 0));
    cout << " done." << endl;
     
    /* Save to file */
    ofstream file(argv[1], ofstream::out | ofstream::trunc);
    if (!file.is_open())
    {
        cerr << "Could not open file: " << argv[1] << endl;
        return 6;
    }
    uint32_t rows = htole32(pca.rows()), cols = htole32(pca.cols());
    file.write(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    file.write(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    float buf;
    for (int i = 0; i < mean.size(); i++)
    {
        buf = htole32(static_cast<float>(mean(i)));
        file.write(reinterpret_cast<char*>(&buf), sizeof(float));
    }
    for (int i = 0; i < pca.size(); i++)
    {
        buf = htole32(static_cast<float>(pca(i)));
        file.write(reinterpret_cast<char*>(&buf), sizeof(float));
    }
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
    cout << "Learns a transformation of the feature space of a specific layer of a CNN to a space with lower" << endl
         << "dimensionality using Principal Components Analysis." << endl << endl
         << "Usage: " << progName << " <pca-file> <num-dim> <net-file> <weights-file> <mean-file> <scales-file> <image-repository> <num-images = 1000> [<layer>]" << endl << endl
         << "ARGUMENTS" << endl << endl
         << "    pca-file               Filename where the mean feature cell and transformation matrix will be stored." << endl
         << endl
         << "    num-dim                Number of dimensions to reduce the feature space to." << endl
         << endl
         << "    net-file               Path to the the prototxt file specifying the structure of the network." << endl
         << endl
         << "    weights-file           Path to the the protobuf file with the pre-trained weights for the network." << endl
         << endl
         << "    mean-file              Path to the the mean image file (may be an empty string)." << endl
         << endl
         << "    scales-file            Path to a file with maxima for each feature channel (may be an empty string)." << endl
         << endl
         << "    image-repository       Path to the image repository." << endl
         << endl
         << "    num-images             Number of images from the repository to take into account." << endl
         << endl
         << "    layer                  The layer to extract features from. Defaults to the last layer before" << endl
         << "                           before the first fully-connected layer." << endl;
}
