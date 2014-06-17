#include "ModelLearner.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include <Eigen/Cholesky>

#include "ffld/Mixture.h"
#include "ffld/Intersector.h"
#include "ffld/timingtools.h"

#include "clustering.h"
#include "ModelEvaluator.h"

using namespace ARTOS;
using namespace FFLD;
using namespace std;


Mixture * loo_who(const Mixture *, const Sample *, const unsigned int, const unsigned int, void *);

typedef struct {
    vector<unsigned int> * clusterSizes;
    vector<HOGPyramid::Scalar> * normFactors;
} loo_data_t;


void ModelLearner::reset()
{
    this->m_models.clear();
    this->m_thresholds.clear();
    this->m_clusterSizes.clear();
    this->m_normFactors.clear();
    this->m_samples.clear();
    this->m_numSamples = 0;
}


ModelLearner::Size ModelLearner::computeOptimalCellNumber(const std::vector<int> & widths, const std::vector<int> & heights)
{
    if (this->m_bg.empty())
    {
        Size s;
        s.width = 0;
        s.height = 0;
        return s;
    }
    
    int i, j, w, h;
    
    // Fill histogram and area vector
    Eigen::Array<float, 1, 201> hist; // histogram of logarithmic aspect ratios with bins from -2 to +2 in steps of 0.02
    hist.setConstant(0.0f);
    vector<int> areas(min(widths.size(), heights.size()));
    int aspectIndex;
    for (i = 0; i < areas.size(); i++)
    {
        w = widths[i];
        h = heights[i];
        areas[i] = w * h;
        aspectIndex = round(log(static_cast<float>(h) / static_cast<float>(w)) * 50 + 100);
        if (aspectIndex >= 0 && aspectIndex < hist.size())
            hist(aspectIndex) += 1;
    }
    
    // Filter histogram with large gaussian smoothing filter and select maximum as aspect ratio
    Eigen::Array<float, 1, 201> filter;
    for (i = 0; i < filter.size(); i++)
        filter(i) = exp(static_cast<float>((100 - i) * (100 - i)) / -400.0f);
    float curValue, maxValue = 0;
    int maxIndex = 0;
    for (i = 0; i < hist.size(); i++)
    {
        curValue = 0;
        for (j = max(i - 100, 0); j < min(i + 100, 200); j++)
            curValue += hist(j) * filter(j - i + 100);
        if (curValue > maxValue)
        {
            maxIndex = i;
            maxValue = curValue;
        }
    }
    float aspect = exp(maxIndex * 0.02f - 2);
    
    // Nasty hack from the original WHO code: pick 20 percentile area and
    // ensure that HOG feature areas are neither too big nor too small
    sort(areas.begin(), areas.end());
    int area = areas[static_cast<size_t>(floor(areas.size() * 0.2))];
    area = max(min(area, 7000), 5000);
    
    // Calculate model size in cells
    float width = sqrt(static_cast<float>(area) / aspect);
    float height = width * aspect;
    Size size;
    size.width = max(static_cast<int>(round(width / this->m_bg.cellSize)), 1);
    size.height = max(static_cast<int>(round(height / this->m_bg.cellSize)), 1);
    return size;
}


void ModelLearner::addPositiveSample(const JPEGImage & sample, const FFLD::Rectangle & boundingBox)
{
    if (sample.empty())
        return;
    WHOSample s;
    s.img = sample;
    s.bboxes.push_back((boundingBox.empty()) ? FFLD::Rectangle(0, 0, s.img.width(), s.img.height()) : boundingBox);
    s.modelAssoc.push_back(0);
    s.whoFeatures.push_back(HOGPyramid::Level());
    this->m_samples.push_back(s);
    this->m_numSamples++;
}


void ModelLearner::addPositiveSample(const JPEGImage & sample, const vector<FFLD::Rectangle> & boundingBoxes)
{
    if (sample.empty())
        return;
    
    // Check if any of the bounding boxes is empty and use only one bounding boxes spanning the entire image
    // in that case
    if (boundingBoxes.empty())
    {
        this->addPositiveSample(sample, FFLD::Rectangle());
        return;
    }
    for (vector<FFLD::Rectangle>::const_iterator it = boundingBoxes.begin(); it != boundingBoxes.end(); it++)
        if (it->empty())
        {
            this->addPositiveSample(sample, *it);
            return;
        }
    
    WHOSample s;
    s.img = sample;
    s.bboxes = boundingBoxes;
    s.modelAssoc.assign(s.bboxes.size(), 0);
    s.whoFeatures.assign(s.bboxes.size(), HOGPyramid::Level());
    this->m_samples.push_back(s);
    this->m_numSamples += boundingBoxes.size();
}


bool ModelLearner::learn(const unsigned int maxAspectClusters, const unsigned int maxWHOClusters, ProgressCallback progressCB, void * cbData)
{
    this->m_models.clear();
    this->m_normFactors.clear();
    this->m_thresholds.clear();
    if (this->m_bg.empty() || this->m_samples.empty() || this->m_bg.getNumFeatures() > HOGPyramid::NbFeatures)
        return false;

    unsigned int c, i, j, k, l, s, t;  // yes, we do need that much iteration variables
    vector<WHOSample>::iterator sample;
    vector<FFLD::Rectangle>::iterator bbox;
    vector<HOGPyramid::Level>::iterator whoStorage;
    
    // Cluster by aspect ratio
    Eigen::VectorXi aspectClusterAssignment = Eigen::VectorXi::Zero(this->getNumSamples());
    unsigned int numAspectClusters = 1;
    if (maxAspectClusters > 1)
    {
        if (this->m_verbose)
            start();
        // Calculate aspect ratios
        Eigen::VectorXf aspects(this->getNumSamples());
        for (sample = this->m_samples.begin(), i = 0; sample != this->m_samples.end(); sample++)
            for (bbox = sample->bboxes.begin(); bbox != sample->bboxes.end(); bbox++, i++)
                aspects(i) = static_cast<float>(bbox->height()) / static_cast<float>(bbox->width());
        // Perform k-means clustering
        Eigen::VectorXf centroids;
        repeatedKMeansClustering(aspects, maxAspectClusters, &aspectClusterAssignment, &centroids, 100);
        mergeNearbyClusters(aspectClusterAssignment, centroids, 0.2f);
        numAspectClusters = centroids.rows();
        if (this->m_verbose)
            cerr << "Formed " << numAspectClusters << " clusters by aspect ratio in " << stop() << " ms." << endl;
    }
    
    // Compute optimal cell number for each aspect ratio
    if (this->m_verbose)
        start();
    vector<Size> cellNumbers(numAspectClusters);
    vector<int> samplesPerAspectCluster(numAspectClusters, 0);
    {
        vector< vector<int> > widths(numAspectClusters), heights(numAspectClusters);
        for (sample = this->m_samples.begin(), i = 0; sample != this->m_samples.end(); sample++)
            for (bbox = sample->bboxes.begin(); bbox != sample->bboxes.end(); bbox++, i++)
            {
                c = aspectClusterAssignment(i);
                widths[c].push_back(bbox->width());
                heights[c].push_back(bbox->height());
                samplesPerAspectCluster[c]++;
            }
        for (i = 0; i < numAspectClusters; i++)
            cellNumbers[i] = this->computeOptimalCellNumber(widths[i], heights[i]);
    }
    if (this->m_verbose)
        cerr << "Computed optimal cell numbers in " << stop() << " ms." << endl;
    
    // Learn models for each aspect ratio cluster
    HOGPyramid::Cell negMean = HOGPyramid::Cell::Zero();
    negMean.head(this->m_bg.getNumFeatures()) = this->m_bg.mean;
    unsigned int curClusterIndex = 0;
    unsigned int progressStep = 0, progressTotal = numAspectClusters * 2;
    if (progressCB != NULL)
        progressCB(progressStep, progressTotal, cbData);
    for (c = 0; c < numAspectClusters; c++)
    {
        Size modelSize = cellNumbers[c];
        if (this->m_verbose)
            cerr << "-- Learning model for aspect ratio cluster " << (c+1) << " --" << endl
                 << "There are " << samplesPerAspectCluster[c] << " samples in this cluster." << endl
                 << "Optimal cell number: " << modelSize.width << " x " << modelSize.height << endl;
        
        // Get background covariance
        if (this->m_verbose)
            start();
        Eigen::LLT<StationaryBackground::Matrix> llt;
        {
            StationaryBackground::Matrix cov = this->m_bg.computeFlattenedCovariance(modelSize.height, modelSize.width, HOGPyramid::NbFeatures);
            if (cov.size() == 0)
            {
                if (this->m_verbose)
                {
                    cerr << "Reconstruction of covariance matrix failed - skipping this cluster" << endl;
                    stop();
                }
                for (sample = this->m_samples.begin(), i = 0; sample != this->m_samples.end(); sample++)
                    for (bbox = sample->bboxes.begin(), j = 0; bbox != sample->bboxes.end(); bbox++, j++, i++)
                        if (aspectClusterAssignment(i) == c)
                            sample->modelAssoc[j] = static_cast<unsigned int>(-1);
                progressStep += 2;
                if (progressCB != NULL)
                    progressCB(progressStep, progressTotal, cbData);
                continue;
            }
            if (this->m_verbose)
            {
                cerr << "Reconstructed covariance in " << stop() << " ms." << endl;
                start();
            }
            // Cholesky decomposition for stable inversion
            float lambda = 0.0f; // regularizer
            StationaryBackground::Matrix identity = StationaryBackground::Matrix::Identity(cov.rows(), cov.cols());
            do
            {
                lambda += 0.01f; // increase regularizer on every attempt
                llt.compute(cov + identity * lambda);
                if (this->m_verbose && llt.info() != Eigen::Success)
                    cerr << "Cholesky decomposition failed - increasing regularizer." << endl;
            }
            while (llt.info() != Eigen::Success);
            if (this->m_verbose)
            {
                cerr << "Cholesky decomposition in " << stop() << " ms." << endl;
                start();
            }
        }
        progressStep++;
        if (progressCB != NULL)
            progressCB(progressStep, progressTotal, cbData);
        
        // Compute negative bias term in advance: mu_0'*S^-1*mu_0
        Eigen::VectorXf hogVector(modelSize.height * modelSize.width * HOGPyramid::NbFeatures);
        Eigen::VectorXf negVector(hogVector.size());
        negVector.setConstant(0.0f);
        // Replicate negative mean over all cells
        for (i = 0, l = 0; i < modelSize.height; i++)
            for (j = 0; j < modelSize.width; j++)
                for (k = 0; k < HOGPyramid::NbFeatures; k++, l++)
                    negVector(l) = negMean(k);
        float biasNeg = negVector.dot(llt.solve(negVector));
        if (this->m_verbose)
        {
            cerr << "Computed negative bias term in " << stop() << " ms." << endl;
            start();
        }
        
        // Extract HOG features from samples, optionally cluster and whiten them 
        HOGPyramid::Level positive = HOGPyramid::Level::Constant( // accumulator for positive features
            modelSize.height, modelSize.width, HOGPyramid::Cell::Zero()
        );
        Eigen::VectorXf posVector(positive.rows() * positive.cols() * HOGPyramid::NbFeatures); // flattened version of `positive`
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> whoCentroids;
        Eigen::VectorXf biases;
        if ((maxWHOClusters <= 1 || samplesPerAspectCluster[c] == 1) && !this->m_loocv)
        {
            // Procedure without WHO clustering and LOOCV:
            // Just average over all positive samples, centre and whiten them.
            
            for (sample = this->m_samples.begin(), i = 0; sample != this->m_samples.end(); sample++)
                for (bbox = sample->bboxes.begin(), j = 0; bbox != sample->bboxes.end(); bbox++, j++, i++)
                    if (aspectClusterAssignment(i) == c)
                    {
                        JPEGImage resizedSample = sample->img.crop(bbox->x(), bbox->y(), bbox->width(), bbox->height())
                                                             .resize(modelSize.width * this->m_bg.cellSize, modelSize.height * this->m_bg.cellSize);
                        HOGPyramid::Level hog;
                        HOGPyramid::Hog(resizedSample, hog, 1, 1, this->m_bg.cellSize); // compute HOG features
                        positive += hog.block(1, 1, positive.rows(), positive.cols()); // cut off padding and add to feature accumulator
                        sample->modelAssoc[j] = curClusterIndex;
                    }
            if (this->m_verbose)
            {
                cerr << "Computed HOG features of positive samples in " << stop() << " ms." << endl;
                start();
            }
            
            // Average positive features and flatten the matrix into a vector
            for (i = 0, l = 0; i < positive.rows(); i++)
                for (j = 0; j < positive.cols(); j++)
                    for (k = 0; k < HOGPyramid::NbFeatures; k++, l++)
                        posVector(l) = positive(i, j)(k) / static_cast<float>(this->getNumSamples());
            
            // Centre positive features
            Eigen::VectorXf posCentred = posVector - negVector;
            if (this->m_verbose)
            {
                cerr << "Centred positive feature vector in " << stop() << " ms." << endl;
                start();
            }
            
            // Now we compute MODEL = cov^-1 * (pos - neg) = cov^-1 * posCentred = (L * LT)^-1 * posCentred = LT^-1 * L^-1 * posCentred
            // llt.solveInPlace() will do this for us by solving the linear equation system cov * MODEL = posCentred
            llt.solveInPlace(posCentred);
            if (this->m_verbose)
                cerr << "Whitened feature vector in " << stop() << " ms." << endl;
            whoCentroids = posCentred.transpose();
            // We can obtain an estimated bias of the model as BIAS = (neg' * cov^-1 * neg - pos' * cov^-1 * pos) / 2
            // (under the assumption, that the a-priori class-probability is 0.5)
            float biasPos = posVector.dot(llt.solve(posVector));
            biases = Eigen::VectorXf::Constant(1, (biasNeg - biasPos) / 2.0f);
            curClusterIndex++;
        }
        else
        {
            // Procedure with WHO clustering or LOOCV:
            // Centre and whiten the HOG feature vector of each sample and use those WHO vectors for
            // clustering. We can then use the centroids of the clusters as models.
            
            // Extract HOG features of each sample, centre, whiten and store them
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hogFeatures(samplesPerAspectCluster[c], posVector.size());
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> whoFeatures(samplesPerAspectCluster[c], posVector.size());
            for (sample = this->m_samples.begin(), s = 0, t = 0; sample != this->m_samples.end(); sample++)
                for (bbox = sample->bboxes.begin(), whoStorage = sample->whoFeatures.begin(); bbox != sample->bboxes.end(); bbox++, whoStorage++, s++)
                    if (aspectClusterAssignment(s) == c)
                    {
                        // Extract HOG features
                        JPEGImage resizedSample = sample->img.crop(bbox->x(), bbox->y(), bbox->width(), bbox->height())
                                                             .resize(modelSize.width * this->m_bg.cellSize, modelSize.height * this->m_bg.cellSize);
                        HOGPyramid::Level hog;
                        HOGPyramid::Hog(resizedSample, hog, 1, 1, this->m_bg.cellSize); // compute HOG features
                        positive = hog.block(1, 1, positive.rows(), positive.cols()); // cut off padding and add to feature accumulator
                        // Flatten HOG feature matrix into vector
                        hogFeatures.row(t).setConstant(0.0f);
                        for (i = 0, l = 0; i < positive.rows(); i++)
                            for (j = 0; j < positive.cols(); j++)
                                for (k = 0; k < HOGPyramid::NbFeatures; k++, l++)
                                    hogFeatures(t, l) = positive(i, j)(k);
                        // Centre feature vector
                        posVector = hogFeatures.row(t).transpose() - negVector;
                        // Whiten feature vector
                        llt.solveInPlace(posVector);
                        // Store
                        whoFeatures.row(t++) = posVector;
                        // Shape back into matrix and store it with the sample if we need it later for LOOCV
                        if (this->m_loocv)
                        {
                            whoStorage->resize(positive.rows(), positive.cols());
                            for (i = 0, l = 0; i < positive.rows(); i++)
                                for (j = 0; j < positive.cols(); j++)
                                    for (k = 0; k < HOGPyramid::NbFeatures; k++, l++)
                                        (*whoStorage)(i, j)(k) = posVector(l);
                        }
                    }
            if (this->m_verbose)
            {
                cerr << "Computed WHO features of positive samples in " << stop() << " ms." << endl;
                start();
            }
            
            // Cluster by WHO features
            Eigen::VectorXi whoClusterAssignment = Eigen::VectorXi::Zero(whoFeatures.rows());
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmpCentroids;
            repeatedKMeansClustering(whoFeatures, min(maxWHOClusters, static_cast<unsigned int>(samplesPerAspectCluster[c])),
                                     &whoClusterAssignment, &tmpCentroids, 30);
            // Ignore clusters with too few samples and compute positive bias terms
            if (this->m_verbose)
            {
                cerr << "Number of samples in WHO clusters:";
                for (i = 0; i < tmpCentroids.rows(); i++)
                    cerr << " " << whoClusterAssignment.cwiseEqual(i).count();
                cerr << endl;
            }
            whoCentroids.resize(tmpCentroids.rows(), tmpCentroids.cols());
            biases.resize(tmpCentroids.rows());
            float biasPos;
            for (i = 0, t = 0; i < tmpCentroids.rows(); i++)
                if (whoClusterAssignment.cwiseEqual(i).count() >= whoClusterAssignment.rows() / 10)
                {
                    whoCentroids.row(t) = tmpCentroids.row(i);
                    hogVector.setConstant(0.0f);
                    for (j = 0; j < whoClusterAssignment.size(); j++)
                        if (whoClusterAssignment(j) == i)
                        {
                            whoClusterAssignment(j) = t;
                            hogVector += hogFeatures.row(j);
                        }
                    hogVector /= static_cast<float>(whoClusterAssignment.cwiseEqual(t).count());
                    biasPos = hogVector.dot(llt.solve(hogVector)); // positive bias term: pos' * cov^-1 * pos
                    biases(t) = (biasNeg - biasPos) / 2.0f; // assumes an a-priori class-probability of 0.5, so that we don't need to add ln(phi/(1-phi))
                    t++;
                }
                else
                {
                    if (this->m_verbose)
                        cerr << "Ignoring WHO cluster #" << i << " (too few samples)." << endl;
                    whoCentroids.conservativeResize(whoCentroids.rows() - 1, whoCentroids.cols());
                    biases.conservativeResize(biases.size() - 1);
                    for (j = 0; j < whoClusterAssignment.size(); j++)
                        if (whoClusterAssignment(j) == i)
                            whoClusterAssignment(j) = -1;
                }
            
            // Save cluster assignment for samples
            for (sample = this->m_samples.begin(), s = 0, t = 0; sample != this->m_samples.end(); sample++)
                for (bbox = sample->bboxes.begin(), j = 0; bbox != sample->bboxes.end(); bbox++, s++, j++)
                    if (aspectClusterAssignment(s) == c)
                    {
                        sample->modelAssoc[j] = (whoClusterAssignment(t) >= 0)
                                                   ? curClusterIndex + whoClusterAssignment(t)
                                                   : static_cast<unsigned int>(-1);
                        t++;
                    }
            curClusterIndex += whoCentroids.rows();
            if (this->m_verbose)
                cerr << "Subdivided aspect ratio cluster in " << whoCentroids.rows() << " clusters by WHO features in " << stop() << " ms." << endl;
        }
        
        // Finally normalize the model vectors and reshape them back into rows and columns
        Eigen::VectorXf centroid;
        for (s = 0; s < whoCentroids.rows(); s++)
        {
            centroid = whoCentroids.row(s);
            this->m_normFactors.push_back(centroid.cwiseAbs().maxCoeff());
            centroid /= this->m_normFactors.back();
            for (i = 0, l = 0; i < positive.rows(); i++)
                for (j = 0; j < positive.cols(); j++)
                    for (k = 0; k < HOGPyramid::NbFeatures; k++, l++)
                        positive(i, j)(k) = centroid(l);
            this->m_models.push_back(positive);
            this->m_thresholds.push_back(-1 * biases(s) / this->m_normFactors.back());
            if (this->m_verbose)
                cerr << "Estimated threshold for model #" << this->m_thresholds.size() - 1 << ": " << this->m_thresholds.back() << endl;
        }
        
        progressStep++;
        if (progressCB != NULL)
            progressCB(progressStep, progressTotal, cbData);
    }
    
    this->m_clusterSizes.assign(this->m_models.size(), 0);
    for (sample = this->m_samples.begin(); sample != this->m_samples.end(); sample++)
        for (i = 0; i < sample->modelAssoc.size(); i++)
            if (sample->modelAssoc[i] < this->m_clusterSizes.size())
                this->m_clusterSizes[sample->modelAssoc[i]]++;
    
    return true;
}


const vector<float> & ModelLearner::optimizeThreshold(const unsigned int maxPositive, const vector<JPEGImage> * negative,
                                                      const float b, ProgressCallback progressCB, void * cbData)
{
    if (this->m_models.size() > 0)
    {
        if (this->m_verbose)
            cerr << "-- Calculating optimal thresholds by F-measure" << ((this->m_loocv) ? " using LOOCV" : "") << " --" << endl;
        this->m_thresholds.resize(this->m_models.size(), 0);
        
        // Build vector of pointers to positive samples
        vector<Sample*> positive;
        positive.reserve(this->m_samples.size());
        for (vector<WHOSample>::iterator sample = this->m_samples.begin(); sample != this->m_samples.end(); sample++)
            positive.push_back(&(*sample));
        
        // Create an evaluator for the learned models
        ModelEvaluator eval;
        for (size_t i = 0; i < this->m_models.size(); i++)
        {
            Mixture mixture;
            mixture.addModel(Model(this->m_models[i], 0));
            stringstream classname;
            classname << i;
            eval.addModel(classname.str(), mixture, 0.0);
        }
        
        // Test models against samples
        if (this->m_verbose)
            start();
        ModelEvaluator::LOOFunc looFunc = NULL;
        void * looData = NULL;
        loo_data_t looDataStruct;
        looDataStruct.clusterSizes = &this->m_clusterSizes;
        looDataStruct.normFactors = &this->m_normFactors;
        if (this->m_loocv)
        {
            looFunc = &loo_who;
            looData = static_cast<void*>(&looDataStruct);
        }
        eval.testModels(positive, maxPositive, negative, 100, progressCB, cbData, looFunc, looData);
        if (this->m_verbose)
        {
            cerr << "Tested models against ";
            if (maxPositive > 0)
                cerr << "~" << maxPositive * this->m_models.size();
            else
                cerr << this->getNumSamples();
            cerr << " positive";
            if (negative != NULL)
                cerr << " and " << negative->size() << " negative";
            cerr << " samples in " << stop() << " ms." << endl;
            start();
        }
        
        // Get threshold with maximum F-measure for each model
        for (size_t i = 0; i < this->m_models.size(); i++)
        {
            this->m_thresholds[i] = eval.getMaxFMeasure(i, b).first;
            if (this->m_verbose)
                cerr << "Threshold for model #" << i << ": " << this->m_thresholds[i] << endl;
        }
        if (this->m_verbose)
            cerr << "Found optimal thresholds in " << stop() << " ms." << endl;
    }
    return this->m_thresholds;
}


const vector<float> & ModelLearner::optimizeThresholdCombination(const unsigned int maxPositive, const vector<JPEGImage> * negative,
                                                                 int mode, const float b, ProgressCallback progressCB, void * cbData)
{
    if (this->m_models.size() > 0)
    {
        if (this->m_verbose)
        {
            cerr << "-- Calculating optimal threshold combination by F-measure" << ((this->m_loocv) ? " using LOOCV" : "") << " --" << endl;
            if (mode == 1)
                cerr << "Mode: Best Combination" << endl;
            else
                cerr << "Mode: Harmony Search" << endl;
            cerr << "Positive samples: ";
            if (maxPositive > 0)
                cerr << "~" << maxPositive * this->m_models.size();
            else
                cerr << this->getNumSamples();
            cerr << endl;
            if (negative != NULL)
                cerr << "Negative samples: " << negative->size() << endl;
        }
        this->m_thresholds.resize(this->m_models.size(), 0);
        
        // Build vector of pointers to positive samples
        vector<Sample*> positive;
        positive.reserve(this->m_samples.size());
        for (vector<WHOSample>::iterator sample = this->m_samples.begin(); sample != this->m_samples.end(); sample++)
            positive.push_back(&(*sample));
        
        // Create an evaluator for the learned models
        ModelEvaluator eval;
        for (size_t i = 0; i < this->m_models.size(); i++)
        {
            Mixture mixture;
            mixture.addModel(Model(this->m_models[i], 0));
            stringstream classname;
            classname << i;
            eval.addModel(classname.str(), mixture, 0.0);
        }
        
        // Test models against samples
        if (this->m_verbose)
            start();
        ModelEvaluator::LOOFunc looFunc = NULL;
        void * looData = NULL;
        loo_data_t looDataStruct;
        looDataStruct.clusterSizes = &this->m_clusterSizes;
        looDataStruct.normFactors = &this->m_normFactors;
        if (this->m_loocv)
        {
            looFunc = &loo_who;
            looData = static_cast<void*>(&looDataStruct);
        }
        if (mode == 1)
            this->m_thresholds = eval.computeOptimalBiasCombination(positive, maxPositive, negative, 1, b, progressCB, cbData, looFunc, looData);
        else
            this->m_thresholds = eval.searchOptimalBiasCombination(positive, maxPositive, negative, 100, b, progressCB, cbData, looFunc, looData);
        
        if (this->m_verbose)
        {
            for (size_t i = 0; i < this->m_thresholds.size(); i++)
                cerr << "Threshold for model #" << i << ": " << this->m_thresholds[i] << endl;
            cerr << "Found optimal thresholds in " << stop() << " ms." << endl;
        }
    }
    return this->m_thresholds;
}


bool ModelLearner::save(const string & filename, const bool addToMixture) const
{
    if (this->m_models.size() == 0)
        return false;

    Mixture mix;
    // Try to read existing mixture file
    if (addToMixture)
    {
        ifstream infile(filename.c_str());
        if (infile.is_open() && infile.good())
            infile >> mix;
    }
    
    for (size_t i = 0; i < this->m_models.size(); i++)
    {
        Model model(this->m_models[i], -1 * this->m_thresholds[i]);
        mix.addModel(model);
    }
    
    // Write to file
    ofstream outfile(filename.c_str(), ofstream::out | ofstream::trunc);
    if (outfile.is_open() && outfile.good())
    {
        outfile << mix;
        return true;
    }
    else
        return false;
}


Mixture * loo_who(const Mixture * orig, const Sample * sample, const unsigned int objectIndex, const unsigned int numLeftOut, void * data)
{
    loo_data_t * looData = reinterpret_cast<loo_data_t*>(data);
    const WHOSample * whoSample = dynamic_cast<const WHOSample*>(sample);
    if (whoSample != NULL && whoSample->whoFeatures[objectIndex].size() > 0
            && whoSample->modelAssoc[objectIndex] < looData->clusterSizes->size()
            && (*(looData->clusterSizes))[whoSample->modelAssoc[objectIndex]] > numLeftOut + 1)
    {
        unsigned int clusterSize = (*(looData->clusterSizes))[whoSample->modelAssoc[objectIndex]];
        HOGPyramid::Scalar normFactor = (*(looData->normFactors))[whoSample->modelAssoc[objectIndex]];
        unsigned int n = clusterSize - numLeftOut;
        const HOGPyramid::Level * origModel = &(orig->models()[0].filters(0));
        const HOGPyramid::Level * sampleFeatures = &(whoSample->whoFeatures[objectIndex]);
        HOGPyramid::Level newModel(origModel->rows(), origModel->cols());
        for (HOGPyramid::Level::Index i = 0; i < newModel.size(); i++)
            newModel(i) = ((*origModel)(i) * (static_cast<HOGPyramid::Scalar>(n) * normFactor)
                           - (*sampleFeatures)(i)) / (static_cast<HOGPyramid::Scalar>(n - 1) * normFactor);
        Mixture * replacement = new Mixture();
        replacement->addModel(Model(newModel, orig->models()[0].bias()));
        return replacement;
    }
    else
        return NULL;
}
