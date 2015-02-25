#include "ModelLearnerBase.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include "ffld/Mixture.h"
#include "ffld/timingtools.h"

#include "clustering.h"
#include "ModelEvaluator.h"

using namespace ARTOS;
using namespace FFLD;
using namespace std;


void ModelLearnerBase::reset()
{
    this->m_models.clear();
    this->m_thresholds.clear();
    this->m_clusterSizes.clear();
    this->m_samples.clear();
    this->m_numSamples = 0;
}


ModelLearnerBase::Size ModelLearnerBase::computeOptimalCellNumber(const std::vector<int> & widths, const std::vector<int> & heights)
{
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
    size.width = max(static_cast<int>(round(width / FeatureExtractor::cellSize)), 1);
    size.height = max(static_cast<int>(round(height / FeatureExtractor::cellSize)), 1);
    return size;
}


bool ModelLearnerBase::addPositiveSample(const JPEGImage & sample, const FFLD::Rectangle & boundingBox)
{
    if (sample.empty())
        return false;
    Sample s;
    s.img = sample;
    s.bboxes.push_back((boundingBox.empty()) ? FFLD::Rectangle(0, 0, s.img.width(), s.img.height()) : boundingBox);
    s.modelAssoc.push_back(0);
    s.data = NULL;
    this->m_samples.push_back(s);
    this->m_numSamples++;
    return true;
}


bool ModelLearnerBase::addPositiveSample(const JPEGImage & sample, const vector<FFLD::Rectangle> & boundingBoxes)
{
    if (sample.empty())
        return false;
    
    // Check if any of the bounding boxes is empty and use only one bounding boxes spanning the entire image
    // in that case
    if (boundingBoxes.empty())
        return this->addPositiveSample(sample, FFLD::Rectangle());
    for (vector<FFLD::Rectangle>::const_iterator it = boundingBoxes.begin(); it != boundingBoxes.end(); it++)
        if (it->empty())
            return this->addPositiveSample(sample, *it);
    
    Sample s;
    s.img = sample;
    s.bboxes = boundingBoxes;
    s.modelAssoc.assign(s.bboxes.size(), 0);
    s.data = NULL;
    this->m_samples.push_back(s);
    this->m_numSamples += boundingBoxes.size();
    return true;
}


bool ModelLearnerBase::learn_init()
{
    this->m_models.clear();
    this->m_thresholds.clear();
    return (!this->m_samples.empty());
}


bool ModelLearnerBase::learn(const unsigned int maxAspectClusters, const unsigned int maxFeatureClusters, ProgressCallback progressCB, void * cbData)
{
    if (!this->learn_init())
        return false;
    
    int i, c;
    vector<Sample>::iterator sample;
    vector<FFLD::Rectangle>::iterator bbox;
    
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
    
    // Perform actual learning in derived class
    this->m_learn(aspectClusterAssignment, samplesPerAspectCluster, cellNumbers, maxFeatureClusters, progressCB, cbData);
    
    // Determine number of samples per cluster
    this->m_clusterSizes.assign(this->m_models.size(), 0);
    if (this->m_models.size() == 0)
        return false;
    for (sample = this->m_samples.begin(); sample != this->m_samples.end(); sample++)
        for (i = 0; i < sample->modelAssoc.size(); i++)
            if (sample->modelAssoc[i] < this->m_clusterSizes.size())
                this->m_clusterSizes[sample->modelAssoc[i]]++;
    
    return true;
    
}


const vector<float> & ModelLearnerBase::optimizeThreshold(const unsigned int maxPositive, const vector<JPEGImage> * negative,
                                                          const float b, ProgressCallback progressCB, void * cbData)
{
    if (this->m_models.size() > 0)
    {
        if (this->m_verbose)
        {
            cerr << "-- Calculating optimal threshold combination by F-measure --" << endl;
            cerr << "Positive samples: ";
            if (maxPositive > 0)
                cerr << "~" << maxPositive * this->m_models.size();
            else
                cerr << this->getNumSamples();
            cerr << endl;
            if (negative != NULL)
                cerr << "Negative samples: " << negative->size() << endl;
        }
        
        // Build vector of pointers to positive samples
        vector<Sample*> positive;
        positive.reserve(this->m_samples.size());
        for (vector<Sample>::iterator sample = this->m_samples.begin(); sample != this->m_samples.end(); sample++)
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
        if (this->m_models.size() == 1)
        {
            eval.testModels(positive, maxPositive, negative, 100, progressCB, cbData);
            this->m_thresholds.resize(this->m_models.size(), 0);
            for (size_t i = 0; i < this->m_models.size(); i++)
                this->m_thresholds[i] = eval.getMaxFMeasure(i, b).first;
        }
        else
            this->m_thresholds = eval.searchOptimalThresholdCombination(positive, maxPositive, negative, 100, b, progressCB, cbData);
        
        if (this->m_verbose)
        {
            for (size_t i = 0; i < this->m_thresholds.size(); i++)
                cerr << "Threshold for model #" << i << ": " << this->m_thresholds[i] << endl;
            cerr << "Found optimal thresholds in " << stop() << " ms." << endl;
        }
    }
    return this->m_thresholds;
}


bool ModelLearnerBase::save(const string & filename, const bool addToMixture) const
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
