#include "ModelLearnerBase.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cmath>

#include "Mixture.h"
#include "timingtools.h"

#include "clustering.h"
#include "ModelEvaluator.h"

using namespace ARTOS;
using namespace std;


void ModelLearnerBase::setFeatureExtractor(const std::shared_ptr<FeatureExtractor> & featureExtractor)
{
    this->m_models.clear();
    this->m_thresholds.clear();
    this->m_clusterSizes.clear();
    this->m_featureExtractor = (featureExtractor) ? featureExtractor : FeatureExtractor::defaultFeatureExtractor();
}


void ModelLearnerBase::reset()
{
    this->m_models.clear();
    this->m_thresholds.clear();
    this->m_clusterSizes.clear();
    this->m_samples.clear();
    this->m_numSamples = 0;
}


bool ModelLearnerBase::addPositiveSample(const SynsetImage & sample)
{
    if (!sample.valid())
        return false;
    
    Sample s;
    s.m_simg = sample;
    this->initSampleFromSynsetImage(s);
    this->m_numSamples += s.m_bboxes.size();
    this->m_samples.push_back(move(s));
    return true;
}


bool ModelLearnerBase::addPositiveSample(SynsetImage && sample)
{
    if (!sample.valid())
        return false;
    
    Sample s;
    s.m_simg = move(sample);
    this->initSampleFromSynsetImage(s);
    this->m_numSamples += s.m_bboxes.size();
    this->m_samples.push_back(move(s));
    return true;
}


bool ModelLearnerBase::addPositiveSample(const JPEGImage & sample, const Rectangle & boundingBox)
{
    if (sample.empty())
        return false;
    
    Sample s;
    s.m_img = sample;
    s.m_bboxes.push_back((boundingBox.empty()) ? Rectangle(0, 0, sample.width(), sample.height()) : boundingBox);
    s.modelAssoc.push_back(0);
    s.data = NULL;
    this->m_samples.push_back(move(s));
    this->m_numSamples++;
    return true;
}


bool ModelLearnerBase::addPositiveSample(JPEGImage && sample, const Rectangle & boundingBox)
{
    if (sample.empty())
        return false;
    
    Sample s;
    s.m_img = move(sample);
    s.m_bboxes.push_back((boundingBox.empty()) ? Rectangle(0, 0, s.m_img.width(), s.m_img.height()) : boundingBox);
    s.modelAssoc.push_back(0);
    s.data = NULL;
    this->m_samples.push_back(move(s));
    this->m_numSamples++;
    return true;
}


bool ModelLearnerBase::addPositiveSample(const JPEGImage & sample, const vector<Rectangle> & boundingBoxes)
{
    if (sample.empty())
        return false;
    
    // Check if any of the bounding boxes is empty and use only one bounding boxes spanning the entire image
    // in that case
    if (boundingBoxes.empty())
        return this->addPositiveSample(sample, Rectangle());
    for (vector<Rectangle>::const_iterator it = boundingBoxes.begin(); it != boundingBoxes.end(); it++)
        if (it->empty())
            return this->addPositiveSample(sample, *it);
    
    Sample s;
    s.m_img = sample;
    s.m_bboxes = boundingBoxes;
    s.modelAssoc.assign(s.m_bboxes.size(), 0);
    s.data = NULL;
    this->m_numSamples += boundingBoxes.size();
    this->m_samples.push_back(move(s));
    return true;
}


bool ModelLearnerBase::addPositiveSample(JPEGImage && sample, const vector<Rectangle> & boundingBoxes)
{
    if (sample.empty())
        return false;
    
    // Check if any of the bounding boxes is empty and use only one bounding boxes spanning the entire image
    // in that case
    if (boundingBoxes.empty())
        return this->addPositiveSample(move(sample), Rectangle());
    for (vector<Rectangle>::const_iterator it = boundingBoxes.begin(); it != boundingBoxes.end(); it++)
        if (it->empty())
            return this->addPositiveSample(move(sample), *it);
    
    Sample s;
    s.m_img = move(sample);
    s.m_bboxes = boundingBoxes;
    s.modelAssoc.assign(s.m_bboxes.size(), 0);
    s.data = NULL;
    this->m_numSamples += boundingBoxes.size();
    this->m_samples.push_back(move(s));
    return true;
}


void ModelLearnerBase::initSampleFromSynsetImage(Sample & s)
{
    if (s.m_simg.loadBoundingBoxes() && !s.m_simg.bboxes.empty())
    {
        // Check if any of the bounding boxes is empty and use only one bounding boxes spanning the entire image
        // in that case
        bool validBBoxes = true;
        for (vector<Rectangle>::const_iterator it = s.m_simg.bboxes.begin(); it != s.m_simg.bboxes.end(); it++)
            if (it->empty())
            {
                validBBoxes = false;
                break;
            }
        if (validBBoxes)
            s.m_bboxes = s.m_simg.bboxes;
    }
    if (s.m_bboxes.empty())
    {
        Rectangle bbox(0, 0, 0, 0);
        JPEGImage img = s.m_simg.getImage();
        bbox.setWidth(img.width());
        bbox.setHeight(img.height());
        s.m_bboxes.push_back(bbox);
    }
    
    s.modelAssoc.assign(s.m_bboxes.size(), 0);
    s.data = NULL;
}


int ModelLearnerBase::learn_init()
{
    this->m_models.clear();
    this->m_thresholds.clear();
    if (this->m_samples.empty())
        return ARTOS_LEARN_RES_NO_SAMPLES;
    return ARTOS_RES_OK;
}


int ModelLearnerBase::learn(const unsigned int maxAspectClusters, const unsigned int maxFeatureClusters, ProgressCallback progressCB, void * cbData)
{
    int res = this->learn_init();
    if (res != ARTOS_RES_OK)
        return res;
    
    int i, c;
    vector<Sample>::iterator sample;
    vector<Rectangle>::const_iterator bbox;
    
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
            for (bbox = sample->bboxes().begin(); bbox != sample->bboxes().end(); bbox++, i++)
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
        vector< vector<Size> > sampleSizes(numAspectClusters);
        for (sample = this->m_samples.begin(), i = 0; sample != this->m_samples.end(); sample++)
            for (bbox = sample->bboxes().begin(); bbox != sample->bboxes().end(); bbox++, i++)
            {
                c = aspectClusterAssignment(i);
                sampleSizes[c].push_back(Size(bbox->width(), bbox->height()));
                samplesPerAspectCluster[c]++;
            }
        for (i = 0; i < numAspectClusters; i++)
            cellNumbers[i] = this->m_featureExtractor->computeOptimalModelSize(sampleSizes[i], this->maximumModelSize());
    }
    if (this->m_verbose)
        cerr << "Computed optimal cell numbers in " << stop() << " ms." << endl;
    
    // Perform actual learning in derived class
    res = this->m_learn(aspectClusterAssignment, samplesPerAspectCluster, cellNumbers, maxFeatureClusters, progressCB, cbData);
    if (res != ARTOS_RES_OK)
        return res;
    
    // Determine number of samples per cluster
    this->m_clusterSizes.assign(this->m_models.size(), 0);
    if (this->m_models.size() == 0)
        return ARTOS_LEARN_RES_FAILED;
    for (sample = this->m_samples.begin(); sample != this->m_samples.end(); sample++)
        for (i = 0; i < sample->modelAssoc.size(); i++)
            if (sample->modelAssoc[i] < this->m_clusterSizes.size())
                this->m_clusterSizes[sample->modelAssoc[i]]++;
    
    return ARTOS_RES_OK;
    
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
            Mixture mixture(this->m_featureExtractor);
            mixture.addModel(Model(this->m_models[i], 0));
            stringstream classname;
            classname << i;
            eval.addModel(classname.str(), move(mixture), 0.0);
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


DPMDetection ModelLearnerBase::getDetector(double threshold, bool verbose, double overlap, int interval) const
{
    Mixture mix(this->m_featureExtractor);
    for (size_t i = 0; i < this->m_models.size(); i++)
    {
        Model model(this->m_models[i], -1 * this->m_thresholds[i]);
        mix.addModel(model);
    }
    return DPMDetection(move(mix), threshold, verbose, overlap, interval);
}


bool ModelLearnerBase::save(const string & filename, const bool addToMixture) const
{
    if (this->m_models.size() == 0)
        return false;

    Mixture mix(this->m_featureExtractor);
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
