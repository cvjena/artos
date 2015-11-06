#include "ModelLearner.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cmath>

#include <Eigen/Cholesky>

#include "ffld/Mixture.h"
#include "ffld/timingtools.h"

#include "clustering.h"
#include "ModelEvaluator.h"

using namespace ARTOS;
using namespace FFLD;
using namespace std;


Mixture * loo_who(const Mixture *, const Sample *, const unsigned int, const unsigned int, void *);

typedef struct {
    vector<unsigned int> * clusterSizes;
    vector<FeatureExtractor::Scalar> * normFactors;
} loo_data_t;


void ModelLearner::reset()
{
    for (vector<Sample>::iterator sample = this->m_samples.begin(); sample != this->m_samples.end(); sample++)
        if (sample->data != NULL)
            delete reinterpret_cast< vector<FeatureExtractor::FeatureMatrix> *>(sample->data);
    ModelLearnerBase::reset();
    this->m_normFactors.clear();
}


bool ModelLearner::addPositiveSample(const SynsetImage & sample)
{
    if (ModelLearnerBase::addPositiveSample(sample))
    {
        Sample & s = this->m_samples.back();
        s.data = reinterpret_cast<void*>(new vector<FeatureExtractor::FeatureMatrix>(s.bboxes().size(), FeatureExtractor::FeatureMatrix()));
        return true;
    }
    else
        return false;
}


bool ModelLearner::addPositiveSample(SynsetImage && sample)
{
    if (ModelLearnerBase::addPositiveSample(move(sample)))
    {
        Sample & s = this->m_samples.back();
        s.data = reinterpret_cast<void*>(new vector<FeatureExtractor::FeatureMatrix>(s.bboxes().size(), FeatureExtractor::FeatureMatrix()));
        return true;
    }
    else
        return false;
}


bool ModelLearner::addPositiveSample(const JPEGImage & sample, const FFLD::Rectangle & boundingBox)
{
    if (ModelLearnerBase::addPositiveSample(sample, boundingBox))
    {
        this->m_samples.back().data = reinterpret_cast<void*>(new vector<FeatureExtractor::FeatureMatrix>(1, FeatureExtractor::FeatureMatrix()));
        return true;
    }
    else
        return false;
}


bool ModelLearner::addPositiveSample(JPEGImage && sample, const FFLD::Rectangle & boundingBox)
{
    if (ModelLearnerBase::addPositiveSample(move(sample), boundingBox))
    {
        this->m_samples.back().data = reinterpret_cast<void*>(new vector<FeatureExtractor::FeatureMatrix>(1, FeatureExtractor::FeatureMatrix()));
        return true;
    }
    else
        return false;
}


bool ModelLearner::addPositiveSample(const JPEGImage & sample, const vector<FFLD::Rectangle> & boundingBoxes)
{
    if (ModelLearnerBase::addPositiveSample(sample, boundingBoxes))
    {
        Sample & s = this->m_samples.back();
        s.data = reinterpret_cast<void*>(new vector<FeatureExtractor::FeatureMatrix>(s.bboxes().size(), FeatureExtractor::FeatureMatrix()));
        return true;
    }
    else
        return false;
}


bool ModelLearner::addPositiveSample(JPEGImage && sample, const vector<FFLD::Rectangle> & boundingBoxes)
{
    if (ModelLearnerBase::addPositiveSample(move(sample), boundingBoxes))
    {
        Sample & s = this->m_samples.back();
        s.data = reinterpret_cast<void*>(new vector<FeatureExtractor::FeatureMatrix>(s.bboxes().size(), FeatureExtractor::FeatureMatrix()));
        return true;
    }
    else
        return false;
}


bool ModelLearner::learn_init()
{
    this->m_normFactors.clear();
    return (ModelLearnerBase::learn_init() && !this->m_bg.empty() && this->m_bg.getNumFeatures() <= FeatureExtractor::numFeatures);
}


void ModelLearner::m_learn(Eigen::VectorXi & aspectClusterAssignment, vector<int> & samplesPerAspectCluster, vector<Size> & cellNumbers,
                           const unsigned int maxWHOClusters, ProgressCallback progressCB, void * cbData)
{
    unsigned int c, i, j, k, l, s, t; // yes, we do need that much iteration variables
    unsigned int numAspectClusters = samplesPerAspectCluster.size();
    vector<Sample>::iterator sample;
    vector<FFLD::Rectangle>::const_iterator bbox;
    vector<FeatureExtractor::FeatureMatrix>::iterator whoStorage;
    
    // Learn models for each aspect ratio cluster
    FeatureExtractor::Cell negMean = FeatureExtractor::Cell::Zero();
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
            StationaryBackground::Matrix cov = this->m_bg.computeFlattenedCovariance(modelSize.height, modelSize.width, FeatureExtractor::numFeatures);
            if (cov.size() == 0)
            {
                if (this->m_verbose)
                {
                    cerr << "Reconstruction of covariance matrix failed - skipping this cluster" << endl;
                    stop();
                }
                for (sample = this->m_samples.begin(), i = 0; sample != this->m_samples.end(); sample++)
                    for (bbox = sample->bboxes().begin(), j = 0; bbox != sample->bboxes().end(); bbox++, j++, i++)
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
        
        Eigen::VectorXf hogVector(modelSize.height * modelSize.width * FeatureExtractor::numFeatures);
        // Replicate negative mean over all cells
        Eigen::VectorXf negVector = negMean.replicate(modelSize.height * modelSize.width, 1);
        // Compute negative bias term in advance: mu_0'*S^-1*mu_0
        float biasNeg = negVector.dot(llt.solve(negVector));
        if (this->m_verbose)
        {
            cerr << "Computed negative bias term in " << stop() << " ms." << endl;
            start();
        }
        
        // Extract HOG features from samples, optionally cluster and whiten them 
        FeatureExtractor::FeatureMatrix positive = FeatureExtractor::FeatureMatrix::Constant( // accumulator for positive features
            modelSize.height, modelSize.width, FeatureExtractor::Cell::Zero()
        );
        Eigen::VectorXf posVector(positive.rows() * positive.cols() * FeatureExtractor::numFeatures); // flattened version of `positive`
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> whoCentroids;
        Eigen::VectorXf biases;
        if ((maxWHOClusters <= 1 || samplesPerAspectCluster[c] == 1) && !this->m_loocv)
        {
            // Procedure without WHO clustering and LOOCV:
            // Just average over all positive samples, centre and whiten them.
            
            for (sample = this->m_samples.begin(), i = 0; sample != this->m_samples.end(); sample++)
                for (bbox = sample->bboxes().begin(), j = 0; bbox != sample->bboxes().end(); bbox++, j++, i++)
                    if (aspectClusterAssignment(i) == c)
                    {
                        JPEGImage resizedSample = sample->img().crop(bbox->x(), bbox->y(), bbox->width(), bbox->height())
                                                               .resize(modelSize.width * this->m_bg.cellSize, modelSize.height * this->m_bg.cellSize);
                        FeatureExtractor::FeatureMatrix hog;
                        FeatureExtractor::extract(resizedSample, hog); // compute HOG features
                        positive += hog; // add to feature accumulator
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
                    for (k = 0; k < FeatureExtractor::numFeatures; k++, l++)
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
                for (bbox = sample->bboxes().begin(), whoStorage = reinterpret_cast< vector<FeatureExtractor::FeatureMatrix> *>(sample->data)->begin(); bbox != sample->bboxes().end(); bbox++, whoStorage++, s++)
                    if (aspectClusterAssignment(s) == c)
                    {
                        // Extract HOG features
                        JPEGImage resizedSample = sample->img().crop(bbox->x(), bbox->y(), bbox->width(), bbox->height())
                                                               .resize(modelSize.width * this->m_bg.cellSize, modelSize.height * this->m_bg.cellSize);
                        FeatureExtractor::extract(resizedSample, positive); // compute HOG features
                        // Flatten HOG feature matrix into vector
                        hogFeatures.row(t).setConstant(0.0f);
                        for (i = 0, l = 0; i < positive.rows(); i++)
                            for (j = 0; j < positive.cols(); j++)
                                for (k = 0; k < FeatureExtractor::numFeatures; k++, l++)
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
                                    for (k = 0; k < FeatureExtractor::numFeatures; k++, l++)
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
                for (bbox = sample->bboxes().begin(), j = 0; bbox != sample->bboxes().end(); bbox++, s++, j++)
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
                    for (k = 0; k < FeatureExtractor::numFeatures; k++, l++)
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
}


const vector<float> & ModelLearner::optimizeThreshold(const unsigned int maxPositive, const vector<JPEGImage> * negative,
                                                      const float b, ProgressCallback progressCB, void * cbData)
{
    if (this->m_models.size() > 0)
    {
        if (this->m_verbose)
        {
            cerr << "-- Calculating optimal threshold combination by F-measure" << ((this->m_loocv) ? " using LOOCV" : "") << " --" << endl;
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
            eval.addModel(classname.str(), move(mixture), 0.0);
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
        if (this->m_models.size() == 1)
        {
            eval.testModels(positive, maxPositive, negative, 100, progressCB, cbData, looFunc, looData);
            this->m_thresholds.resize(this->m_models.size(), 0);
            for (size_t i = 0; i < this->m_models.size(); i++)
                this->m_thresholds[i] = eval.getMaxFMeasure(i, b).first;
        }
        else
            this->m_thresholds = eval.searchOptimalThresholdCombination(positive, maxPositive, negative, 100, b, progressCB, cbData, looFunc, looData);
        
        if (this->m_verbose)
        {
            for (size_t i = 0; i < this->m_thresholds.size(); i++)
                cerr << "Threshold for model #" << i << ": " << this->m_thresholds[i] << endl;
            cerr << "Found optimal thresholds in " << stop() << " ms." << endl;
        }
    }
    return this->m_thresholds;
}


Mixture * loo_who(const Mixture * orig, const Sample * sample, const unsigned int objectIndex, const unsigned int numLeftOut, void * data)
{
    loo_data_t * looData = reinterpret_cast<loo_data_t*>(data);
    const vector<FeatureExtractor::FeatureMatrix> * whoFeatures = reinterpret_cast<const vector<FeatureExtractor::FeatureMatrix> *>(sample->data);
    if ((*whoFeatures)[objectIndex].size() > 0
            && sample->modelAssoc[objectIndex] < looData->clusterSizes->size()
            && (*(looData->clusterSizes))[sample->modelAssoc[objectIndex]] > numLeftOut + 1)
    {
        unsigned int clusterSize = (*(looData->clusterSizes))[sample->modelAssoc[objectIndex]];
        FeatureExtractor::Scalar normFactor = (*(looData->normFactors))[sample->modelAssoc[objectIndex]];
        unsigned int n = clusterSize - numLeftOut;
        const FeatureExtractor::FeatureMatrix * origModel = &(orig->models()[0].filters(0));
        const FeatureExtractor::FeatureMatrix * sampleFeatures = &((*whoFeatures)[objectIndex]);
        FeatureExtractor::FeatureMatrix newModel(origModel->rows(), origModel->cols());
        for (FeatureExtractor::FeatureMatrix::Index i = 0; i < newModel.size(); i++)
            newModel(i) = ((*origModel)(i) * (static_cast<FeatureExtractor::Scalar>(n) * normFactor)
                           - (*sampleFeatures)(i)) / (static_cast<FeatureExtractor::Scalar>(n - 1) * normFactor);
        Mixture * replacement = new Mixture();
        replacement->addModel(Model(newModel, orig->models()[0].bias()));
        return replacement;
    }
    else
        return NULL;
}
