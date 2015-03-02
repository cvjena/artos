#include "ModelEvaluator.h"

#include <set>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <fstream>

#include <Eigen/Core>

#include "harmony_search.h"
#include "ffld/Intersector.h"

using namespace ARTOS;
using namespace FFLD;
using namespace std;


float hs_fmeasure_cb(const vector<float> & biases, const vector<int> & biasIndices, void * vdata);

typedef struct {
    const vector< vector<const Detection*> > * objectDetections;
    const vector< vector<unsigned int> > * fpPerBias;
    unsigned int numPositive;
    float b;
    float b2;
    float b21;
} hs_fmeasure_data_t;


vector< pair<float, float> > ModelEvaluator::calculateFMeasures(const unsigned int modelIndex, const float b) const
{
    vector< pair<float, float> > fMeasures;
    if (modelIndex >= 0 && modelIndex < this->m_results.size())
    {
        const vector<TestResult> & results = this->m_results[modelIndex];
        fMeasures.reserve(results.size());
        // Calculate F-measure by evaluating ((1 + b²) * TP) / (b² * NP + TP + FP),
        // where TP is the number of true positives, FP is the number of false positives
        // and NP is the total number of positive samples.
        const float b2 = (b > 0) ? b * b : 1;
        const float b12 = 1 + b2;
        for (vector<TestResult>::const_iterator result = results.begin(); result != results.end(); result++)
            fMeasures.push_back(pair<float, float>(
                    result->threshold,
                    (b12 * result->tp) / (b2 * result->np + result->tp + result->fp)
            ));
    }
    return fMeasures;
}


pair<float, float> ModelEvaluator::getMaxFMeasure(const unsigned int modelIndex, const float b) const
{
    vector< pair<float, float> > fMeasures = this->calculateFMeasures(modelIndex, b);
    pair<float, float> maxFMeasure(0.0f, 0.0f);
    for (vector< pair<float, float> >::const_iterator it = fMeasures.begin(); it != fMeasures.end(); it++)
        if (it->second >= maxFMeasure.second)
            maxFMeasure = *it;
    return maxFMeasure;
}


float ModelEvaluator::computeAveragePrecision(const unsigned int modelIndex) const
{
    float ap = 0.0;
    if (modelIndex >= 0 && modelIndex < this->m_results.size())
    {
        float maxPrecision = 0.0, currentRecall = 2.0, recall;
        for (vector<TestResult>::const_iterator result = this->m_results[modelIndex].begin(); result != this->m_results[modelIndex].end(); result++)
        {
            recall = static_cast<float>(result->tp) / static_cast<float>(result->np);
            if (recall < currentRecall)
                ap += maxPrecision * (currentRecall - recall);
            currentRecall = recall;
            if (result->tp + result->fp > 0)
                maxPrecision = max(static_cast<float>(result->tp) / static_cast<float>(result->tp + result->fp), maxPrecision);
        }
        if (currentRecall > 1e-6) // always include recall == 0
            ap += maxPrecision * currentRecall;
    }
    return ap;
}


void ModelEvaluator::testModels(const vector<Sample*> & positive, unsigned int maxSamples,
                                const vector<FFLD::JPEGImage> * negative,
                                const unsigned int granularity,
                                ProgressCallback progressCB, void * cbData, LOOFunc looFunc, void * looData)
{
    if (this->getNumModels() == 0)
        return;
    this->m_results.assign(this->getNumModels(), vector<TestResult>());
    
    // Test model against samples
    SampleDetectionsVector * detections = new SampleDetectionsVector();
    vector<unsigned int> numPositive = this->runDetector(*detections, positive, maxSamples, negative, progressCB, cbData, looFunc, looData);
    if (!detections->empty())
    {
        SampleDetectionsVector::const_iterator detIt;
        for (unsigned int modelIndex = 0; modelIndex < this->getNumModels(); modelIndex++)
        {
            // Determine minimum and maximum detection score
            float minScore = numeric_limits<float>::max(), maxScore = numeric_limits<float>::min();
            for (detIt = detections->begin(); detIt != detections->end(); detIt++)
                if (detIt->second.modelIndex == modelIndex)
                {
                    if (detIt->second.score > maxScore)
                        maxScore = detIt->second.score;
                    if (detIt->second.score < minScore)
                        minScore = detIt->second.score;
                }
            
            // Count true positives and total amount of detections for each threshold
            // from minScore to maxScore in steps of 1/granularity
            int iMinScore = round(minScore * granularity);
            int iMaxScore = round(maxScore * granularity);
            Eigen::Array<float, 1, Eigen::Dynamic> tp(iMaxScore - iMinScore + 1);
            tp.setConstant(0);
            Eigen::Array<float, 1, Eigen::Dynamic> fp(tp);
            Eigen::Array< Eigen::Array<bool, 1, Eigen::Dynamic>, 1, Eigen::Dynamic> detected(positive.size());
            {
                int sampleIndex, bboxIndex;
                int scoreIndex;
                bool isPositive;
                Sample * sample;
                for (detIt = detections->begin(); detIt != detections->end(); detIt++)
                    if (detIt->second.modelIndex == modelIndex)
                    {
                        sampleIndex = detIt->first;
                        scoreIndex = round(detIt->second.score * granularity) - iMinScore;
                        isPositive = false;
                        if (sampleIndex >= 0)
                        {
                            sample = positive[sampleIndex];
                            if (detected(sampleIndex).size() == 0)
                                detected(sampleIndex) = Eigen::Array<bool, 1, Eigen::Dynamic>::Constant(1, sample->bboxes.size(), false);
                            // Treat as true positive if detection area overlaps with bounding box
                            // by at least 50%
                            Intersector intersect(detIt->second);
                            for (bboxIndex = 0; bboxIndex < sample->bboxes.size(); bboxIndex++)
                                if (!detected(sampleIndex)(bboxIndex)
                                        && sample->modelAssoc[bboxIndex] == detIt->second.modelIndex
                                        && intersect(sample->bboxes[bboxIndex]))
                                {
                                    isPositive = true;
                                    detected(sampleIndex)(bboxIndex) = true; // count only one detection for the same object
                                    break;
                                }
                        }
                        if (isPositive)
                            tp.head(scoreIndex + 1) += 1;
                        else
                            fp.head(scoreIndex + 1) += 1;
                    }
            }
            
            // Convert indices to thresholds and store results
            this->m_results[modelIndex].reserve(tp.size());
            TestResult result;
            result.np = numPositive[modelIndex];
            for (int i = 0; i < tp.size(); i++)
            {
                result.threshold = static_cast<float>(iMinScore + i) / granularity;
                result.tp = tp(i);
                result.fp = fp(i);
                this->m_results[modelIndex].push_back(result);
            }
        }
    }
    delete detections;
}


vector<float> ModelEvaluator::searchOptimalThresholdCombination(
                                const vector<Sample*> & positive, unsigned int maxSamples,
                                const vector<FFLD::JPEGImage> * negative,
                                const unsigned int granularity, const float b,
                                ProgressCallback progressCB, void * cbData, LOOFunc looFunc, void * looData)
{
    vector<float> bestBiasCombo(this->getNumModels(), 0.0f);
    if (this->getNumModels() == 0)
        return bestBiasCombo;
    
    // Test model against samples
    SampleDetectionsVector * detections = new SampleDetectionsVector();
    vector<unsigned int> numPositive = this->runDetector(*detections, positive, maxSamples, negative, progressCB, cbData, looFunc, looData);
    unsigned int numPositiveTotal = accumulate(numPositive.begin(), numPositive.end(), static_cast<unsigned int>(0));
    if (!detections->empty())
    {
        SampleDetectionsVector::const_iterator detIt;
        int i, sampleIndex, bboxIndex, objectBaseIndex;
        bool isPositive;
        Sample * sample;
        
        // Build a list of all possible biases
        vector< vector<float> > biases;
        biases.reserve(this->getNumModels());
        {
            vector< set<float> > biasSets(this->getNumModels());
            for (detIt = detections->begin(); detIt != detections->end(); detIt++)
                biasSets[detIt->second.modelIndex].insert(static_cast<int>(detIt->second.score * granularity) / static_cast<float>(granularity));
            for (vector< set<float> >::const_iterator it = biasSets.begin(); it != biasSets.end(); it++)
                biases.push_back(vector<float>(it->begin(), it->end()));
        }
        
        // Associate detections with samples
        vector< vector<const Detection*> > objectDetections;
        vector< vector<unsigned int> > fpPerBias;
        fpPerBias.reserve(biases.size());
        for (vector< vector<float> >::const_iterator it = biases.begin(); it != biases.end(); it++)
            fpPerBias.push_back(vector<unsigned int>(it->size(), 0));
        for (sampleIndex = 0, objectBaseIndex = 0; sampleIndex < positive.size(); sampleIndex++)
        {
            sample = positive[sampleIndex];
            for (i = 0; i < sample->bboxes.size(); i++)
                objectDetections.push_back(vector<const Detection*>());
            for (detIt = detections->begin(); detIt != detections->end(); detIt++)
                if (detIt->first == sampleIndex)
                {
                    isPositive = false;
                    // Treat as true positive if detection area overlaps with bounding box
                    // by at least 50%
                    Intersector intersect(detIt->second);
                    for (bboxIndex = 0; bboxIndex < sample->bboxes.size(); bboxIndex++)
                        if (intersect(sample->bboxes[bboxIndex]))
                        {
                            isPositive = true;
                            objectDetections[objectBaseIndex + bboxIndex].push_back(&detIt->second);
                            break;
                        }
                    if (!isPositive)
                        for (i = 0; i < fpPerBias[detIt->second.modelIndex].size() && biases[detIt->second.modelIndex][i] <= detIt->second.score; i++)
                            fpPerBias[detIt->second.modelIndex][i]++;
                }
            objectBaseIndex += sample->bboxes.size();
        }
        // Count false positives per model and bias
        for (detIt = detections->begin(); detIt != detections->end(); detIt++)
            if (detIt->first < 0)
                for (i = 0; i < fpPerBias[detIt->second.modelIndex].size() && biases[detIt->second.modelIndex][i] <= detIt->second.score; i++)
                    fpPerBias[detIt->second.modelIndex][i]++;
        
        // Approximate best bias combination using Harmony Search
        hs_fmeasure_data_t cb_data;
        cb_data.objectDetections = &objectDetections;
        cb_data.fpPerBias = &fpPerBias;
        cb_data.numPositive = numPositiveTotal;
        cb_data.b = b;
        cb_data.b2 = b * b;
        cb_data.b21 = 1 + cb_data.b2;
        bestBiasCombo = repeated_harmony_search(hs_fmeasure_cb, biases, reinterpret_cast<void*>(&cb_data), true);
    }
    delete detections;
    return bestBiasCombo;
}


vector<unsigned int> ModelEvaluator::runDetector(SampleDetectionsVector & detections,
                                                 const vector<Sample*> & positive, unsigned int maxSamples,
                                                 const vector<FFLD::JPEGImage> * negative,
                                                 ProgressCallback progressCB, void * cbData, LOOFunc looFunc, void * looData)
{
    size_t numModels = this->getNumModels();
    vector<unsigned int> numPositive(numModels);
    if (numModels == 0)
        return numPositive;
    
    // Set detection thresholds to a minimal value
    for (map<string, double>::iterator it = this->thresholds.begin(); it != this->thresholds.end(); it++)
        it->second = -100.0;
    
    // Set up parameters for progress callback
    unsigned int totalNumSamples, numSamplesProcessed = 0;
    if (maxSamples == 0 || maxSamples > positive.size())
        totalNumSamples = positive.size();
    else
        totalNumSamples = maxSamples * numPositive.size();
    if (negative != NULL)
        totalNumSamples += negative->size();
    
    // Save pointers to original models for the case that LOO-Cross-Validation is performed
    map<string, Mixture*> originalMixtures;
    Mixture * replacement = NULL;
    vector<Mixture*> replacementModels;
    vector<Mixture*>::iterator mixtureIt;
    vector<unsigned int> numLeftOut;
    vector<std::string> classnames;
    if (looFunc != NULL)
    {
        originalMixtures = this->mixtures;
        replacementModels.reserve(numModels);
        numLeftOut.assign(numModels, 0);
        for (size_t i = 0; i < numModels; i++)
            classnames.push_back(this->getClassnameFromIndex(i));
    }
    
    // Run detector against positive samples
    detections.clear();
    vector<Detection> sampleDetections;
    vector<Detection>::const_iterator detection;
    vector<unsigned int>::const_iterator modelAssocIt;
    unsigned int modelAssocIndex;
    bool needSample;
    for (int i = 0; i < positive.size() && (maxSamples == 0 || *(min_element(numPositive.begin(), numPositive.end())) < maxSamples); i++)
    {
        // Check if we need this sample at all
        needSample = false;
        for (modelAssocIt = positive[i]->modelAssoc.begin(); modelAssocIt != positive[i]->modelAssoc.end(); modelAssocIt++)
            if (maxSamples == 0 || *modelAssocIt < numModels && numPositive[*modelAssocIt] < maxSamples)
                needSample = true;
        if (!needSample)
            continue;
        for (modelAssocIt = positive[i]->modelAssoc.begin(), modelAssocIndex = 0; modelAssocIt != positive[i]->modelAssoc.end(); modelAssocIt++, modelAssocIndex++)
            if (*modelAssocIt < numModels)
            {
                numPositive[*modelAssocIt] += 1;
                if (looFunc != NULL) // Check for leave-one-out replacement
                {
                    replacement = looFunc(this->mixtures[classnames[*modelAssocIt]], positive[i], modelAssocIndex, numLeftOut[*modelAssocIt], looData);
                    if (replacement != NULL && replacement != this->mixtures[classnames[*modelAssocIt]])
                    {
                        replacementModels.push_back(replacement);
                        numLeftOut[*modelAssocIt]++;
                        this->mixtures[classnames[*modelAssocIt]] = replacement;
                        this->initw = -1;
                        this->inith = -1;
                    }
                }
            }
        // Run detector and store detections
        try
        {
            this->detect(positive[i]->img, sampleDetections);
        }
        catch (const bad_alloc&)
        {
            sampleDetections.clear();
            for (modelAssocIt = positive[i]->modelAssoc.begin(); modelAssocIt != positive[i]->modelAssoc.end(); modelAssocIt++)
                if (*modelAssocIt < numModels)
                    numPositive[*modelAssocIt] -= 1;
        }
        for (detection = sampleDetections.begin(); detection != sampleDetections.end(); detection++)
            detections.push_back(pair<int, Detection>(i, *detection));
        sampleDetections.clear();
        // Restore original models in the case of leave-one-out validation
        if (!replacementModels.empty())
        {
            for (mixtureIt = replacementModels.begin(); mixtureIt != replacementModels.end(); mixtureIt++)
                delete *mixtureIt;
            replacementModels.clear();
            numLeftOut.assign(numModels, 0);
            this->mixtures = originalMixtures;
            this->initw = -1;
            this->inith = -1;
        }
        // Update progress
        numSamplesProcessed++;
        if (progressCB != NULL && !progressCB(numSamplesProcessed, totalNumSamples, cbData))
        {
            detections.clear();
            numPositive.assign(numPositive.size(), 0);
            return numPositive;
        }
    }
    
    // Run detector against negative samples
    if (negative != NULL)
    {
        for (int i = 0; i < negative->size(); i++)
            if (!(*negative)[i].empty())
            {
                try
                {
                    this->detect((*negative)[i], sampleDetections);
                }
                catch (const bad_alloc&)
                {
                    sampleDetections.clear();
                }
                for (detection = sampleDetections.begin(); detection != sampleDetections.end(); detection++)
                    detections.push_back(pair<int, Detection>(-1 * (i + 1), *detection));
                sampleDetections.clear();
                if (progressCB != NULL && !progressCB(++numSamplesProcessed, totalNumSamples, cbData))
                {
                    detections.clear();
                    numPositive.assign(numPositive.size(), 0);
                    return numPositive;
                }
            }
    }
    if (progressCB != NULL && !progressCB(totalNumSamples, totalNumSamples, cbData))
    {
        detections.clear();
        numPositive.assign(numPositive.size(), 0);
        return numPositive;
    }
    return numPositive;
}


bool ModelEvaluator::dumpTestResults(const string & filename, const int modelIndex,
                                     const bool headline, const unsigned int measures,
                                     const char separator) const
{
    if (this->m_results.size() == 0 || (modelIndex >= 0 && modelIndex > this->m_results.size()))
        return false;
    
    ofstream file(filename.c_str(), ofstream::out | ofstream::trunc);
    if (!file)
        return false;

    const bool includePrecision = (measures & PRECISION);
    const bool includeRecall = (measures & RECALL);
    const bool includeFMeasure = (measures & FMEASURE);
    vector< pair<float, float> > fMeasures;
    const vector<TestResult> * results;
    
    // Write headline
    if (headline)
    {
        if (modelIndex < 0 && this->m_results.size() > 1)
            file << "Model" << separator;
        file << "Threshold" << separator << "TP" << separator << "FP" << separator << "NP";
        if (includePrecision)
            file << separator << "Precision";
        if (includeRecall)
            file << separator << "Recall";
        if (includeFMeasure)
            file << separator << "F-Measure";
        file << endl;
    }
    
    // Write data for each model
    size_t model, i;
    const TestResult * result;
    for (model = 0; model < this->m_results.size(); model++)
    {
        if (modelIndex >= 0 && model != modelIndex)
            continue;
        results = &this->m_results[model];
        if (includeFMeasure)
            fMeasures = this->calculateFMeasures(model);
        for (i = 0; i < results->size(); i++)
        {
            result = &((*results)[i]);
            if (modelIndex < 0 && this->m_results.size() > 1)
                file << model << separator;
            file << result->threshold << separator;
            file << result->tp << separator;
            file << result->fp << separator;
            file << result->np;
            if (includePrecision)
                file << separator << ((result->tp + result->fp > 0) ? (static_cast<float>(result->tp) / (result->tp + result->fp)) : 1.0f);
            if (includeRecall)
                file << separator << static_cast<float>(result->tp) / result->np;
            if (includeFMeasure)
                file << separator << fMeasures[i].second;
            file << endl;
        }
    }
    
    file.close();
    return true;
}



float hs_fmeasure_cb(const vector<float> & biases, const vector<int> & biasIndices, void * vdata)
{
    hs_fmeasure_data_t * data = reinterpret_cast<hs_fmeasure_data_t*>(vdata);
    // Count true and false positives
    unsigned int i, tp = 0, fp = 0;
    vector< vector<const Detection*> >::const_iterator objIt;
    vector<const Detection*>::const_iterator detPIt;
    for (objIt = (data->objectDetections)->begin(); objIt != (data->objectDetections)->end(); objIt++)
        for (detPIt = objIt->begin(); detPIt != objIt->end(); detPIt++)
            if ((*detPIt)->score >= biases[(*detPIt)->modelIndex])
            {
                tp++;
                break;
            }
    for (i = 0; i < biasIndices.size(); i++)
        fp += (*(data->fpPerBias))[i][biasIndices[i]];
    // Calculate F-measure by evaluating ((1 + b²) * TP) / (b² * NP + TP + FP),
    // where TP is the number of true positives, FP is the number of false positives
    // and NP is the total number of positive samples.
    return (data->b21 * tp) / (data->b2 * data->numPositive + tp + fp);
}
