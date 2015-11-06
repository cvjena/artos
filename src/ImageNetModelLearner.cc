#include "ImageNetModelLearner.h"

#include <vector>
#include <utility>
#include <iostream>

#include "ffld/JPEGImage.h"
#include "ffld/timingtools.h"

using namespace ARTOS;
using namespace FFLD;
using namespace std;


void ImageNetModelLearner::reset()
{
    ModelLearner::reset();
    this->m_addedSynsets.clear();
}

unsigned int ImageNetModelLearner::addPositiveSamplesFromSynset(const string & synsetId, const unsigned int maxSamples)
{
    return this->addPositiveSamplesFromSynset(this->m_repo.getSynset(synsetId), maxSamples);
}

unsigned int ImageNetModelLearner::addPositiveSamplesFromSynset(const Synset & synset, const unsigned int maxSamples)
{
    if (synset.id.empty())
        return 0;
    this->m_addedSynsets.insert(synset.id);

    // Fetch samples from synset
    if (this->m_verbose)
        start();
    unsigned int numSamples = 0, numBBoxes;
    for (SynsetImageIterator imgIt = synset.getImageIterator(true); imgIt.ready() && (maxSamples == 0 || (unsigned int) imgIt < maxSamples); ++imgIt)
    {
        SynsetImage simg = *imgIt;
        if (simg.loadBoundingBoxes())
        {
            numBBoxes = simg.bboxes.size();
#ifndef NO_CACHE_POSITIVES
            if (!simg.getImage().empty() && this->addPositiveSample(move(simg)))
                numSamples += numBBoxes;
#else
            if (this->addPositiveSample(move(simg)))
                numSamples += numBBoxes;
#endif
        }
    }
    if (this->m_verbose)
        cerr << "Fetched " << this->getNumSamples() << " samples from synset in " << stop() << " ms." << endl;
    
    return numSamples;
}

const vector<float> & ImageNetModelLearner::optimizeThreshold(const unsigned int maxPositive, const unsigned int numNegative,
                                                              const float b, ProgressCallback progressCB, void * cbData)
{
    vector<JPEGImage> * negative = NULL;
    if (numNegative > 0 && this->m_repo.getNumSynsets() > this->m_addedSynsets.size())
    {
        negative = new vector<JPEGImage>();
        for (MixedImageIterator imgIt = this->m_repo.getMixedIterator(); imgIt.ready() && negative->size() < numNegative; ++imgIt)
        {
            SynsetImage img = *imgIt;
            if (this->m_addedSynsets.find(img.getSynsetId()) == this->m_addedSynsets.end())
                negative->push_back(img.getImage());
        }
    }
    const vector<float> & th = ModelLearner::optimizeThreshold(maxPositive, negative, b, progressCB, cbData);
    if (negative != NULL)
        delete negative;
    return th;
}
