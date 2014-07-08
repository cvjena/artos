/**
* @file
* Defines the default feature extractor (usually HOGFeatureExtractor) and a feature pyramid class based on that extractor.

* If you want to use your own feature extractor by default, provide a class with at least the
* same interface as HOGFeatureExtractor, include it here and re-define `FeatureExtractor` in this file.
* Such a custom feature extractor will then be used for model learning and estimation of background statistics,
* but not for the FFLD-based object detection, which always uses HOG features.
*/

#ifndef ARTOS_FEATUREEXTRACTOR_H
#define ARTOS_FEATUREEXTRACTOR_H

#include <vector>
#include "ffld/JPEGImage.h"
#include "HOGFeatureExtractor.h"

namespace ARTOS
{

typedef HOGFeatureExtractor FeatureExtractor;


/**
* A pyramid of features extracted from multiple scales of an image using the default FeatureExtractor.
*
* The scale of the pyramid level of index @c i is given by the following formula:
* 2^(1 - @c i / @c interval), so that the first scale is at double the resolution of the original image).
*
* This class has been derived from FFLD's HOGPyramid.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class FeaturePyramid
{

public:

    /**
    * Constructs an empty pyramid. An empty pyramid has no level.
    */
    FeaturePyramid() : m_interval(0), m_levels() {};
    
    /**
    * Constructs a pyramid from parameters and a list of levels.
    * @param[in] interval Number of levels per octave in the pyramid.
    * @param[in] levels List of pyramid levels (at least 1).
    */
    FeaturePyramid(int interval, const std::vector<FeatureExtractor::FeatureMatrix> & levels);
    
    /**
    * Constructs a pyramid from a JPEGImage.
    * @param[in] image The JPEGImage.
    * @param[in] interval Number of levels per octave in the pyramid (at least 1).
    */
    FeaturePyramid(const FFLD::JPEGImage & image, int interval = 10);
    
    /**
    * @return True if the pyramid is empty. An empty pyramid has no level.
    */
    bool empty() const { return this->m_levels.empty(); };
    
    /**
    * @return Number of levels per octave in the pyramid.
    */
    int interval() const { return this->m_interval; };
    
    /**
    * @return A reference to the pyramid levels.
    * @note Scales are given by the following formula: 2^(1 - @c index / @c interval).
    */
    std::vector<FeatureExtractor::FeatureMatrix> & levels() { return this->m_levels; };
    
    /**
    * @return A const reference to the pyramid levels.
    * @note Scales are given by the following formula: 2^(1 - @c index / @c interval).
    */
    const std::vector<FeatureExtractor::FeatureMatrix> & levels() const { return this->m_levels; };

protected:

    int m_interval;
    std::vector<FeatureExtractor::FeatureMatrix> m_levels;

};

}

#endif