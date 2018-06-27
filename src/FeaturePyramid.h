#ifndef ARTOS_FEATUREPYRAMID_H
#define ARTOS_FEATUREPYRAMID_H

#include "FeatureExtractor.h"

namespace ARTOS
{

/**
* A pyramid of features extracted from multiple scales of an image using the default FeatureExtractor.
*
* The scale of the pyramid level of index @c i is given by the following formula:
* 2^(1 - @c i / @c interval), so that the first scale is at double the resolution of the original image.
* Some scales may be omitted due to restrictions of the feature extractor.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class FeaturePyramid
{

public:

    /**
    * Constructs an empty pyramid. An empty pyramid has no level.
    */
    FeaturePyramid() : m_interval(0), m_levels(), m_scales(), m_featureExtractor(FeatureExtractor::defaultFeatureExtractor()) {};
    
    /**
    * Constructs a pyramid from parameters and a list of levels.
    * @param[in] interval Number of levels per octave in the pyramid.
    * @param[in] levels List of pyramid levels (at least 1).
    * @param[in] scales The scales belonging to the levels. If not given, this pyramid
    * will contain no information about scales.
    */
    FeaturePyramid(int interval, const std::vector<FeatureMatrix> & levels, const std::vector<double> * scales = NULL);
    
    /**
    * Constructs a pyramid from parameters and a list of levels.
    * @param[in] interval Number of levels per octave in the pyramid.
    * @param[in] levels List of pyramid levels (at least 1) to be moved.
    * @param[in] scales The scales belonging to the levels. If not given, this pyramid
    * will contain no information about scales.
    */
    FeaturePyramid(int interval, std::vector<FeatureMatrix> && levels, const std::vector<double> * scales = NULL);
    
    /**
    * Constructs a pyramid from a JPEGImage.
    * @param[in] image The JPEGImage.
    * @param[in] featureExtractor The feature extractor to be used by this pyramid.
    * @param[in] interval Number of levels per octave in the pyramid (at least 1).
    * @param[in] minSize Minimum number of cells in x or y direction in the smallest scale in the pyramid.
    */
    FeaturePyramid(const JPEGImage & image, const std::shared_ptr<FeatureExtractor> & featureExtractor = nullptr, int interval = 10, unsigned int minSize = 5);
    
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
    */
    std::vector<FeatureMatrix> & levels() { return this->m_levels; };
    
    /**
    * @return A const reference to the pyramid levels.
    */
    const std::vector<FeatureMatrix> & levels() const { return this->m_levels; };
    
    /**
    * @return A reference to a vector with the scales of each level in the pyramid.
    */
    std::vector<double> & scales() { return this->m_scales; };
    
    /**
    * @return A const reference to a vector with the scales of each level in the pyramid.
    */
    const std::vector<double> & scales() const { return this->m_scales; };
    
    /**
    * @return A shared pointer to the feature extractor used by this feature pyramid.
    */
    std::shared_ptr<FeatureExtractor> featureExtractor() { return this->m_featureExtractor; };
    
    /**
    * @return A const shared pointer to the feature extractor used by this feature pyramid.
    */
    std::shared_ptr<const FeatureExtractor> featureExtractor() const { return this->m_featureExtractor; };

    /**
    * Replaces the contents of this feature pyramid with data read from a binary file.
    * @param[in] filename Path of the file to deserialize the feature pyramid from.
    * @return True if the file could be read successfully, false if it is inaccessible or invalid.
    * empty() may also be used to check the status of this operation after it has finished.
    */
    bool readFromFile(const std::string & filename);
    
    /**
    * Writes the this feature pyramid to a binary file.
    * @param[in] filename Path to write the file to.
    * @return Returns true if this pyramid is not empty and the file could be written successfully, otherwise false.
    */
    bool writeToFile(const std::string & filename);

    /**
    * @return The size of the serialized representation of this feature pyramid (obtained from operator<<) in bytes.
    */
    size_t serializedSize() const;

protected:

    int m_interval;
    std::vector<FeatureMatrix> m_levels;
    std::vector<double> m_scales;
    std::shared_ptr<FeatureExtractor> m_featureExtractor;
    
    /**
    * Constructs `m_levels` according to `m_scales` using `m_featureExtractor`.
    * @param[in] img The image to extract features from.
    */
    void buildLevels(const JPEGImage & img);
    
    /**
    * Constructs `m_levels` according to `m_scales` by placing multiple scales of the image
    * together on a plane of fixed size in order to reduce the number of calls to `m_featureExtractor->extract()`.
    * `m_featureExtractor->borderSize()` will be used as padding between images on the same plane.
    * @param[in] img The image to extract features from.
    */
    void buildLevelsPatchworked(const JPEGImage & img);

};


/**
* Serializes a feature pyramid to a binary stream.
* 
* The serial format is as follows:
* It begins with a 32-bit unsigned integer `l` specifying the number of levels in the pyramid,
* a 32-bit unsigned integer `i` specifying the number of intervals per octave,
* and an 8-bit unsigned integer specifying the number of bits used for the floating point format.
* 
* `l` serial representations of levels follow. Each one begins with a 64-bit floating point
* number specifying the scale of the level and three 32-bit integers `n`, `m`, `c`, giving
* the number of rows, columns, and channels in the feature matrix, respectively.
* Then follows the actual data of the feature matrix as a sequence of `n*m*c` floating point numbers.
* 
* No conversion of endianess is performed.
* 
* @param[in] os The stream where the serialization of the feature pyramid should be written.
* @param[in] pyramid The feature pyramid to be serialized.
*/
std::ostream & operator<<(std::ostream & os, const FeaturePyramid & pyramid);

/**
* Unserializes a feature pyramid from a binary stream given in the format described in operator<<.
* @param[in] is The stream containing the serialization of the feature pyramid.
* @param[out] pyramid The FeaturePyramid object which the stream should be unserialized into.
*/
std::istream & operator>>(std::istream & is, FeaturePyramid & pyramid);

}

#endif