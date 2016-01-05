#ifndef ARTOS_FEATUREEXTRACTOR_H
#define ARTOS_FEATUREEXTRACTOR_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <istream>
#include <ostream>
#include <cstdint>
#include <Eigen/Core>
#include "defs.h"
#include "exceptions.h"
#include "FeatureMatrix.h"
#include "JPEGImage.h"

namespace ARTOS
{

/**
* @brief Abstract base class for feature extractors.
*
* The purpose of classes inheriting from this one is to transform images into some kind of features, HOG for instance.  
* They may optionally divide the image into cells of a fixed size and extract feature vectors of same length
* from each cell in order to preserve spatial information. Those feature vectors of each cell will then be
* concatenated to form the final feature vector of the entire image.
*
* This class also provides static methods for enumerating and creating feature extractors as well as changing the
* default feature extractor used throughout ARTOS.
*
* If you want to implement your own feature extractor, derive it from this class and add it to the featureExtractors
* map in FeatureExtractor.cc.  
* Any parameters supported by a feature extractor should be added to the m_intParams, m_scalarParams or m_stringParams
* map with their default values then the feature extractor is constructed.
*
* Some feature extractors may require that some of their parameters are set first before they may be used. If those
* parameters have not been set yet, any method of that feature extractor may throw a UseBeforeSetupException.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class FeatureExtractor
{

public:

    enum class ParameterType : uint8_t { INT, SCALAR, STRING };
    
    /**
    * Contains information about a parameter of a feature extraction algorithm.
    */
    struct ParameterInfo
    {
        std::string name;       /**< The name of the parameter. */
        ParameterType type;     /**< The type of the parameter. */
        union
        {
            int32_t intValue;           /**< The current value of the parameter if it is of type `INT`. */
            FeatureScalar scalarValue;  /**< The current value of the parameter if it is of type `SCALAR`. */
            const char * stringValue;   /**< The current value of the parameter if it is of type `STRING`. */
        };
        
        ParameterInfo(const std::string & name_, int32_t value)
        : name(name_), type(ParameterType::INT), intValue(value) {};
        
        ParameterInfo(const std::string & name_, FeatureScalar value)
        : name(name_), type(ParameterType::SCALAR), scalarValue(value) {};
        
        ParameterInfo(const std::string & name_, const std::string & value)
        : name(name_), type(ParameterType::STRING), stringValue(value.c_str()) {};
    };


public:

    virtual ~FeatureExtractor() {};

    /**
    * @return Returns the unique identifier of this kind of feature extractor. That type specifier
    * must consist of alphanumeric characters + dashes + underscores only and must begin with a letter.
    */
    virtual const char * type() const =0;
    
    /**
    * @return Human-readable name of this feature extractor.
    */
    virtual const char * name() const =0;

    /**
    * @return Returns the number of features this feature extractor extracts from each cell.
    */
    virtual int numFeatures() const =0;
    
    /**
    * @return Returns the number of relevant features per cell (for instance, the last dimension
    * may be a truncation dimension and always set to 0 and, thus, not relevant).
    */
    virtual int numRelevantFeatures() const { return this->numFeatures(); };
    
    /**
    * @return Returns the size of the cells used by this feature extractor in x and y direction.
    */
    virtual Size cellSize() const =0;
    
    /**
    * Specifies the size of the border along the x and y dimension of an image, which gets lost
    * during feature extraction. This may be due to unpadded convolutions, for instance.  
    * For example, a border of size (4, 2) would indicate, that only the region between (4, 2) and
    * (width - 5, height - 3) of an image would be transformed into features.
    *
    * @return Returns the size of the border along each image dimension in pixels.
    */
    virtual Size borderSize() const { return Size(); };
    
    /**
    * @return Returns a Size struct with the maximum sizes for image in x and y direction which
    * can be processed by this feature extractor. If any dimension is 0, the size of the image
    * along that dimension does not need to be limited.
    */
    virtual Size maxImageSize() const { return Size(); };
    
    /**
    * @return Returns true if this feature extractors implements the extract() method with explicit
    * cell size specification which differs from the default cell size reported by cellSize().
    * If this method returns true, the 3-parameter version of extract() should not throw an UnsupportedException.
    */
    virtual bool supportsVariableCellSize() const { return false; };
    
    /**
    * @return Returns true if it is safe to call extract() in parallel from multiple threads.
    */
    virtual bool supportsMultiThread() const { return true; };
    
    /**
    * Specifies if it is considered reasonable to process feature extraction of multiple
    * scales of an image by patchworking them together, so that multiple scales are processed at
    * once on a single plane, which will have the size of the largest scale.
    * The patchworkPadding() reported by the feature extractor will be used as padding between images
    * on the same plane.
    *
    * There are only very few cases there patchwork feature extraction is actually beneficial.
    * It should not be used careless, since the black padding between the images on a patchwork plane
    * may lead to artifacts.
    *
    * @return Returns true if this feature extractor should be applied to patchworks for processing
    * of multiple scales of an image.
    */
    virtual bool patchworkProcessing() const { return false; };
    
    /**
    * Specifies the amount of padding which should be added between images on the same plane when
    * features are extracted using patchworking (see patchworkProcessing()).
    *
    * @return Returns the amount of padding in pixels to add between two images.
    */
    virtual Size patchworkPadding() const { return Size(); };

    /**
    * Converts a width and height given in cells to pixels.
    *
    * @param[in] cells The size given in cells.
    *
    * @return Returns the corresponding size in pixels.
    */
    virtual Size cellsToPixels(const Size & cells) const;
    
    /**
    * Converts a width and height given in pixels to cells.
    *
    * @param[in] pixels The size given in pixels.
    *
    * @return Returns the corresponding size in cells.
    */
    virtual Size pixelsToCells(const Size & pixels) const;

    /**
    * Converts coordinates given in cells to pixel coordinates.
    *
    * @param[in] cells The coordinates given in cells.
    *
    * @return Returns the corresponding pixel coordinates.
    */
    virtual Size cellCoordsToPixels(const Size & cells) const;
    
    /**
    * Converts coordinates given in pixels to cell coordinates.
    *
    * @param[in] pixels The coordinates given in pixels.
    *
    * @return Returns the corresponding cell coordinates.
    */
    virtual Size pixelCoordsToCells(const Size & pixels) const;
    
    /**
    * Computes features for a given image.
    *
    * @param[in] img The image to compute features for.
    *
    * @param[out] feat Destination matrix to store the extracted features in.
    * It will be resized to fit the number of cells in the given image.
    *
    * @note If the implementation of this function in the derived class is not thread-safe,
    * you must override supportsMultiThread().
    */
    virtual void extract(const JPEGImage & img, FeatureMatrix & feat) const =0;
    
    /**
    * Computes features for a given image using a non-default cell size.
    *
    * This may be useful for oversampling, where it is more efficient to divide the cell size
    * by two instead of doubling the size of the entire image. But an arbitrary feature extractor
    * does not need to support this. In that case, a NotSupported exception is thrown.
    *
    * A derived feature extractor implementing this method must also override supportsVariableCellSize()
    * to return `true`.
    *
    * @param[in] img The image to compute features for.
    *
    * @param[out] feat Destination matrix to store the extracted features in.
    * It will be resized to fit the number of cells in the given image.
    *
    * @param[in] cellSize The size of the feature cells.
    *
    * @throws NotSupportedException This feature extractor does not support variable cell sizes.
    *
    * @note If the implementation of this function in the derived class is not thread-safe,
    * you must override supportsMultiThread().
    */
    virtual void extract(const JPEGImage & img, FeatureMatrix & feat, const Size & cellSize) const
    { throw NotSupportedException("This feature extractor does not support variable cell sizes."); };
    
    /**
    * Transforms a feature matrix into a feature representation of the horizontally flipped image.
    *
    * Sometimes there is an easy way to compute the feature representation of the horizontally flipped
    * version of an image given the features of that original image, so that it is not necessary to
    * actually flip the image and compute its features from scratch. This is what this method does,
    * though an arbitrary feature extractor does not need to support it.
    *
    * @param[in] feat The features extracted from the original image using this feature extractor.
    *
    * @param[out] flipped Destination matrix to store the features of the horizontally flipped image in.
    * It will be resized to the same size as feat.
    *
    * @throws NotSupportedException This feature extractor does not support feature flipping.
    */
    virtual void flip(const FeatureMatrix & feat, FeatureMatrix & flipped) const
    { throw NotSupportedException("This feature extractor does not support flipping of feature matrices."); };

    /**
    * Retrieves the value of an integer parameter specific to the concrete feature extraction method.
    *
    * @param[in] paramName The name of the parameter to be retrieved.
    *
    * @return Returns the value of the given integer parameter.
    *
    * @throws UnknownParameterException There is no integer parameter with the given name.
    */
    virtual int32_t getIntParam(const std::string & paramName) const;

    /**
    * Retrieves the value of a parameter of type FeatureScalar specific to the concrete feature extraction method.
    *
    * @param[in] paramName The name of the parameter to be retrieved.
    *
    * @return Returns the value of the given FeatureScalar parameter.
    *
    * @throws UnknownParameterException There is no parameter with the given type and name.
    */
    virtual FeatureScalar getScalarParam(const std::string & paramName) const;

    /**
    * Retrieves the value of a string parameter specific to the concrete feature extraction method.
    *
    * @param[in] paramName The name of the parameter to be retrieved.
    *
    * @return Returns the value of the given string parameter.
    *
    * @throws UnknownParameterException There is no string parameter with the given name.
    */
    virtual std::string getStringParam(const std::string & paramName) const;
    
    /**
    * Changes the value of an integer parameter specific to the concrete feature extraction method.
    *
    * @param[in] val The new value for the parameter.
    *
    * @throws UnknownParameterException There is no string parameter with the given name.
    *
    * @throws std::invalid_argument The given value is not allowed for the given parameter.
    */
    virtual void setParam(const std::string & paramName, int32_t val);
    
    /**
    * Changes the value of a parameter of type FeatureScalar specific to the concrete feature extraction method.
    *
    * @param[in] val The new value for the parameter.
    *
    * @throws UnknownParameterException There is no parameter with type FeatureScalar and the given name.
    *
    * @throws std::invalid_argument The given value is not allowed for the given parameter.
    */
    virtual void setParam(const std::string & paramName, FeatureScalar val);
    
    /**
    * Changes the value of a string parameter specific to this algorithm.
    *
    * @param[in] val The new value for the parameter.
    *
    * @throws UnknownParameterException There is no string parameter with the given name.
    *
    * @throws std::invalid_argument The given value is not allowed for the given parameter.
    */
    virtual void setParam(const std::string & paramName, const std::string & val);
    
    /**
    * Enumerates all parameters known by this feature extractor along with their respective values.
    *
    * @param[out] params A vector which will be cleared and then filled with ParameterInfo structs
    * containing information about the parameters known by this feature extractor.
    */
    virtual void listParameters(std::vector<ParameterInfo> & params);

    /**
    * Proposes an optimal size of a model for images with given sizes.
    * This information will be used by the ModelLearner.
    *
    * @param[in] sizes A vector with the dimensions of the images.
    *
    * @param[in] maxSize Optionally, the highest allowable extensions of the model size in each dimension.
    * This function is guaranteed to return a model size within this bounds. Setting any dimension to 0
    * does not limit the computed size in that dimension.
    *
    * @return Returns a proposal for the size of a model for images with the given sizes.
    * The size must be specified in cells.
    */
    virtual Size computeOptimalModelSize(const std::vector<Size> & sizes, const Size & maxSize = Size()) const;
    
    /**
    * Compares this feature extractor to another one for equality.
    *
    * Two feature extractors compare equal if their type and all their parameters are equal.
    *
    * @param[in] other The right-hand side of the comparison.
    *
    * @return Returns true if this feature extractor compares equal to `other`, otherwise false.
    */
    bool operator==(const FeatureExtractor & other);
    
    /**
    * Compares this feature extractor to another one for inequality.
    *
    * Two feature extractors compare equal if their type and all their parameters are equal.
    *
    * @param[in] other The right-hand side of the comparison.
    *
    * @return Returns false if this feature extractor compares equal to `other`, otherwise true.
    */
    bool operator!=(const FeatureExtractor & other) { return !(*this == other); };


public:

    /**
    * Finds the most common aspect ratio among a list of image dimensions.
    *
    * @param[in] sizes A vector with the dimensions of the images.
    *
    * @param[out] areas Optionally, a pointer to a vector of integers which will be cleared and filled
    * with the area of each image (i.e. width * height).
    *
    * @return Returns the most common aspect ratio (height / width) among the given sizes.
    */
    static float commonAspectRatio(const std::vector<Size> & sizes, std::vector<int> * areas = NULL);
    
    /**
    * Creates a feature extractor identified by its type specifier. The feature extractor
    * will be default-constructed.
    *
    * @param[in] type The type specifier of the feature extractor to create.
    *
    * @return Returns a shared pointer to the new feature extractor.
    *
    * @throws UnknownFeatureExtractorException There is no feature extractor with the given type.
    */
    static std::shared_ptr<FeatureExtractor> create(const std::string & type);

    /**
    * @return Returns a shared pointer to the default feature extractor, which will be created if it hasn't been already.
    */
    static std::shared_ptr<FeatureExtractor> defaultFeatureExtractor();
    
    /**
    * Changes the default feature extractor returned by defaultFeatureExtractor().
    *
    * @param[in] newDefault Shared pointer to the new default feature extractor.
    */
    static void setDefaultFeatureExtractor(const std::shared_ptr<FeatureExtractor> & newDefault);
    
    /**
    * Changes the default feature extractor returned by defaultFeatureExtractor().
    *
    * @param[in] type The type specifier of the new default feature extractor.
    *
    * @throws UnknownFeatureExtractorException There is no feature extractor with the given type.
    */
    static void setDefaultFeatureExtractor(const std::string & type);
    
    /**
    * @return Returns the number of available feature extractor implementations.
    */
    static int numFeatureExtractors();
    
    /**
    * Lists the type specifiers of all available feature extractors.
    *
    * @param[out] featureExtractors A vector which will be cleared and then filled with the type specifiers
    * (corresponding to `type()`) of all available feature extractors.
    */
    static void listFeatureExtractors(std::vector<std::string> & featureExtractors);
    
    /**
    * Lists all available feature extractors.
    *
    * @param[out] featureExtractors A vector which will be cleared and then filled with shared pointers
    * to default-constructed instances of all available feature extractors.
    */
    static void listFeatureExtractors(std::vector< std::shared_ptr<FeatureExtractor> > & featureExtractors);


protected:

    std::map<std::string, int32_t> m_intParams;
    std::map<std::string, FeatureScalar> m_scalarParams;
    std::map<std::string, std::string> m_stringParams;

    friend std::ostream & operator<<(std::ostream & os, const FeatureExtractor & featureExtractor);
    
    friend std::istream & operator>>(std::istream & is, FeatureExtractor & featureExtractor);


private:

    typedef std::shared_ptr<FeatureExtractor> (*featureExtractorFactory)();

    static std::shared_ptr<FeatureExtractor> dfltFeatureExtractor;
    
    static std::map<std::string, featureExtractorFactory> featureExtractorFactories;

};


/**
* Serializes the parameters of a FeatureExtractor to a stream as key-value pairs.
*
* @param[in] os The output stream.
*
* @param[in] featureExtractor The feature extractor to be serialized.
*
* @return Reference to os.
*/
std::ostream & operator<<(std::ostream & os, const FeatureExtractor & featureExtractor);

/**
* Deserializes the parameters of a FeatureExtractor from a stream, where they are expected
* as a space-separated sequence of names and values, all on a single line.
*
* @param[in] is The input stream.
*
* @param[in] featureExtractor The feature extractor which will be configured with the
* deserialized parameters.
*
* @return Reference to is.
*
* @throws DeserializationException Data on the stream is in an unrecognized format.
*
* @throws UnknownParameterException One or more parameters listed on the stream are not known
* by the given feature extractor.
*
* @throws std::invalid_argument A value found on the stream is not allowed for the corresponding parameter.
*/
std::istream & operator>>(std::istream & is, FeatureExtractor & featureExtractor);

}

#endif