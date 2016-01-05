#ifndef ARTOS_HOGFEATUREEXTRACTOR_H
#define ARTOS_HOGFEATUREEXTRACTOR_H

#include "FeatureExtractor.h"

namespace ARTOS
{

/**
* Computes histograms of oriented gradients (HOG) as image features based on the
* implementation of the FFLD library.
*
* The HOG features used for that implementation are described in "Object Detection
* with Discriminatively Trained Part Based Models" by Felzenszwalb, Girshick,
* McAllester and Ramanan, PAMI10.
*
* **Parameters of this feature extractor:**
*     - *cellSizeX* (`int`) - size of pooling cells in x direction in pixels.
*     - *cellSizeY* (`int`) - size of pooling cells in y direction in pixels.
*
* To use another feature extractor by default, re-define `DefaultFeatureExtractor` in
* FeatureExtractor.h.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class HOGFeatureExtractor : public FeatureExtractor
{

public:

    /**
    * Constructs a HOG feature extractor with a default cell size of 8 pixels.
    */
    HOGFeatureExtractor();
    
    /**
    * Constructs a HOG feature extractor with a specific cell size.
    *
    * @param[in] cellSize The size of pooling cells in each direction in pixels.
    */
    HOGFeatureExtractor(const Size & cellSize);

    /**
    * @return Returns the unique identifier of this kind of feature extractor. That type specifier
    * must consist of alphanumeric characters + dashes + underscores only.
    */
    virtual const char * type() const override { return "HOG"; };
    
    /**
    * @return Human-readable name of this feature extractor.
    */
    virtual const char * name() const override;
    
    /**
    * @return Returns the number of features this feature extractor extracts from each cell.
    */
    virtual int numFeatures() const override;
    
    /**
    * @return Returns the number of relevant features per cell (for instance, the last dimension
    * may be a truncation dimension and always set to 0 and, thus, not relevant).
    */
    virtual int numRelevantFeatures() const override;
    
    /**
    * @return Returns the size of the cells used by this feature extractor in x and y direction.
    */
    virtual Size cellSize() const override;
    
    /**
    * @return Returns true if this feature extractors implements the extract() method with explicit
    * cell size specification which differs from the default cell size reported by cellSize().
    * If this method returns true, the 3-parameter version of extract() should not throw an UnsupportedException.
    */
    virtual bool supportsVariableCellSize() const override;
    
    /**
    * Converts a width and height given in pixels to cells.
    *
    * @param[in] pixels The size given in pixels.
    *
    * @return Returns the corresponding size in cells.
    */
    virtual Size pixelsToCells(const Size & pixels) const override;
    
    /**
    * Computes HOG features for a given image.
    *
    * @param[in] img The image to compute HOG features for.
    *
    * @param[out] feat Destination matrix to store the extracted features in.
    * It will be resized to fit the number of cells in the given image.
    */
    virtual void extract(const JPEGImage & img, FeatureMatrix & feat) const override
    { this->extract(img, feat, this->cellSize()); };
    
    /**
    * Computes HOG features for a given image using a non-default cell size.
    *
    * This may be useful for oversampling, where it is more efficient to divide the cell size
    * by two instead of doubling the size of the entire image. But an arbitrary feature extractor
    * does not need to support this. In that case, a NotSupported exception is thrown.
    *
    * @param[in] img The image to compute HOG features for.
    *
    * @param[out] feat Destination matrix to store the extracted features in.
    * It will be resized to fit the number of cells in the given image.
    *
    * @param[in] cellSize The size of the feature cells.
    */
    virtual void extract(const JPEGImage & img, FeatureMatrix & feat, const Size & cellSize) const override;
    
    /**
    * Transforms a feature matrix into a feature representation of the horizontally flipped image.
    *
    * For HOG features, there is an easy way to compute the feature representation of the horizontally
    * flipped version of an image given the features of that original image, so that it is not necessary
    * to actually flip the image and compute its features from scratch. This is what this method does.
    *
    * @param[in] feat The features extracted from the original image using this feature extractor.
    *
    * @param[out] flipped Destination matrix to store the features of the horizontally flipped image in.
    * It will be resized to the same size as feat.
    */
    virtual void flip(const FeatureMatrix & feat, FeatureMatrix & flipped) const override;
    
    /**
    * Changes the value of an integer parameter specific to the concrete feature extraction method.
    *
    * @param[in] val The new value for the parameter.
    *
    * @throws UnknownParameterException There is no string parameter with the given name.
    *
    * @throws std::invalid_argument The given value is not allowed for the given parameter.
    */
    virtual void setParam(const std::string & paramName, int32_t val) override;
    
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
    virtual Size computeOptimalModelSize(const std::vector<Size> & sizes, const Size & maxSize = Size()) const override;


public:

    /**
    * Extracts HOG features from a given image and stores them in a FeatureMatrix.
    *
    * The implementation of this method is mainly based on the HOG implementation of FFLD.
    *
    * @param[in] image The image to extract HOG features from.
    *
    * @param[out] feat The FeatureMatrix where the HOG features will be stored. The matrix will
    * be resized to `(ceil(image.width() / cellSize) + 2 * padding.width, ceil(image.height() / cellSize) + 2 * padding.height)`.
    *
    * @param[in] padding Number of empty cells to pad the feature matrix with around the borders in
    * each direction before interpolation of gradient orientations among neighbouring cells. Must be at least (1, 1).
    *
    * @param[in] cellSize The number of pixels in each direction per histogram cell.
    * Each dimension must be a multiple of 2.
    */
    static void HOG(const JPEGImage & image, FeatureMatrix & feat, const Size & padding, const Size & cellSize);


private:

    Size m_cellSize;

};

}

#endif