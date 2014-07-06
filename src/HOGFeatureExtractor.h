#ifndef ARTOS_HOGFEATUREEXTRACTOR_H
#define ARTOS_HOGFEATUREEXTRACTOR_H

#include "ffld/JPEGImage.h"
#include "ffld/HOGPyramid.h"

namespace ARTOS
{

/**
* Computes histograms of oriented gradients (HOG) as image features by delegating
* the main computations to the FFLD library.
*
* The HOG features used for that implementation are described in "Object Detection
* with Discriminatively Trained Part Based Models" by Felzenszwalb, Girshick,
* McAllester and Ramanan, PAMI10.
*
* To use another feature extractor by default, re-define `FeatureExtractor` in
* FeatureExtractor.h.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class HOGFeatureExtractor
{

public:

    typedef FFLD::HOGPyramid::Scalar Scalar; /**< Scalar type of the components of a cell's feature vector. */
    typedef FFLD::HOGPyramid::Cell Cell; /**< Feature vector type of a single cell. */
    typedef FFLD::HOGPyramid::Level FeatureMatrix; /**< Array of cells which forms the feature vector of the entire image. */
    
    static const int numFeatures = FFLD::HOGPyramid::NbFeatures; /**< Number of features per cell. */
    static const int numRelevantFeatures = 31; /**< Number of relevant features per cell (for instance, the last dimension may be a truncation dimension and always set to 0). */
    static const int cellSize = 8; /**< Number of pixels of each cell in both x and y direction. */
    
    /**
    * Computes HOG features for a given image.
    *
    * @param[in] img The image to compute HOG features for.
    *
    * @param[out] feat Destination matrix to store the extracted features in.
    * It will be resized to fit the number of cells in the given image.
    *
    * @param[in] cs Optionally, the size of the feature cells if different from the default cellSize.
    */
    static void extract(const FFLD::JPEGImage & img, FeatureMatrix & feat, int cs = cellSize)
    {
        if (cs <= 0)
            cs = cellSize;
        FFLD::HOGPyramid::Level hog;
        FFLD::HOGPyramid::Hog(img, hog, 1, 1, cs);
        if (hog.rows() > 2 && hog.cols() > 2)
            feat = hog.block(1, 1, hog.rows() - 2, hog.cols() - 2); // cut off padding
        else
            feat = hog;
    };

};

}

#endif