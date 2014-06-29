#ifndef ARTOS_FEATUREEXTRACTOR_H
#define ARTOS_FEATUREEXTRACTOR_H

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

    typedef FFLD::HOGPyramid::Scalar Scalar; /**< Scalar type of the components of a cell's feature vector. */
    typedef FFLD::HOGPyramid::Cell Cell; /**< Feature vector type of a single cell. */
    typedef FFLD::HOGPyramid::Level FeatureMatrix; /**< Array of cells which forms the feature vector of the entire image. */
    
    static const int numFeatures = HOGPyramid::NbFeatures; /**< Number of features per cell. */
    static const int cellSize = 8; /**< Number of pixels of each cell in both x and y direction. */
    
    /**
    * Computes HOG features for a given image.
    *
    * @param[in] img The image to compute HOG features for.
    *
    * @param[out] feat Destination matrix to store the extracted features in.
    * It will be resized to fit the number of cells in the given image.
    */
    static void extract(const FFLD::JPEGImage & img, FeatureMatrix & feat)
    {
        FFLD::HOGPyramid::Hog(img, feat, 1, 1, cellSize);
        if (feat.rows() > 2 && feat.cols() > 2)
            feat = feat.block(1, 1, feat.rows() - 2, feat.cols() - 2); // cut off padding
    };

};

}

#endif