/**
* @file
* Defines the default feature extractor (usually HOGFeatureExtractor).
* If you want to use your own feature extractor by default, provide a class with at least the
* same interface as HOGFeatureExtractor, include it here and re-define `FeatureExtractor` in this file.
*/

#ifndef ARTOS_FEATUREEXTRACTOR_H
#define ARTOS_FEATUREEXTRACTOR_H

#include "HOGFeatureExtractor.h"

namespace ARTOS
{
    typedef HOGFeatureExtractor FeatureExtractor;
}

#endif