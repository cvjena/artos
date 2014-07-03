#ifndef ARTOS_DEFS_H
#define ARTOS_DEFS_H

#include <vector>
#include "ffld/JPEGImage.h"
#include "ffld/Rectangle.h"

namespace ARTOS
{

typedef bool (*ProgressCallback)(unsigned int, unsigned int, void*);

/**
* Capsules information about a sample used for model learning and evaluation.
*/
struct Sample
{
    FFLD::JPEGImage img; /**< The entire image. */
    std::vector<FFLD::Rectangle> bboxes; /**< Vector of bounding boxes around objects on the image. */
    std::vector<unsigned int> modelAssoc; /**< Associates objects in bounding boxes with learned models. */
    virtual ~Sample() { };
};

}

#endif