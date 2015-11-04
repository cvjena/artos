#ifndef ARTOS_DEFS_H
#define ARTOS_DEFS_H

#include <vector>
#include "ffld/JPEGImage.h"
#include "ffld/Rectangle.h"
#include "SynsetImage.h"

namespace ARTOS
{

typedef bool (*ProgressCallback)(unsigned int, unsigned int, void*);

/**
* Capsules information about a sample used for model learning and evaluation.
*/
struct Sample
{
    FFLD::JPEGImage m_img; /**< The entire image. */
    SynsetImage m_simg; /**< The image as SynsetImage object (as a preferred alternative to m_img). */
    std::vector<FFLD::Rectangle> m_bboxes; /**< Vector of bounding boxes around objects on the image. */
    std::vector<unsigned int> modelAssoc; /**< Associates objects in bounding boxes with learned models. */
    void * data; /**< Arbitrary pointer to custom data associated with the sample. */
    virtual ~Sample() { };
    
    FFLD::JPEGImage img() const { return (this->m_simg.valid()) ? this->m_simg.getImage() : this->m_img; };
    const std::vector<FFLD::Rectangle> & bboxes() const { return this->m_bboxes; };
};

}

#endif