#ifndef ARTOS_DEFS_H
#define ARTOS_DEFS_H

#include <vector>
#include <utility>
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
    
    Sample() = default;
    
    Sample(const Sample&) = default;
    
    Sample(Sample && other)
    : m_img(std::move(other.m_img)), m_simg(std::move(other.m_simg)), m_bboxes(std::move(other.m_bboxes)),
      modelAssoc(std::move(other.modelAssoc)), data(other.data)
    { other.data = NULL; };
    
    virtual ~Sample() { };
    
    virtual Sample & operator=(const Sample&) = default;
    
    virtual Sample & operator=(Sample && other)
    {
        this->m_img = std::move(other.m_img);
        this->m_simg = std::move(other.m_simg);
        this->m_bboxes = std::move(other.m_bboxes);
        this->modelAssoc = std::move(other.modelAssoc);
        this->data = other.data;
        other.data = NULL;
        return *this;
    }
    
    /**
    * @return The actual image, either retrieved from m_simg (preferred) or directly from m_img.
    */
    FFLD::JPEGImage img() const { return (this->m_simg.valid()) ? this->m_simg.getImage() : this->m_img; };
    
    /**
    * @return Vector of bounding boxes around objects on the image.
    */
    const std::vector<FFLD::Rectangle> & bboxes() const { return this->m_bboxes; };
};

}

#endif