#ifndef ARTOS_DEFS_H
#define ARTOS_DEFS_H

#include <vector>
#include <utility>
#include "JPEGImage.h"
#include "Rectangle.h"
#include "SynsetImage.h"

namespace ARTOS
{

typedef bool (*ProgressCallback)(unsigned int, unsigned int, void*);


/**
* A simple struct consisting of a width and a height.
*/
struct Size
{
    int width;
    int height;
    
    Size() : width(0), height(0) {}; /**< Constructs a Size with width and height 0. */
    explicit Size(int d) : width(d), height(d) {}; /**< Constructs a quadratic Size with width and height @p d. */
    explicit Size(int w, int h) : width(w), height(h) {}; /**< Constructs a Size with width @p w and height @p h. */
    
    /**
    * @return The size of the smaller dimension.
    */
    int min() const { return (this->width < this->height) ? this->width : this->height; };
    
    /**
    * @return The size of the larger dimension.
    */
    int max() const { return (this->width > this->height) ? this->width : this->height; };
    
    Size operator+(int s) const { return Size(this->width + s, this->height + s); };
    Size operator+(const Size & s) const { return Size(this->width + s.width, this->height + s.height); };
    Size operator-(int s) const { return Size(this->width - s, this->height - s); };
    Size operator-(const Size & s) const { return Size(this->width - s.width, this->height - s.height); };
    Size operator*(int s) const { return Size(this->width * s, this->height * s); };
    Size operator*(const Size & s) const { return Size(this->width * s.width, this->height * s.height); };
    Size operator/(int s) const { return Size(this->width / s, this->height / s); };
    Size operator/(const Size & s) const { return Size(this->width / s.width, this->height / s.height); };
    
    Size & operator+=(int s) { this->width += s; this->height += s; };
    Size & operator+=(const Size & s) { this->width += s.width; this->height += s.height; };
    Size & operator-=(int s) { this->width -= s; this->height -= s; };
    Size & operator-=(const Size & s) { this->width -= s.width; this->height -= s.height; };
    Size & operator*=(int s) { this->width *= s; this->height *= s; };
    Size & operator*=(const Size & s) { this->width *= s.width; this->height *= s.height; };
    Size & operator/=(int s) { this->width /= s; this->height /= s; };
    Size & operator/=(const Size & s) { this->width /= s.width; this->height /= s.height; };
    
    bool operator==(const Size & s) const { return (this->width == s.width && this->height == s.height); };
    bool operator!=(const Size & s) const { return !(*this == s); };
};


/**
* Capsules information about a sample used for model learning and evaluation.
*/
struct Sample
{
    JPEGImage m_img; /**< The entire image. */
    SynsetImage m_simg; /**< The image as SynsetImage object (as a preferred alternative to m_img). */
    std::vector<Rectangle> m_bboxes; /**< Vector of bounding boxes around objects on the image. */
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
    JPEGImage img() const { return (this->m_simg.valid()) ? this->m_simg.getImage() : this->m_img; };
    
    /**
    * @return Vector of bounding boxes around objects on the image.
    */
    const std::vector<Rectangle> & bboxes() const { return this->m_bboxes; };
};

}


namespace std
{

inline ARTOS::Size min(const ARTOS::Size & s1, const ARTOS::Size & s2)
{
    return ARTOS::Size((s1.width < s2.width) ? s1.width : s2.width, (s1.height < s2.height) ? s1.height : s2.height);
}

inline ARTOS::Size max(const ARTOS::Size & s1, const ARTOS::Size & s2)
{
    return ARTOS::Size((s1.width > s2.width) ? s1.width : s2.width, (s1.height > s2.height) ? s1.height : s2.height);
}

}

#endif