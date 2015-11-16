//--------------------------------------------------------------------------------------------------
// Implementation of the paper "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012.
//
// Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLD (the Fast Fourier Linear Detector)
//
// FFLD is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
//
// FFLD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with FFLD. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#ifndef ARTOS_JPEGIMAGE_H
#define ARTOS_JPEGIMAGE_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "FeatureMatrix.h"

namespace ARTOS
{

/**
* The JPEGImage class allows to load/save an image from/to a jpeg file, and to apply basic
* operations such as resizing or cropping to it. The pixels are stored contiguously in row-major
* order (scanline 0: RGB RGB RGB, scanline 1: RGB RGB RGB...)
*/
class JPEGImage
{

public:

    /**
    * Constructs an empty image. An empty image has zero size.
    */
    JPEGImage();
    
    /**
    * Constructs an image with the given @p width, @p height and @p depth, and initializes it from
    * the given @p bits.
    * @note The returned image might be empty if any of the parameters is incorrect.
    */
    JPEGImage(int width, int height, int depth, const uint8_t * bits = 0);
    
    /**
    * Constructs an image and tries to load the image from the jpeg file with the given
    * @p filename.
    * @note The returned image might be empty if the image could not be loaded.
    */
    JPEGImage(const std::string & filename);
    
    /**
    * Constructs an image and tries to load the image from the given, already opened file stream
    * @p filehandle Handle to the file stream opened using `fopen()`.
    * @note The returned image might be empty if the image could not be loaded.
    */
    JPEGImage(FILE * filehandle);
    
    /**
    * Copies image data from another JPEGImage object.
    * @p other The JPEGImage whose data is to be copied.
    */
    JPEGImage(const JPEGImage & other) = default;
    
    /**
    * Moves image data from another JPEGImage object and leaves that empty.
    * @p other The JPEGImage whose data is to be moved to this one.
    */
    JPEGImage(JPEGImage && other);
    
    /**
    * Copies image data from another JPEGImage object.
    * @p other The JPEGImage whose data is to be copied.
    */
    JPEGImage & operator=(const JPEGImage & other) = default;
    
    /**
    * Moves image data from another JPEGImage object and leaves that empty.
    * @p other The JPEGImage whose data is to be moved to this one.
    */
    JPEGImage & operator=(JPEGImage && other);
    
    /**
    * @return Returns the width of the image.
    */
    int width() const;
    
    /**
    * @return Returns the height of the image.
    */
    int height() const;
    
    /**
    * @return Returns the depth of the image. The image depth is the number of color channels.
    */
    int depth() const;
    
    /**
    * @return Returns a const pointer to the pixel data.
    * @note Returns a null pointer if the image is empty.
    */
    const uint8_t * bits() const;
    
    /**
    * @return Returns a pointer to the pixel data.
    * @note Returns a null pointer if the image is empty.
    */
    uint8_t * bits();
    
    /**
    * @return Returns a const pointer to the pixel data at the scanline with index y. The first
    * scanline is at index 0. Returns a null pointer if the image is empty or if y is out of bounds.
    */
    const uint8_t * scanLine(int y) const;
    
    /**
    * @return Returns a pointer to the pixel data at the scanline with index y. The first scanline
    * is at index 0. Returns a null pointer if the image is empty or if y is out of bounds.
    */
    uint8_t * scanLine(int y);
    
    /**
    * @return Returns a FeatureMatrix object wrapping the data of this image.
    */
    FeatureMatrix_<uint8_t> toMatrix();
    
    /**
    * @return Returns a constant FeatureMatrix object wrapping the data of this image.
    */
    const FeatureMatrix_<uint8_t> toMatrix() const;
    
    /**
    * @return Returns true if the image is empty. An empty image has zero size.
    */
    bool empty() const;
    
    /**
    * Saves the image to a jpeg file with the given @p filename and @p quality.
    */
    void save(const std::string & filename, int quality = 100) const;
    
    /**
    * Returns a copy of the image scaled to the given @p width and @p height.
    * If either the width or the height is zero or negative, the method returns an empty image.
    */
    JPEGImage resize(int width, int height) const;
    
    /**
    * Returns a copy of a region of the image located at @p x and @p y, and of dimensions @p width
    * and @p height.
    * @note The returned image might be smaller if some of the coordinates are outside the image.
    */
    JPEGImage crop(int x, int y, int width, int height) const;
    
    /**
    * Returns a copy of a region of the image located at @p x and @p y, and of dimensions @p width
    * and @p height.
    * In contrast to crop(), if some of the coordinates are outside the image, the image will be mirrored
    * along the borders.
    */
    JPEGImage cropPadded(int x, int y, int width, int height) const;
    
private:

    /**
    * Blur and downscale an image by a factor 2
    */
    static void Halve(const uint8_t * src, int srcWidth, int srcHeight, uint8_t * dst,
                      int dstWidth, int dstHeight, int depth);
    
    /**
    * Resize an image to the specified dimensions using bilinear interpolation
    */
    static void Resize(const uint8_t * src, int srcWidth, int srcHeight, uint8_t * dst,
                       int dstWidth, int dstHeight, int depth);
    
    int width_;
    int height_;
    int depth_;
    std::vector<uint8_t> bits_;
};

}

#endif
