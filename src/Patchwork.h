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

#ifndef ARTOS_PATCHWORK_H
#define ARTOS_PATCHWORK_H

#include "FeaturePyramid.h"
#include "blf.h"

extern "C" {
#include <fftw3.h>
}

namespace ARTOS
{

/**
* The Patchwork class computes full convolutions much faster than the HOGPyramid class.
*/
class Patchwork
{

public:

    /**
    * Type of a complex scalar value.
    */
    typedef std::complex<FeatureScalar> Scalar;
    
    /**
    * Type of a patchwork plane (matrix of complex cells).
    */
    typedef FeatureMatrix_<Scalar> Plane;
    
    /**
    * Type of a patchwork filter (plane + original filter size).
    */
    typedef std::pair<Plane, std::pair<int, int> > Filter;
    
    /**
    * Constructs an empty patchwork. An empty patchwork has no plane.
    */
    Patchwork();
    
    /**
    * Constructs a patchwork from a pyramid.
    *
    * @param[in] pyramid The pyramid of features.
    *
    * @param[in] padding Padding to add between levels from the pyramid in each direction.
    * The padding should be at least half as large as the largest filter.
    *
    * @note If the pyramid (including padding) is larger than the last maxRows and maxCols passed to the Init method
    * or it has more features than specified there, the Patchwork will be empty.
    */
    Patchwork(const FeaturePyramid & pyramid, const Size & padding);
    
    /**
    * @return Returns the amount of zero padding added between levels from the pyramid
    * in each direction.
    */
    const Size & padding() const;
    
    /**
    * @return Returns the number of levels per octave in the pyramid.
    */
    int interval() const;
    
    /**
    * @return Returns true if the patchwork is empty. An empty patchwork has no plane.
    */
    bool empty() const;
    
    /**
    * Computes the convolutions of the patchwork with filters (useful to compute the SVM margins).
    *
    * @param[in] filters The filters.
    *
    * @param[out] convolutions The convolutions (filters x levels).
    */
    void convolve(const std::vector<Filter> & filters,
                  std::vector< std::vector<ScalarMatrix> > & convolutions) const;
    
    /**
    * Initializes the FFTW library.
    *
    * @param[in] maxRows Maximum number of rows of a pyramid level (including padding).
    *
    * @param[in] maxCols Maximum number of columns of a pyramid level (including padding).
    *
    * @param[in] numFeatures Number of features per cell.
    *
    * @returns Returns true if the initialization was successful.
    *
    * @note Must be called before any other method (including constructors).
    */
    static bool Init(int maxRows, int maxCols, int numFeatures);
    
    /**
    * @return Returns the current maximum number of rows of a pyramid level (including padding).
    */
    static int MaxRows();
    
    /**
    * @return Returns the current maximum number of columns of a pyramid level (including padding).
    */
    static int MaxCols();
    
    /**
    * @return Returns the current number of features per cell.
    */
    static int NumFeatures();
    
    /**
    * @return Returns the current number of calls to Init() made so far. This can be used to check
    * if any cached filters are still up to date.
    */
    static int NumInits();
    
    /**
    * Returns a transformed version of a filter to be used by the @c convolve method.
    *
    * @param[in] filter Filter to transform.
    *
    * @param[out] result Transformed filter.
    *
    * @note If Init has not been called yet or if the filter is larger than the last maxRows and
    * maxCols passed to the Init method, the result will be empty.
    */
    static void TransformFilter(const FeatureMatrix & filter, Filter & result);


private:
    
    Size padding_;
    int interval_;
    std::vector<PatchworkRectangle> rectangles_;
    std::vector<Plane> planes_;
    
    static int MaxRows_;
    static int MaxCols_;
    static int HalfCols_;
    static int NumFeat_;
    static int NumInits_;
    
    static fftwf_plan Forwards_;
    static fftwf_plan Inverse_;
};

}

#endif
