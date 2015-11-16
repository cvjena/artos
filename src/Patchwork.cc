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

#include "Patchwork.h"

#include <algorithm>
#include <cstdio>
#include <numeric>

using namespace ARTOS;
using namespace std;

int Patchwork::MaxRows_(0);
int Patchwork::MaxCols_(0);
int Patchwork::HalfCols_(0);
int Patchwork::NumFeat_(0);
int Patchwork::NumInits_(0);

fftwf_plan Patchwork::Forwards_(0);
fftwf_plan Patchwork::Inverse_(0);

Patchwork::Patchwork() : padding_(0), interval_(0)
{
}

Patchwork::Patchwork(const FeaturePyramid & pyramid, const Size & padding) : padding_(padding), interval_(pyramid.interval())
{
    if (pyramid.featureExtractor()->numFeatures() != NumFeat_)
        return;
    
    const int nbLevels = pyramid.levels().size();
    rectangles_.resize(nbLevels);
    
    // Add padding to the bottom/right sides of levels since convolutions with Fourier wrap around
    for (int i = 0; i < nbLevels; ++i)
    {
        rectangles_[i].setWidth(pyramid.levels()[i].cols() + padding_.width);
        rectangles_[i].setHeight(pyramid.levels()[i].rows() + padding_.height);
    }
    
    // Build the patchwork planes
    const int nbPlanes = BLF(rectangles_, MaxCols_, MaxRows_);
    
    // Constructs an empty patchwork in case of error
    if (nbPlanes <= 0)
        return;
    
    planes_.resize(nbPlanes);
    for (int i = 0; i < nbPlanes; ++i)
        planes_[i] = Plane(MaxRows_, HalfCols_, Plane::Cell::Zero(NumFeat_));
    
    // Fill the planes with the levels from the pyramid
    for (int i = 0; i < nbLevels; ++i)
    {
        Eigen::Map<ScalarMatrix>
            plane(reinterpret_cast<FeatureScalar*>(planes_[rectangles_[i].plane()].raw()),
                  MaxRows_, HalfCols_ * 2 * NumFeat_);
        
        plane.block(rectangles_[i].y(), rectangles_[i].x() * NumFeat_,
                    rectangles_[i].height() - padding_.height, (rectangles_[i].width() - padding_.width) * NumFeat_) =
            pyramid.levels()[i].data();
    }
    
    // Transform the planes
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < nbPlanes; ++i)
        fftwf_execute_dft_r2c(Forwards_, reinterpret_cast<float *>(planes_[i].raw()),
                              reinterpret_cast<fftwf_complex *>(planes_[i].raw()));
}

const Size & Patchwork::padding() const
{
    return this->padding_;
}

int Patchwork::interval() const
{
    return interval_;
}

bool Patchwork::empty() const
{
    return planes_.empty();
}

void Patchwork::convolve(const vector<Filter> & filters,
                         vector<vector<ScalarMatrix> > & convolutions) const
{
    int i, j, k, l;
    const int nbFilters = filters.size();
    const int nbPlanes = planes_.size();
    const int nbLevels = rectangles_.size();
    
    // Early return if the patchwork or the filters are empty
    if (empty() || !nbFilters)
    {
        convolutions.clear();
        return;
    }
    
    // Pointwise multiply the transformed filters with the patchwork's planes
    // The performace measurements reported in the paper were done without reallocating the sums
    // each time by making them static
    // Even though it was faster (~10%) I removed it as it was not clean/thread safe
    vector<vector<Plane::ScalarMatrix> > sums(nbFilters);
    for (i = 0; i < nbFilters; ++i)
    {
        sums[i].resize(nbPlanes);
        for (j = 0; j < nbPlanes; ++j)
            sums[i][j].resize(MaxRows_, HalfCols_);
    }
    
    // The following assumptions are not dangerous in the sense that the program will only work
    // slower if they do not hold
    const int cacheSize = 32768; // Assume L1 cache of 32K
    const int fragmentsSize = (nbPlanes + 1) * NumFeat_ * sizeof(Scalar); // Assume nbPlanes < nbFilters
    const int step = min(cacheSize / fragmentsSize,
#ifdef _OPENMP
                         MaxRows_ * HalfCols_ / omp_get_max_threads());
#else
                         MaxRows_ * HalfCols_);
#endif
    
#pragma omp parallel for private(i,j,k,l)
    for (i = 0; i <= MaxRows_ * HalfCols_ - step; i += step)
        for (j = 0; j < nbFilters; ++j)
            for (k = 0; k < nbPlanes; ++k)
                for (l = 0; l < step; ++l)
                    sums[j][k](i + l) =
                        filters[j].first.cell(i + l).cwiseProduct(planes_[k].cell(i + l)).sum();
    
    for (i = MaxRows_ * HalfCols_ - ((MaxRows_ * HalfCols_) % step); i < MaxRows_ * HalfCols_; ++i)
        for (j = 0; j < nbFilters; ++j)
            for (k = 0; k < nbPlanes; ++k)
                sums[j][k](i) = filters[j].first.cell(i).cwiseProduct(planes_[k].cell(i)).sum();
    
    // Transform back the results and store them in convolutions
    convolutions.resize(nbFilters);
    for (i = 0; i < nbFilters; ++i)
        convolutions[i].resize(nbLevels);
    
#pragma omp parallel for private(i,j)
    for (i = 0; i < nbFilters * nbPlanes; ++i)
    {
        const int f = i / nbPlanes; // Filter index
        const int p = i % nbPlanes; // Plane index
        
        Eigen::Map<ScalarMatrix> output(reinterpret_cast<FeatureScalar*>(sums[f][p].data()),
                                 MaxRows_, HalfCols_ * 2);
        
        fftwf_execute_dft_c2r(Inverse_, reinterpret_cast<fftwf_complex *>(sums[f][p].data()),
                              output.data());
        
        for (j = 0; j < nbLevels; ++j)
            if (rectangles_[j].plane() == p)
            {
                const int rows = rectangles_[j].height() - padding_.height;
                const int cols = rectangles_[j].width() - padding_.width;
                if (rows > 0 && cols > 0)
                {
                    const int x = rectangles_[j].x();
                    const int y = rectangles_[j].y();
                    convolutions[f][j] = output.block(y, x, rows, cols);
                }
            }
    }
}

bool Patchwork::Init(int maxRows, int maxCols, int numFeatures)
{
    // It is an error if maxRows or maxCols are too small
    if ((maxRows < 2) || (maxCols < 2))
        return false;
    
    // Temporary matrices
    FeatureMatrix tmp(maxRows, maxCols + 2, numFeatures); // +2 columns required by fftw as padding
    
    int dims[2] = {maxRows, maxCols};
    
    // Use fftwf_import_wisdom_from_file and not fftwf_import_wisdom_from_filename as old versions
    // of fftw seem to not include it
    FILE * file = fopen("wisdom.fftw", "r");
    
    if (file) {
        fftwf_import_wisdom_from_file(file);
        fclose(file);
    }
    
    const fftwf_plan forwards =
        fftwf_plan_many_dft_r2c(2, dims, numFeatures, tmp.raw(), 0,
                                numFeatures, 1,
                                reinterpret_cast<fftwf_complex *>(tmp.raw()), 0,
                                numFeatures, 1, FFTW_PATIENT);
    
    const fftwf_plan inverse =
        fftwf_plan_dft_c2r_2d(dims[0], dims[1], reinterpret_cast<fftwf_complex *>(tmp.raw()),
                              tmp.raw(), FFTW_PATIENT);
    
    file = fopen("wisdom.fftw", "w");
    
    if (file) {
        fftwf_export_wisdom_to_file(file);
        fclose(file);
    }
    
    // If successful, set static variables
    if (forwards && inverse) {
        MaxRows_ = maxRows;
        MaxCols_ = maxCols;
        HalfCols_ = maxCols / 2 + 1;
        NumFeat_ = numFeatures;
        NumInits_++;
        if (Forwards_ != 0)
            fftwf_destroy_plan(Forwards_);
        Forwards_ = forwards;
        if (Inverse_ != 0)
            fftwf_destroy_plan(Inverse_);
        Inverse_ = inverse;
        return true;
    }
    
    return false;
}

int Patchwork::MaxRows()
{
    return MaxRows_;
}

int Patchwork::MaxCols()
{
    return MaxCols_;
}

int Patchwork::NumFeatures()
{
    return NumFeat_;
}

int Patchwork::NumInits()
{
    return NumInits_;
}

void Patchwork::TransformFilter(const FeatureMatrix & filter, Filter & result)
{
    // Early return if no filter given or if Init was not called or if the filter is too large
    if (filter.empty() || !MaxRows_ || filter.rows() > MaxRows_ || filter.cols() > MaxCols_ || filter.channels() != NumFeat_)
    {
        result = Filter();
        return;
    }
    
    // Copy the filter to a plane
    result.first = Plane(MaxRows_, HalfCols_, Plane::Cell::Zero(NumFeat_));
    result.second = pair<int, int>(filter.rows(), filter.cols());
    
    FeatureMatrix plane(reinterpret_cast<FeatureScalar*>(result.first.raw()),
                        MaxRows_, HalfCols_ * 2, NumFeat_);
    
    for (int y = 0; y < filter.rows(); ++y)
        for (int x = 0; x < filter.cols(); ++x)
            plane((MaxRows_ - y) % MaxRows_, (MaxCols_ - x) % MaxCols_)
                    = filter(y, x) / static_cast<FeatureScalar>(MaxRows_ * MaxCols_);
    
    // Transform that plane 
    fftwf_execute_dft_r2c(Forwards_, reinterpret_cast<float *>(plane.raw()),
                          reinterpret_cast<fftwf_complex *>(result.first.raw()));
}
