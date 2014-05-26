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

#ifndef FFLD_PATCHWORK_H
#define FFLD_PATCHWORK_H

#include "HOGPyramid.h"
#include "Rectangle.h"

#include <utility>

extern "C" {
#include <fftw3.h>
}

namespace FFLD
{
/// The Patchwork class computes full convolutions much faster than the HOGPyramid class.
class Patchwork
{
public:
	/// Type of a scalar value.
	typedef std::complex<HOGPyramid::Scalar> Scalar;
	
	/// Type of a matrix.
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
	
	/// Type of a patchwork plane cell (fixed-size complex vector of size NbFeatures).
	typedef Eigen::Array<Scalar, HOGPyramid::NbFeatures, 1> Cell;
	
	/// Type of a patchwork plane (matrix of cells).
	typedef Eigen::Matrix<Cell, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Plane;
	
	/// Type of a patchwork filter (plane + original filter size).
	typedef std::pair<Plane, std::pair<int, int> > Filter;
	
	/// Constructs an empty patchwork. An empty patchwork has no plane.
	Patchwork();
	
	/// Constructs a patchwork from a pyramid.
	/// @param[in] pyramid The pyramid of features.
	/// @note If the pyramid is larger than the last maxRows and maxCols passed to the Init method
	/// the Patchwork will be empty.
	/// @note Assumes that the features of the pyramid levels are zero in the padded regions but for
	/// the last feature, which is assumed to be one.
	Patchwork(const HOGPyramid & pyramid);
	
	/// Returns the amount of horizontal zero padding (in cells).
	int padx() const;
	
	/// Returns the amount of vertical zero padding (in cells).
	int pady() const;
	
	/// Returns the number of levels per octave in the pyramid.
	int interval() const;
	
	/// Returns whether the patchwork is empty. An empty patchwork has no plane.
	bool empty() const;
	
	/// Returns the convolutions of the patchwork with filters (useful to compute the SVM margins).
	/// @param[in] filters The filters.
	/// @param[out] convolutions The convolutions (filters x levels).
	void convolve(const std::vector<Filter> & filters,
				  std::vector<std::vector<HOGPyramid::Matrix> > & convolutions) const;
	
	/// Initializes the FFTW library.
	/// @param[in] maxRows Maximum number of rows of a pyramid level (including padding).
	/// @param[in] maxCols Maximum number of columns of a pyramid level (including padding).
	/// @returns Whether the initialization was successful.
	/// @note Must be called before any other method (including constructors).
	static bool Init(int maxRows, int maxCols);
	
	/// Returns the current maximum number of rows of a pyramid level (including padding).
	static int MaxRows();
	
	/// Returns the current maximum number of columns of a pyramid level (including padding).
	static int MaxCols();
	
	/// Returns a transformed version of a filter to be used by the @c convolve method.
	/// @param[in] filter Filter to transform.
	/// @param[out] result Transformed filter.
	/// @note If Init was not already called or if the filter is larger than the last maxRows and
	/// maxCols passed to the Init method the result will be empty.
	static void TransformFilter(const HOGPyramid::Level & filter, Filter & result);
	
private:
	// Bottom-Left fill algorithm
	static int BLF(std::vector<std::pair<Rectangle, int> > & rectangles);
	
	int padx_;
	int pady_;
	int interval_;
	std::vector<std::pair<Rectangle, int> > rectangles_;
	std::vector<Plane> planes_;
	
	static int MaxRows_;
	static int MaxCols_;
	static int HalfCols_;
	
#ifndef FFLD_HOGPYRAMID_DOUBLE
	static fftwf_plan Forwards_;
	static fftwf_plan Inverse_;
#else
	static fftw_plan Forwards_;
	static fftw_plan Inverse_;
#endif
};
}

// Some compilers complain about the lack of a NumTraits for Eigen::Array<Scalar, NbFeatures, 1>
namespace Eigen
{
template <>
struct NumTraits<Array<FFLD::Patchwork::Scalar, FFLD::HOGPyramid::NbFeatures, 1> > :
	GenericNumTraits<Array<FFLD::Patchwork::Scalar, FFLD::HOGPyramid::NbFeatures, 1> >
{
	static inline FFLD::HOGPyramid::Scalar dummy_precision()
	{
		return 0; // Never actually called
	}
};
}

#endif
