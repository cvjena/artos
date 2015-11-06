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

#ifndef FFLD_HOGPYRAMID_H
#define FFLD_HOGPYRAMID_H

#include "JPEGImage.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace FFLD
{
/// The HOGPyramid class computes and stores the HOG features extracted from a jpeg image at
/// multiple scales. The scale of the pyramid level of index @c i is given by the following formula:
/// 2^(1 - @c i / @c interval), so that the first scale is at double the resolution of the original
/// image). Each level is padded with zeros horizontally and vertically by a fixed amount. The last
/// feature is special: it takes the value one in the padding and zero otherwise.
/// @note Define the PASCAL_HOGPYRAMID_FELZENSZWALB_FEATURES flag during compilation to use
/// Felzenszwalb's original features (slower and not as accurate as they do no angular
/// interpolation, provided for compatibility only).
/// @note Define the PASCAL_HOGPYRAMID_DOUBLE to use double scalar values instead of float (slower,
/// uses twice the amount of memory, and the increase in precision is not necessarily useful).
class HOGPyramid
{
public:
	/// Number of HOG features (guaranteed to be even). Fixed at compile time for both ease of use
	/// and optimal performance.
	static const int NbFeatures = 32;
	
	/// Type of a scalar value.
#ifndef FFLD_HOGPYRAMID_DOUBLE
	typedef float Scalar;
#else
	typedef double Scalar;
#endif
	
	/// Type of a matrix.
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
	
	/// Type of a sparse matrix.
	typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;
	
	/// Type of a pyramid level cell (fixed-size vector of length NbFeatures).
	typedef Eigen::Array<Scalar, NbFeatures, 1> Cell;
	
	/// Type of a pyramid level (matrix of cells).
	typedef Eigen::Matrix<Cell, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Level;
	
	/// Constructs an empty pyramid. An empty pyramid has no level.
	HOGPyramid();
	
	/// Constructs a pyramid from parameters and a list of levels.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
	/// @param[in] levels List of pyramid levels.
	/// @note The amount of padding and the interval should be at least 1.
	HOGPyramid(int padx, int pady, int interval, const std::vector<Level> & levels);
	
	/// Constructs a pyramid from parameters and a list of levels.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
	/// @param[in] levels List of pyramid levels whose contents will be moved.
	/// @note The amount of padding and the interval should be at least 1.
	HOGPyramid(int padx, int pady, int interval, std::vector<Level> && levels);
	
	/// Constructs a pyramid from the JPEGImage of a Scene.
	/// @param[in] image The JPEGImage of the Scene.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
	/// @note The amount of padding and the interval should be at least 1.
	HOGPyramid(const JPEGImage & image, int padx, int pady, int interval = 10);
	
	/// Constructs a pyramid by copying the levels of another one.
	/// @param[in] other The HOGPyramid those levels are to be copied.
	HOGPyramid(const HOGPyramid & other) = default;
	
	/// Constructs a pyramid by moving the levels of another one and leaving that one empty.
	/// @param[in] other The HOGPyramid those levels are to be moved.
	HOGPyramid(HOGPyramid && other);
	
	/// Copies the levels of another HOGPyramid to this one.
	/// @param[in] other The HOGPyramid those levels are to be copied.
	HOGPyramid & operator=(const HOGPyramid & other);
	
	/// Moves the levels of another HOGPyramid to this one and leaves the other one empty.
	/// @param[in] other The HOGPyramid those levels are to be moved.
	HOGPyramid & operator=(HOGPyramid && other);
	
	/// Returns whether the pyramid is empty. An empty pyramid has no level.
	bool empty() const;
	
	/// Returns the amount of horizontal zero padding (in cells).
	int padx() const;
	
	/// Returns the amount of vertical zero padding (in cells).
	int pady() const;
	
	/// Returns the number of levels per octave in the pyramid.
	int interval() const;
	
	/// Returns the pyramid levels.
	/// @note Scales are given by the following formula: 2^(1 - @c index / @c interval).
	const std::vector<Level> & levels() const;
	
	/// Returns the convolutions of the pyramid with a filter (useful to compute the SVM margins).
	/// @param[in] filter Filter.
	/// @param[out] convolutions Convolution for each level.
	void convolve(const Level & filter, std::vector<Matrix> & convolutions) const;
	
	/// Returns the sparse convolutions of the pyramid with a filter (useful to compute a subset of
	/// the SVM margins).
	/// @param[in] filter The filter.
	/// @param[out] convolutions The results of the convolutions for each level.
	void convolve(const Level & filter, std::vector<SparseMatrix> & convolutions) const;
	
	/// Returns the sum of the convolutions of the pyramid with a pyramid of labels (useful to
	/// compute the gradient of the SVM loss).
	/// @param[in] labels The pyramid of labels.
	/// @param[out] sum The sum of the results of the convolutions.
	/// @note The size of the sum is inferred from the size of the labels, which should thus
	/// all have the size of their corresponding level minus the size of the sum plus one.
	/// @note In case labels are empty the sum will be empty too.
	void convolve(const std::vector<Matrix> & labels, Level & sum) const;
	
	/// Returns the sum of the sparse convolutions of the pyramid with a pyramid of labels (useful
	/// to compute the gradient of the SVM loss).
	/// @param[in] labels The pyramid of labels.
	/// @param[out] sum The sum of the results of the convolutions.
	/// @note The size of the sum is inferred from the size of the labels, which should thus all
	/// have the size of their corresponding level minus the size of the sum plus one.
	/// @note In case labels are empty the sum will be empty too.
	void convolve(const std::vector<SparseMatrix> & labels, Level & sum) const;
	
	/// Converts a pyramid level to a simple matrix (useful to apply standard matrix operations to
	/// it).
	/// @note The size of the matrix will be rows x (cols * NbFeatures).
	static Eigen::Map<Matrix, Eigen::Aligned> Convert(Level & level);
	
	/// Converts a const pyramid level to a simple const matrix (useful to apply standard matrix
	/// operations to it).
	/// @note The size of the matrix will be rows x (cols * NbFeatures).
	static Eigen::Map<const Matrix, Eigen::Aligned> Convert(const Level & level);
	
	/// Returns the flipped version (horizontally) of a filter.
	static HOGPyramid::Level Flip(const HOGPyramid::Level & filter);
	
#ifndef FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES
	// Efficiently computes Histogram of Oriented Gradient (HOG) features
	// Code to compute HOG features as described in "Object Detection with Discriminatively Trained
	// Part Based Models" by Felzenszwalb, Girshick, McAllester and Ramanan, PAMI10
	// cellSize should be either 4 or 8
	static void Hog(const JPEGImage & image, Level & level, int padx, int pady,
					int cellSize = 8);
#else
	// Felzenszwalb version (not as accurate, provided for compatibility only)
	static void Hog(const uint8_t * bits, int width, int height, int depth, Level & level,
					int cellSize = 8);
#endif
	
private:
	// Computes the 2D convolution of a pyramid level with a filter
	static void Convolve(const Level & x, const Level & y, Matrix & z);
	
	// Computes the sparse 2D convolution of a pyramid level with a filter
	static void Convolve(const Level & x, const Level & y, SparseMatrix & z);
	
	// Computes the 2D convolution of a pyramid level with labels
	static void Convolve(const Level & x, const Matrix & z, Level & y);
	
	// Computes the 2D convolution of a pyramid level with sparse labels
	static void Convolve(const Level & x, const SparseMatrix & z, Level & y);
	
	int padx_;
	int pady_;
	int interval_;
	std::vector<Level> levels_;
};
}

// Some compilers complain about the lack of a NumTraits for Eigen::Array<Scalar, NbFeatures, 1>
namespace Eigen
{
template <>
struct NumTraits<Array<FFLD::HOGPyramid::Scalar, FFLD::HOGPyramid::NbFeatures, 1> > :
	GenericNumTraits<Array<FFLD::HOGPyramid::Scalar, FFLD::HOGPyramid::NbFeatures, 1> >
{
	static inline FFLD::HOGPyramid::Scalar dummy_precision()
	{
		return 0; // Never actually called
	}
};
}

#endif
