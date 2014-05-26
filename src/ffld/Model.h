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

#ifndef FFLD_MODEL_H
#define FFLD_MODEL_H

#include "HOGPyramid.h"

namespace FFLD
{	
/// The Model class can represent both a deformable part-based model or a training sample with
/// fixed latent variables (parts' positions). In both cases the members are the same: a list of
/// parts and a bias. If it is a sample, for each part the filter is set to the corresponding
/// features, the offset is set to the part position relative to the root, and the deformation is
/// set to the deformation gradients (<tt>dx^2 dx dy^2 dy</tt>), where dx, dy are the differences
/// between the part position and the reference part location. The dot product between the
/// deformation gradient and the model deformation then computes the deformation cost.
class Model
{
public:
	/// Type of a scalar value.
	typedef HOGPyramid::Scalar Scalar;
	
	/// Type of a 2d position (x and y).
	typedef Eigen::Vector2i Position;
	
	/// Type of a matrix of 2d positions.
	typedef Eigen::Matrix<Position, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Positions;
	
	/// Type of a 2d quadratic deformation (<tt>ax^2 + bx + cy^2 + dy</tt>).
	typedef Eigen::Matrix<Scalar, 4, 1> Deformation;
	
	/// Constructs an empty model. An empty model has an empty root and no part.
	Model();
	
	/// Constructs an one-part model (i. e. only root without other parts) from given HOG features and given bias.
	Model(const HOGPyramid::Level & root, const Scalar bias);
	
	/// Returns whether the model is empty. An empty model has an empty root and no part.
	bool empty() const;
	
	/// Returns the size of the root (<tt>rows x cols</tt>).
	std::pair<int, int> rootSize() const;
	
	/// Returns the number of parts.
	int nbParts() const;
	
	/// Returns the size of the parts (<tt>rows x cols</tt>).
	std::pair<int, int> partSize() const;
	
	/// Returns the model bias.
	Scalar bias() const;
	
	/// Returns reference to a filter of specific part.
	const HOGPyramid::Level & filters(std::size_t index) const;
	
	/// Returns the scores of the convolutions + distance transforms of the parts with a pyramid of
	/// features (useful to compute the SVM margins).
	/// @param[in] pyramid Pyramid of features.
	/// @param[out] scores Scores for each pyramid level.
	/// @param[out] positions Positions of each part of the model for each pyramid level
	/// (<tt>parts x levels</tt>).
	/// @note Unefficient, use the convolve method of the Mixture class for the Fourier accelerated
	/// version.
	void convolve(const HOGPyramid & pyramid, std::vector<HOGPyramid::Matrix> & scores,
				  std::vector<std::vector<Positions> > * positions = 0) const;
	
	/// Returns the flipped version (horizontally) of a model or a fixed sample.
	Model flip() const;
	
	/// Serializes a model to a stream.
	friend std::ostream & operator<<(std::ostream & os, const Model & model);
	
	/// Unserializes a model from a stream.
	friend std::istream & operator>>(std::istream & is, Model & model);
	
	/// Make the Mixture class a friend so that it can access private members (necessary to
	/// implement Fourier accelerated convolutions).
	friend class Mixture;
	
private:
	/// The part structure stores all the information about a part (or the root).
	struct Part
	{
		HOGPyramid::Level filter; ///< Part filter.
		Position offset; ///< Part offset (x, y) relative to the root.
		Deformation deformation; ///< Deformation cost (<tt>ax^2 + bx + cy^2 + dy</tt>).
	};
	
	/// Helper for the first convolution method. Computes the scores of the model given the
	/// convolutions of a pyramid with the parts.
	/// @param[in] pyramid Pyramid on which the scores were computed. Only its parameters
	/// (padding, interval) are needed, not the levels.
	/// @param[in] convolutions Convolutions of each part of the model for each pyramid level
	/// (<tt>parts x levels</tt>).
	/// @param[out] scores Scores of the model for each pyramid level.
	/// @param[out] positions Positions of each part of the model for each pyramid level
	/// (<tt>parts x levels</tt>).
	void convolve(const HOGPyramid & pyramid,
				  std::vector<std::vector<HOGPyramid::Matrix> > & convolutions,
				  std::vector<HOGPyramid::Matrix> & scores,
				  std::vector<std::vector<Positions> > * positions = 0) const;
	
	/// Computes a 1D quadratic distance transform (maximum convolution with a quadratic
	/// function) in linear time. For every position @c i it computes the maxima
	/// @code y[i] = \max_j x[j] + a * (i + offset - j)^2 + b * (i + offset - j) @endcode
	/// and optionally the argmaxes <tt>m[i]</tt>'s (i.e. the optimal @c j's for every @c i's).
	/// @param[in] x Array to transform.
	/// @param[in] n Length of the array.
	/// @param[in] a Coefficient of the quadratic term.
	/// @param[in] b Coefficient of the linear term.
	/// @param z Temporary buffer of length at least n + 1.
	/// @param v Temporary buffer of length at least n + 1.
	/// @param[out] y Result of the maximum convolution.
	/// @param[out] m Indices of the maxima.
	/// @param[in] offset Spatial offset between the input and the output.
	/// @param[in] t Lookup table of length n + 1 where each entry is equal to
	/// <tt>t[i] = 1 / (a * i)</tt>.
	/// @param[in] incx Stride of the array.
	/// @param[in] incy Stride of the result.
	/// @param[in] incm Stride of the indices.
	/// @note The temporary buffers @p z and @p v must be pre-allocated.
	/// @note The lookup table @t is optional but avoids a costly division.
	static void DT1D(const Scalar * x, int n, Scalar a, Scalar b, Scalar * z, int * v, Scalar * y,
					 int * m = 0, int offset = 0, const Scalar * t = 0, int incx = 1,
					 int incy = 1, int incm = 1);
	
	/// Computes a 2D quadratic distance transform by successively transforming the rows and the
	/// columns of the input matrix using the 1D transform.
	/// @param[in,out] matrix Matrix to tranform in place.
	/// @param[in] part Part from which to read the deformation cost and offset.
	/// @param tmp Temporary buffer of length at least the size of the matrix.
	/// @param[out] positions Optimal position of each part for each root location.
	static void DT2D(HOGPyramid::Matrix & matrix, const Model::Part & part, Scalar * tmp,
					 Positions * positions = 0);
	
	std::vector<Part> parts_; ///< The parts making up the model (the first one is the root).
	Scalar bias_; ///< The model bias.
};

/// Serializes a model to a stream.
std::ostream & operator<<(std::ostream & os, const Model & model);

/// Unserializes a model from a stream.
std::istream & operator>>(std::istream & is, Model & model);
}

#endif
