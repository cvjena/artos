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

#ifndef FFLD_MIXTURE_H
#define FFLD_MIXTURE_H

#include "Model.h"
#include "Patchwork.h"

namespace FFLD
{
/// The Mixture class represents a mixture of deformable part-based models.
class Mixture
{
public:
	/// Type of a matrix of indices.
	typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Indices;
	
	/// Constructs an empty mixture. An empty mixture has no model.
	Mixture();
	
	/// Constructs a mixture from parameters.
	/// @param[in] models A list of models (mixture components).
	Mixture(const std::vector<Model> & models);
	
	/// Constructs a mixture from parameters.
	/// @param[in] models A list of models (mixture components).
	Mixture(std::vector<Model> && models);
	
	/// Copy constructor.
	/// @param[in] other Another mixture to be copied.
	Mixture(const Mixture & other);
	
	/// Move constructor.
	/// @param[in] other Another mixture to be moved.
	Mixture(Mixture && other);
	
	/// Copy assignment operator.
	/// @param[in] other Another mixture to be copied.
	Mixture & operator=(const Mixture & other) = default;
	
	/// Move assignment operator.
	/// @param[in] other Another mixture to be moved.
	Mixture & operator=(Mixture && other);
	
	/// Returns whether the mixture is empty. An empty mixture has no model.
	bool empty() const;
	
	/// Returns the list of models (mixture components).
	const std::vector<Model> & models() const;
	
	/// Adds a model as new component of the mixture.
	void addModel(const Model & model);
	
	/// Adds a model as new component of the mixture.
	void addModel(Model && model);
	
	/// Returns the minimum root filter size (<tt>rows x cols</tt>).
	std::pair<int, int> minSize() const;
	
	/// Returns the maximum root filter size (<tt>rows x cols</tt>).
	std::pair<int, int> maxSize() const;
	
	/// Returns the scores of the convolutions + distance transforms of the models with a
	/// pyramid of features (useful to compute the SVM margins).
	/// @param[in] pyramid Pyramid of features.
	/// @param[out] scores Scores for each pyramid level.
	/// @param[out] argmaxes Indices of the best model (mixture component) for each pyramid
	/// level.
	/// @param[out] positions Positions of each part of each model for each pyramid level
	/// (<tt>models x parts x levels</tt>).
	void convolve(const HOGPyramid & pyramid, std::vector<HOGPyramid::Matrix> & scores,
				  std::vector<Indices> & argmaxes,
				  std::vector<std::vector<std::vector<Model::Positions> > > * positions = 0)
				 const;
	
	/// Cache the transformed version of the models' filters.
	void cacheFilters() const;
	
private:
	/// Returns the scores of the convolutions + distance transforms of the models with a
	/// pyramid of features (useful to compute the SVM margins).
	/// @param[in] pyramid Pyramid of features.
	/// @param[out] scores Scores of each model for each pyramid level
	/// (<tt>models x levels</tt>).
	/// @param[out] positions Positions of each part of each model for each pyramid level
	/// (<tt>models x parts x levels</tt>).
	void convolve(const HOGPyramid & pyramid,
				  std::vector<std::vector<HOGPyramid::Matrix> > & scores,
				  std::vector<std::vector<std::vector<Model::Positions> > > * positions = 0)
				 const;
	
	std::vector<Model> models_; ///< The mixture components.
	
	// Used to speed up the convolutions
	mutable std::vector<Patchwork::Filter> filterCache_; // Cache of transformed filters
	volatile mutable bool cached_;
};

/// Serializes a mixture to a stream.
std::ostream & operator<<(std::ostream & os, const Mixture & mixture);

/// Unserializes a mixture from a stream.
std::istream & operator>>(std::istream & is, Mixture & mixture);
}

#endif
