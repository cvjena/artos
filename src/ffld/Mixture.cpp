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

#include "Intersector.h"
#include "Mixture.h"

#include <algorithm>
#include <fstream>

using namespace Eigen;
using namespace FFLD;
using namespace std;

Mixture::Mixture() : cached_(false)
{
}

Mixture::Mixture(const vector<Model> & models) : models_(models), cached_(false)
{
}

Mixture::Mixture(const Mixture & other) : models_(other.models_), cached_(false)
{
}

bool Mixture::empty() const
{
	return models_.empty();
}

const vector<Model> & Mixture::models() const
{
	return models_;
}

void Mixture::addModel(const Model & model)
{
	models_.push_back(model);
	cached_ = false;
}

pair<int, int> Mixture::minSize() const
{
	pair<int, int> size(0, 0);
	
	if (!models_.empty()) {
		size = models_[0].rootSize();
		
		for (unsigned int i = 1; i < models_.size(); ++i) {
			size.first = min(size.first, models_[i].rootSize().first);
			size.second = min(size.second, models_[i].rootSize().second);
		}
	}
	
	return size;
}

pair<int, int> Mixture::maxSize() const
{
	pair<int, int> size(0, 0);
	
	if (!models_.empty()) {
		size = models_[0].rootSize();
		
		for (unsigned int i = 1; i < models_.size(); ++i) {
			size.first = max(size.first, models_[i].rootSize().first);
			size.second = max(size.second, models_[i].rootSize().second);
		}
	}
	
	return size;
}

void Mixture::convolve(const HOGPyramid & pyramid, vector<HOGPyramid::Matrix> & scores,
					   vector<Indices> & argmaxes,
					   vector<vector<vector<Model::Positions> > > * positions) const
{
	if (empty() || pyramid.empty()) {
		scores.clear();
		argmaxes.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	const int nbModels = models_.size();
	const int nbLevels = pyramid.levels().size();
	
	// Convolve with all the models
	vector<vector<HOGPyramid::Matrix> > tmp(nbModels);
	convolve(pyramid, tmp, positions);
	
	// In case of error
	if (tmp.empty()) {
		scores.clear();
		argmaxes.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	// Resize the scores and argmaxes
	scores.resize(nbLevels);
	argmaxes.resize(nbLevels);
	
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < nbLevels; ++i) {
		scores[i].resize(pyramid.levels()[i].rows() - maxSize().first + 1,
						 pyramid.levels()[i].cols() - maxSize().second + 1);
		
		argmaxes[i].resize(scores[i].rows(), scores[i].cols());
		
		for (int y = 0; y < scores[i].rows(); ++y) {
			for (int x = 0; x < scores[i].cols(); ++x) {
				int argmax = 0;
				
				for (int j = 1; j < nbModels; ++j)
					if (tmp[j][i](y, x) > tmp[argmax][i](y, x))
						argmax = j;
				
				scores[i](y, x) = tmp[argmax][i](y, x);
				argmaxes[i](y, x) = argmax;
			}
		}
	}
}

void Mixture::convolve(const HOGPyramid & pyramid,
					   vector<vector<HOGPyramid::Matrix> > & scores,
					   vector<vector<vector<Model::Positions> > > * positions) const
{
	if (empty() || pyramid.empty()) {
		scores.clear();
		
		if (positions)
			positions->clear();
	}
	
	const int nbModels = models_.size();
	
	scores.resize(nbModels);
	
	if (positions)
		positions->resize(nbModels);
	
	// Transform the filters if needed
#pragma omp critical
	if (filterCache_.empty())
		cacheFilters();
	
	while (!cached_);
	
	// Create a patchwork
	const Patchwork patchwork(pyramid);
	
	// Convolve the patchwork with the filters
	vector<vector<HOGPyramid::Matrix> > convolutions(filterCache_.size());
	patchwork.convolve(filterCache_, convolutions);
	
	// In case of error
	if (convolutions.empty()) {
		scores.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	// Save the offsets of each model in the filter list
	vector<int> offsets(nbModels);
	
	for (int i = 0, j = 0; i < nbModels; ++i) {
		offsets[i] = j;
		j += models_[i].parts_.size();
	}
	
	// For each model
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < nbModels; ++i) {
		vector<vector<HOGPyramid::Matrix> > tmp(models_[i].parts_.size());
		
		for (size_t j = 0; j < tmp.size(); ++j)
			tmp[j].swap(convolutions[offsets[i] + j]);
		
		models_[i].convolve(pyramid, tmp, scores[i], positions ? &(*positions)[i] : 0);
	}
}

void Mixture::cacheFilters() const
{
	// Count the number of filters
	int nbFilters = 0;
	
	for (size_t i = 0; i < models_.size(); ++i)
		nbFilters += models_[i].parts_.size();
	
	// Transform all the filters
	filterCache_.resize(nbFilters);
	
	for (size_t i = 0, j = 0; i < models_.size(); ++i) {
		int k;
#pragma omp parallel for private(k)
		for (k = 0; k < models_[i].parts_.size(); ++k)
			Patchwork::TransformFilter(models_[i].parts_[k].filter, filterCache_[j + k]);
		
		j += models_[i].parts_.size();
	}
	
	cached_ = true;
}

ostream & FFLD::operator<<(ostream & os, const Mixture & mixture)
{
	// Save the number of models (mixture components)
	os << mixture.models().size() << endl;
	
	// Save the models themselves
	for (unsigned int i = 0; i < mixture.models().size(); ++i)
		os << mixture.models()[i] << endl;
	
	return os;
}

istream & FFLD::operator>>(istream & is, Mixture & mixture)
{
	int nbModels;
	is >> nbModels;
	
	if (!is || (nbModels <= 0)) {
		mixture = Mixture();
		return is;
	}
	
	vector<Model> models(nbModels);
	
	for (int i = 0; i < nbModels; ++i) {
		is >> models[i];
		
		if (!is || models[i].empty()) {
			mixture = Mixture();
			return is;
		}
	}
	
	mixture = Mixture(models);
	
	return is;
}
