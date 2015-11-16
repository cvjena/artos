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

#include "Mixture.h"

#include <algorithm>
#include <utility>
#include <fstream>
#include <sstream>
#include <cstring>
#include "strutils.h"

using namespace ARTOS;
using namespace std;

Mixture::Mixture() : featureExtractor_(FeatureExtractor::defaultFeatureExtractor()), cached_(0)
{
}

Mixture::Mixture(const shared_ptr<FeatureExtractor> & featureExtractor)
: featureExtractor_((featureExtractor) ? featureExtractor : FeatureExtractor::defaultFeatureExtractor()), cached_(0)
{
}

Mixture::Mixture(const vector<Model> & models, const shared_ptr<FeatureExtractor> & featureExtractor)
: models_(models), featureExtractor_((featureExtractor) ? featureExtractor : FeatureExtractor::defaultFeatureExtractor()), cached_(0)
{
    for (const auto & m : models_)
        if (!m.empty() && m.nbFeatures() != featureExtractor_->numFeatures())
            throw IncompatibleException("Number of features of models to be added to a mixture does not match the one reported by the given FeatureExtractor.");
}

Mixture::Mixture(vector<Model> && models, const shared_ptr<FeatureExtractor> & featureExtractor)
: models_(std::move(models)), featureExtractor_((featureExtractor) ? featureExtractor : FeatureExtractor::defaultFeatureExtractor()), cached_(0)
{
    for (const auto & m : models_)
        if (!m.empty() && m.nbFeatures() != featureExtractor_->numFeatures())
            throw IncompatibleException("Number of features of models to be added to a mixture does not match the one reported by the given FeatureExtractor.");
}

Mixture::Mixture(const Mixture & other) : models_(other.models_), featureExtractor_(other.featureExtractor_), cached_(0)
{
}

Mixture::Mixture(Mixture && other) : models_(std::move(other.models_)), featureExtractor_(other.featureExtractor_), cached_(0)
{
}

Mixture & Mixture::operator=(Mixture && other)
{
    models_ = std::move(other.models_);
    featureExtractor_ = other.featureExtractor_;
    filterCache_.clear();
    cached_ = 0;
    other.filterCache_.clear();
    other.cached_ = 0;
    return *this;
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
    if (!model.empty() && model.nbFeatures() != featureExtractor_->numFeatures())
        throw IncompatibleException("Tried to mix models with a different number of features.");
    models_.push_back(model);
    cached_ = 0;
}

void Mixture::addModel(Model && model)
{
    if (!model.empty() && model.nbFeatures() != featureExtractor_->numFeatures())
        throw IncompatibleException("Tried to mix models with a different number of features.");
    models_.push_back(std::move(model));
    cached_ = 0;
}

Size Mixture::minSize() const
{
    Size size(0, 0);
    
    if (!models_.empty())
    {
        size = models_[0].rootSize();
        for (unsigned int i = 1; i < models_.size(); ++i)
            size = min(size, models_[i].rootSize());
    }
    
    return size;
}

Size Mixture::maxSize() const
{
    Size size(0, 0);
    
    if (!models_.empty())
    {
        size = models_[0].rootSize();
        for (unsigned int i = 1; i < models_.size(); ++i)
            size = max(size, models_[i].rootSize());
    }
    
    return size;
}

shared_ptr<FeatureExtractor> Mixture::featureExtractor() const
{
    return this->featureExtractor_;
}

void Mixture::convolve(const FeaturePyramid & pyramid, vector<ScalarMatrix> & scores,
                       vector<Indices> & argmaxes,
                       vector< vector< vector< Model::Positions> > > * positions) const
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
    vector< vector< ScalarMatrix> > tmp(nbModels);
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
        // The FFLD version extracted only the valid area of the convolution here,
        // i.e. (rows() - maxSize().height + 1, cols() - maxSize().width + 1), but in
        // ARTOS we've got better results on the image borders using the full size.
        scores[i].resize(pyramid.levels()[i].rows(), pyramid.levels()[i].cols());
        
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

void Mixture::convolve(const FeaturePyramid & pyramid,
                       vector< vector<ScalarMatrix> > & scores,
                       vector< vector< vector<Model::Positions> > > * positions) const
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
    if (cached_ != Patchwork::NumInits() || filterCache_.empty())
    {
        cached_ = 0;
        cacheFilters();
    }
    
    while (!cached_);
    
    // Create a patchwork
    const Patchwork patchwork(pyramid, this->maxSize() / 2 + 1);
    
    // Convolve the patchwork with the filters
    vector< vector<ScalarMatrix> > convolutions(filterCache_.size());
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
        vector< vector<ScalarMatrix> > tmp(models_[i].parts_.size());
        
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
    
    cached_ = Patchwork::NumInits();
}

ostream & ARTOS::operator<<(ostream & os, const Mixture & mixture)
{
    // Save the type and parameters of the feature extractor
    os << mixture.featureExtractor()->type() << endl << *(mixture.featureExtractor()) << endl;
    
    // Save the number of models (mixture components)
    os << mixture.models().size() << endl;
    
    // Save the models themselves
    for (unsigned int i = 0; i < mixture.models().size(); ++i)
        os << mixture.models()[i] << endl;
    
    return os;
}

istream & ARTOS::operator>>(istream & is, Mixture & mixture)
{
    mixture = Mixture();
    
    // Detect file format:
    // >= v2 starts with the type and parameters of the feature extractor
    // v1 assumes HOG feature extractor and starts straight with the number of models
    shared_ptr<FeatureExtractor> featureExtractor;
    string line;
    getline(is, line);
    if (line.empty())
        throw DeserializationException("The given stream could not be deserialized into a mixture.");
    line = trim(line);
    
    char * trailing;
    int nbModels = strtol(line.c_str(), &trailing, 10);
    if (*trailing == '\0') // old format
        featureExtractor = FeatureExtractor::create("HOG");
    else // new format
    {
        featureExtractor = FeatureExtractor::create(line);
        is >> *featureExtractor;
        is >> nbModels;
    }
    
    if (!is || (nbModels <= 0))
        throw DeserializationException("The given stream could not be deserialized into a mixture.");
    
    // Deserialize models
    vector<Model> models(nbModels);
    
    for (int i = 0; i < nbModels; ++i)
    {
        is >> models[i];
        
        if (!is || models[i].empty())
        {
            stringstream errmsg("Failed to deserialize model #");
            errmsg << (i+1);
            throw DeserializationException(errmsg.str());
        }
    }
    
    // Finish
    mixture = Mixture(models, featureExtractor);
    
    return is;
}
