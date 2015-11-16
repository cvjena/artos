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

#ifndef ARTOS_MIXTURE_H
#define ARTOS_MIXTURE_H

#include "Model.h"
#include "Patchwork.h"

namespace ARTOS
{

/**
* The Mixture class represents a mixture of deformable part-based models.
*/
class Mixture
{
public:

    /**
    * Type of a matrix of indices.
    */
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Indices;
    
    /**
    * Constructs an empty mixture. An empty mixture has no model.
    * The new mixture will be associated with the default feature extractor.
    */
    Mixture();
    
    /**
    * Constructs an empty mixture associated with a specific feature extractor.
    * An empty mixture has no models.
    *
    * @param[in] featureExtractor A shared pointer to the feature extractor which has been
    * used to create any model added to the mixture later. If a nullptr is passed here, the
    * default feature extractor will be used.
    */
    Mixture(const std::shared_ptr<FeatureExtractor> & featureExtractor);
    
    /**
    * Constructs a mixture from a list of components.
    *
    * @param[in] models A list of models (mixture components).
    *
    * @param[in] featureExtractor A shared pointer to the feature extractor which has been
    * used to create the models. If a nullptr is passed here, the default feature extractor
    * will be used.
    *
    * @throws IncompatibleException The number of features of one of the models does not
    * match the number of features reported by the given feature extractor.
    */
    Mixture(const std::vector<Model> & models, const std::shared_ptr<FeatureExtractor> & featureExtractor = nullptr);
    
    /**
    * Constructs a mixture from a list of components.
    *
    * @param[in] models A list of models (mixture components).
    *
    * @param[in] featureExtractor A shared pointer to the feature extractor which has been
    * used to create the models. If a nullptr is passed here, the default feature extractor
    * will be used.
    *
    * @throws IncompatibleException The number of features of one of the models does not
    * match the number of features reported by the given feature extractor.
    */
    Mixture(std::vector<Model> && models, const std::shared_ptr<FeatureExtractor> & featureExtractor = nullptr);
    
    /**
    * Copy constructor.
    *
    * @param[in] other Another mixture to be copied.
    */
    Mixture(const Mixture & other);
    
    /**
    * Move constructor.
    *
    * @param[in] other Another mixture to be moved.
    */
    Mixture(Mixture && other);
    
    /**
    * Copy assignment operator.
    *
    * @param[in] other Another mixture to be copied.
    */
    Mixture & operator=(const Mixture & other) = default;
    
    /**
    * Move assignment operator.
    *
    * @param[in] other Another mixture to be moved.
    */
    Mixture & operator=(Mixture && other);
    
    /**
    * Returns whether the mixture is empty. An empty mixture has no model.
    */
    bool empty() const;
    
    /**
    * Returns the list of models (mixture components).
    */
    const std::vector<Model> & models() const;
    
    /**
    * Adds a model as new component of the mixture.
    *
    * @throws IncompatibleException The number of features of the model to be added does not
    * match the number of features reported by the feature extractor associated with the mixture.
    */
    void addModel(const Model & model);
    
    /**
    * Adds a model as new component of the mixture.
    *
    * @throws IncompatibleException The number of features of the model to be added does not
    * match the number of features reported by the feature extractor associated with the mixture.
    */
    void addModel(Model && model);
    
    /**
    * Returns the minimum root filter size (`cols x rows`).
    */
    Size minSize() const;
    
    /**
    * Returns the maximum root filter size (`cols x rows`).
    */
    Size maxSize() const;
    
    /**
    * Returns a shared pointer to the FeatureExtractor used to create the models in this mixture.
    */
    std::shared_ptr<FeatureExtractor> featureExtractor() const;
    
    /**
    * Returns the scores of the convolutions + distance transforms of the models with a
    * pyramid of features (useful to compute the SVM margins).
    *
    * @param[in] pyramid Pyramid of features.
    *
    * @param[out] scores Scores for each pyramid level.
    *
    * @param[out] argmaxes Indices of the best model (mixture component) for each pyramid
    * level.
    *
    * @param[out] positions Positions of each part of each model for each pyramid level
    * (`models x parts x levels`).
    */
    void convolve(const FeaturePyramid & pyramid, std::vector<ScalarMatrix> & scores,
                  std::vector<Indices> & argmaxes,
                  std::vector< std::vector< std::vector<Model::Positions> > > * positions = 0)
                 const;
    
    /**
    * Cache the transformed version of the models' filters.
    */
    void cacheFilters() const;
    
private:

    /**
    * Returns the scores of the convolutions + distance transforms of the models with a
    * pyramid of features (useful to compute the SVM margins).
    *
    * @param[in] pyramid Pyramid of features.
    *
    * @param[out] scores Scores of each model for each pyramid level
    * (`models x levels`).
    *
    * @param[out] positions Positions of each part of each model for each pyramid level
    * (`models x parts x levels`).
    */
    void convolve(const FeaturePyramid & pyramid,
                  std::vector< std::vector<ScalarMatrix> > & scores,
                  std::vector< std::vector< std::vector<Model::Positions> > > * positions = 0)
                 const;
    
    std::vector<Model> models_; /**< The mixture components. */
    
    std::shared_ptr<FeatureExtractor> featureExtractor_; /**< The feature extractor which has been used to create the models in the mixture. */
    
    // Used to speed up the convolutions
    mutable std::vector<Patchwork::Filter> filterCache_; /**< Cache of transformed filters. */
    volatile mutable int cached_; /**< Value of Patchwork::NumInits() as when the filters have been cached the last time. */

};

/**
* Serializes a mixture to a stream, including the parameters of the associated feature extractor.
*/
std::ostream & operator<<(std::ostream & os, const Mixture & mixture);

/**
* Unserializes a mixture from a stream. The mixture will be empty in the case of error.
*
* @throws DeserializationException Data on the stream is in an unrecognized format.
*
* @throws UnknownFeatureExtractorException The feature extractor type specified in the model file is unknown.
*
* @throws UnknownParameterException One or more parameters listed on the stream are not known
* by the given feature extractor.
*
* @throws std::invalid_argument A value found on the stream for a parameter of the feature extractor
* is not allowed for the corresponding parameter.
*/
std::istream & operator>>(std::istream & is, Mixture & mixture);

}

#endif
