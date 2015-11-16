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

#ifndef ARTOS_MODEL_H
#define ARTOS_MODEL_H

#include "FeaturePyramid.h"
#include <Eigen/StdVector>

namespace ARTOS
{

/**
* The Model class can represent both a deformable part-based model or a training sample with
* fixed latent variables (parts' positions). In both cases the members are the same: a list of
* parts and a bias. If it is a sample, for each part the filter is set to the corresponding
* features, the offset is set to the part position relative to the root, and the deformation is
* set to the deformation gradients (`dx^2 dx dy^2 dy`), where dx, dy are the differences
* between the part position and the reference part location. The dot product between the
* deformation gradient and the model deformation then computes the deformation cost.
*/
class Model
{
public:
 
    typedef FeatureScalar Scalar; /**< Type of a scalar value. */
    typedef Eigen::Vector2i Position; /**< Type of a 2d position (x and y). */
    typedef Eigen::Matrix<Position, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Positions; /**< Type of a matrix of 2d positions. */
    typedef Eigen::Matrix<Scalar, 4, 1> Deformation; /**< Type of a 2d quadratic deformation (`ax^2 + bx + cy^2 + dy`). */
    
    /**
    * Constructs an empty model. An empty model has an empty root and no part.
    */
    Model();
    
    /**
    * Constructs an one-part model (i. e. only root without other parts) from given features and given bias.
    *
    * @param[in] root The root model of the new one-part model.
    *
    * @param[in] bias The bias of the new model.
    */
    Model(const FeatureMatrix & root, const Scalar bias);
    
    /**
    * Constructs an one-part model (i. e. only root without other parts) from given features and given bias.
    *
    * @param[in] root The root model of the new one-part model.
    *
    * @param[in] bias The bias of the new model.
    */
    Model(FeatureMatrix && root, const Scalar bias);
    
    /**
    * Copy constructor
    */
    Model(const Model&) = default;
    
    /**
    * Move constructor
    */
    Model(Model && other);
    
    /**
    * @return Returns whether the model is empty. An empty model has an empty root and no part.
    */
    bool empty() const;
    
    /**
    * @return Returns the number of channels (i.e. features) of the root (0 if the model is empty).
    */
    int nbFeatures() const;
    
    /**
    * @return Returns the size of the root (`cols x rows`).
    */
    Size rootSize() const;
    
    /**
    * @return Returns the number of parts.
    */
    int nbParts() const;
    
    /**
    * Returns the size of the parts (`cols x rows`).
    */
    Size partSize() const;
    
    /**
    * @return Returns the model bias.
    */
    Scalar bias() const;
    
    /**
    * @return Returns reference to a filter of specific part.
    */
    const FeatureMatrix & filters(std::size_t index) const;
    
    /**
    * Computes a flipped version of this model by flipping all of its parts using the
    * given feature extractor.
    *
    * @param[in] featureExtractor Pointer to the feature extractor to be used for feature flipping.
    *
    * @see FeatureExtractor::flip()
    *
    * @return Returns a flipped version of this model.
    */
    Model flip(const std::shared_ptr<FeatureExtractor> & featureExtractor) const;
    
    /**
    * Copies another model.
    */
    Model & operator=(const Model &) = default;
    
    /**
    * Moves the data of another model to this one and leaves the other one empty.
    */
    Model & operator=(Model && other);
    
    /**
    * Serializes a model to a stream.
    */
    friend std::ostream & operator<<(std::ostream & os, const Model & model);
    
    /**
    * Unserializes a model from a stream.
    */
    friend std::istream & operator>>(std::istream & is, Model & model);
    
    /**
    * Make the Mixture class a friend so that it can access private members (necessary to
    * implement Fourier accelerated convolutions).
    */
    friend class Mixture;
    
private:

    /**
    * The part structure stores all the information about a part (or the root).
    */
    struct Part
    {
        FeatureMatrix filter; /**< Part filter. */
        Position offset; /**< Part offset (x, y) relative to the root. */
        Deformation deformation; /**< Deformation cost (`ax^2 + bx + cy^2 + dy`). */
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    
    /**
    * Helper for the public convolution method of the Mixture class.
    * Computes the scores of the model given the convolutions of a pyramid with the parts.
    *
    * @param[in] pyramid Pyramid on which the scores were computed.
    *
    * @param[in] convolutions Convolutions of each part of the model for each pyramid level
    * (`parts x levels`).
    *
    * @param[out] scores Scores of the model for each pyramid level.
    *
    * @param[out] positions Positions of each part of the model for each pyramid level
    * (`parts x levels`).
    */
    void convolve(const FeaturePyramid & pyramid,
                  std::vector< std::vector<ScalarMatrix> > & convolutions,
                  std::vector<ScalarMatrix> & scores,
                  std::vector< std::vector<Positions> > * positions = 0) const;
    
    /**
    * Computes a 1D quadratic distance transform (maximum convolution with a quadratic
    * function) in linear time. For every position @c i it computes the maxima
    * @code y[i] = \max_j x[j] + a * (i + offset - j)^2 + b * (i + offset - j) @endcode
    * and optionally the argmaxes <tt>m[i]</tt>'s (i.e. the optimal @c j's for every @c i's).
    *
    * @param[in] x Array to transform.
    * @param[in] n Length of the array.
    * @param[in] a Coefficient of the quadratic term.
    * @param[in] b Coefficient of the linear term.
    * @param z Temporary buffer of length at least n + 1.
    * @param v Temporary buffer of length at least n + 1.
    * @param[out] y Result of the maximum convolution.
    * @param[out] m Indices of the maxima.
    * @param[in] offset Spatial offset between the input and the output.
    * @param[in] t Lookup table of length n + 1 where each entry is equal to
    * `t[i] = 1 / (a * i)`.
    * @param[in] incx Stride of the array.
    * @param[in] incy Stride of the result.
    * @param[in] incm Stride of the indices.
    * @note The temporary buffers @p z and @p v must be pre-allocated.
    * @note The lookup table @t is optional but avoids a costly division.
    */
    static void DT1D(const Scalar * x, int n, Scalar a, Scalar b, Scalar * z, int * v, Scalar * y,
                     int * m = 0, int offset = 0, const Scalar * t = 0, int incx = 1,
                     int incy = 1, int incm = 1);
    
    /**
    * Computes a 2D quadratic distance transform by successively transforming the rows and the
    * columns of the input matrix using the 1D transform.
    *
    * @param[in,out] matrix Matrix to tranform in place.
    * @param[in] part Part from which to read the deformation cost and offset.
    * @param tmp Temporary buffer of length at least the size of the matrix.
    * @param[out] positions Optimal position of each part for each root location.
    */
    static void DT2D(ScalarMatrix & matrix, const Model::Part & part, Scalar * tmp,
                     Positions * positions = 0);
    
    std::vector< Part, Eigen::aligned_allocator<Part> > parts_; /**< The parts making up the model (the first one is the root). */
    Scalar bias_; /**< The model bias. */
};

/**
* Serializes a model to a stream.
*/
std::ostream & operator<<(std::ostream & os, const Model & model);

/**
* Unserializes a model from a stream.
*/
std::istream & operator>>(std::istream & is, Model & model);

}

#endif
