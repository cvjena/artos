#ifndef ARTOS_STATIONARYBACKGROUND_H
#define ARTOS_STATIONARYBACKGROUND_H

#include <string>
#include <Eigen/Core>
#include "defs.h"
#include "SynsetIterators.h"

namespace ARTOS
{

/**
* Class for reading and writing stationary background statistics, i. e. general mean and covariance, from/to a binary file.
*
* Beyond that, this class is also capable of reconstructing a covariance matrix of a specific size from the stationary
* autocorrelation function read before.  
* The great deal behind this: Each covariance between two cells \f$x_{i,j}\f$ and \f$x_{m,n}\f$ only depends on the offset \f$(m-i, n-j)\f$.
* Since there are only N offsets given a total of N cells, the autocorrelation function only has to come up with
* \f$\mathcal{O}(N \cdot num\_features^2)\f$ instead of \f$\mathcal{O}(N^2 \cdot num\_features^2)\f$, which the full covariance matrix needs.  
* Therefore, N, i. e. the number of spatial offsets used for learning the statistics, is the upper bound for the number
* of rows and columns of reconstructed covariance matrices.
*
* This class may also be used to learn such statistics from ImageNet images using learnMean() and learnCovariance().
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class StationaryBackground
{

public:

    /**
    * Mean feature vector for a single cell.
    */
    typedef Eigen::Array<float, Eigen::Dynamic, 1> MeanVector;
    
    /**
    * Covariance matrix between features for a specific offset.
    */
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CovMatrix;
    
    /**
    * Vector of feature covariance matrices.
    */
    typedef Eigen::Array<CovMatrix, Eigen::Dynamic, 1> CovMatrixArray;
    
    /**
    * Matrix of feature covariance matrices (used in reconstruction process).
    */
    typedef Eigen::Matrix<CovMatrix, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CovMatrixMatrix;
    
    /**
    * Array of spatial offsets. Each row is a (dx, dy) tuple.
    */
    typedef Eigen::Array<int, Eigen::Dynamic, 2, Eigen::RowMajor> OffsetArray;
    
    /**
    * A generic 2-D matrix of floats.
    */
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    
    
    /**
    * Constructs uninitialized background statistics, which can be read later using `readFromFile`.
    */
    StationaryBackground() : mean(), cov(), offsets(), cellSize(0) { };
    
    /**
    * Constructs empty background statistics with given dimensions.
    *
    * @param[in] numOffsets Number of spatial offsets.
    *
    * @param[in] numFeatures Number of features.
    *
    * @param[in] cellSize Number of pixels each cell spans in both dimensions.
    */
    StationaryBackground(const unsigned int numOffsets, const unsigned int numFeatures = 31, const unsigned int cellSize = 8);
    
    /**
    * Constructs a new background statistics object and reads the statistics from a given file.
    *
    * @param[in] backgroundFile Path of the binary background statistics file.
    */
    StationaryBackground(const std::string & backgroundFile) : mean(), cov(), offsets(), cellSize(0)
    { this->readFromFile(backgroundFile); };
    
    /**
    * Copy constructor.
    */
    StationaryBackground(const StationaryBackground & other)
    : mean(other.mean), cov(other.cov), offsets(other.offsets), cellSize(other.cellSize) { };
    
    /**
    * Reads background statistics from a file.
    *
    * @param[in] filename Path of the binary background statistics file.
    *
    * @return True if the file could be read successfully, false if it is inaccessible or invalid.
    * empty() may also be used to check the status of this operation after it has finished.
    */
    bool readFromFile(const std::string & filename);
    
    /**
    * Writes the background statistics hold by this object to a file.
    *
    * @param[in] filename Path to write the file to.
    *
    * @return Returns true if these statistics aren't empty and the file could be written successfully, otherwise false.
    */
    bool writeToFile(const std::string & filename);

    /**
    * Resets this object to it's initial state by clearing all statistics and resizing all matrices and vectors to zero size.
    */
    void clear();

    /**
    * Determines if statistics are present or not.
    *
    * @return True if the matrices and vectors holding the statistics have zero size.
    */
    bool empty() const { return (this->mean.size() == 0); };
    
    /**
    * @return Number of features each cell has.
    */
    unsigned int getNumFeatures() const { return this->mean.size(); };
    
    /**
    * @return Number of spatial offsets available.
    */
    unsigned int getNumOffsets() const { return this->cov.size(); };
    
    /**
    * @return Maximum available offset in x or y direction or -1 if the offset array is empty.
    */
    int getMaxOffset() const { return (this->offsets.rows() > 0) ? this->offsets.abs().maxCoeff() : -1; };
    
    /**
    * Reconstructs a covariance matrix from the spatial autocorrelation function for a specific number of
    * rows and columns. The resulting 4-D matrix is of the form `cov(xy1, xy2)(feature1, feature2)`, while
    * linearisation of a spatial position (x, y) to xy is done in row-major order.
    *
    * @param[in] rows Number of rows to reconstruct the covariance matrix for.
    *
    * @param[in] cols Number of columns to reconstruct the covariance matrix for.
    *
    * @return The reconstructed covariance matrix.
    *
    * @see computeFlattenedCovariance
    */
    CovMatrixMatrix computeCovariance(const int rows, const int cols);
    
    /**
    * Reconstructs a covariance matrix from the spatial autocorrelation function for a specific number of
    * rows and columns and flattens it into a 2-D matrix, so that the k-th feature of the cell at position (x, y)
    * yields to the index `(x * cols + y) * numFeatures + k` in the resulting matrix.
    */
    Matrix computeFlattenedCovariance(const int rows, const int cols, unsigned int features = 0);
    
    /**
    * Learns a mean feature vector from features extracted from different positions at various scales of a set of images.
    * After that, learnCovariance() may be used to learn a spatial autocorrelation function too.
    * Use writeToFile() to save the learned statistics afterwards.
    *
    * @param[in] imgIt An ImageIterator providing images to learn background statistics from.
    * The iterator will be rewound at the beginning of the process.
    *
    * @param[in] numImages Maximum number of images to learn from. If set to 0, all images provided by the
    * iterator will be used (may take really, really long!).
    *
    * @param[in] progressCB Optionally, a callback that is called to populate the progress of the procedure.
    * The first parameter to the callback will be the number of processed images and the second parameter will be equal to `numImages`.
    * For example, the argument list (5, 10) means that the learning is half way done.  
    * The callback may return false to abort the operation. To continue, it must return true.  
    * Note, that the callback will be used only if `numImages` is different from 0.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    */
    void learnMean(ImageIterator & imgIt, const unsigned int numImages = 0,
                   ProgressCallback progressCB = NULL, void * cbData = NULL);
    
    /**
    * Learns a spatial autocorrelation function from features extracted from different positions at various scales
    * of a set of images. Use writeToFile() to save the learned statistics afterwards.
    *
    * To compute the autocorrelation function of two feature planes \$fI\$f and \$fJ\$f, this function leverages the
    * Fourier transform and computes:
    *
    * \f[ \mathcal{F}^{-1} \left ( \overline{\mathcal{F} \left(I \right) } \circ \mathcal{F} \left(J \right) \right ) \f]
    *
    * This is an order of magnitude faster than the method originally used by Hariharan et al., which is provided by
    * learnCovariance_ineff(). Since the Fourier transform implies a cyclic model at the borders of the image, the
    * autocorrelation function computed with this efficient method won't exactly equal the one computed by `learnCovariance_ineff()`.
    * This inaccuracy results in a slight decrease of the performance of models learnt with the efficiently computed statistics
    * (the deviation will be around 5% of the average precision of the other model, e. g. from 20% to 19%). 
    *
    * @note Computing the covariance matrices requires a mean feature vector, which has to be loaded from file
    * or learned using learnMean() in advance.
    *
    * @param[in] imgIt An ImageIterator providing images to learn background statistics from.
    * The iterator will be rewound at the beginning of the process.
    *
    * @param[in] numImages Maximum number of images to learn from. If set to 0, all images provided by the
    * iterator will be used (potentially dangerous when used with infinite iterators like MixedImageIterator!).
    *
    * @param[in] maxOffset Maximum available offset in x or y direction of the autocorrelation function to be learned.
    * Determines the maximum size of the reconstructible covariance matrix, which will be `maxOffset + 1`.
    * In contrast to learnCovariance_ineff(), this parameter has nearly no impact on the required time.
    *
    * @param[in] progressCB Optionally, a callback that is called to populate the progress of the procedure.
    * The first parameter to the callback will be the number of processed images and the second parameter will be equal to `numImages`.
    * For example, the argument list (5, 10) means that the learning is half way done.  
    * The callback may return false to abort the operation. To continue, it must return true.  
    * Note, that the callback will be used only if `numImages` is different from 0.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    */
    void learnCovariance(ImageIterator & imgIt, const unsigned int numImages = 0, const unsigned int maxOffset = 19,
                         ProgressCallback progressCB = NULL, void * cbData = NULL);
    
    /**
    * Learns a spatial autocorrelation function from features extracted from different positions at various scales
    * of a set of images. Use writeToFile() to save the learned statistics afterwards.
    *
    * @note This is the inefficient variant of computing such an autocorrelation function, adapted from the original
    * code of Hariharan et al. It will take very, very long and, though it is provided here for computations with maximum accuracy,
    * one should rather use the efficient learnCovariance() variant, which leverages the Fourier transform.
    *
    * @note Computing the covariance matrices requires a mean feature vector, which has to be loaded from file
    * or learned using learnMean() in advance.
    *
    * @param[in] imgIt An ImageIterator providing images to learn background statistics from.
    * The iterator will be rewound at the beginning of the process.
    *
    * @param[in] numImages Maximum number of images to learn from. If set to 0, all images provided by the
    * iterator will be used (potentially dangerous when used with infinite iterators like MixedImageIterator!).
    *
    * @param[in] maxOffset Maximum available offset in x or y direction of the autocorrelation function to be learned.
    * Determines the maximum size of the reconstructible covariance matrix, which will be `maxOffset + 1`.
    *
    * @param[in] progressCB Optionally, a callback that is called to populate the progress of the procedure.
    * The first parameter to the callback will be the number of processed images and the second parameter will be equal to `numImages`.
    * For example, the argument list (5, 10) means that the learning is half way done.  
    * The callback may return false to abort the operation. To continue, it must return true.  
    * Note, that the callback will be used only if `numImages` is different from 0.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    */
    void learnCovariance_ineff(ImageIterator & imgIt, const unsigned int numImages = 0, const unsigned int maxOffset = 19,
                               ProgressCallback progressCB = NULL, void * cbData = NULL);
    
    
    MeanVector mean; /**< Stationary negative mean of features. */
    
    CovMatrixArray cov; /**< Stationary covariance matrices for each offset as spatial autocorrelation function. */
    
    OffsetArray offsets; /**< Array of (dx, dy) spatial offsets corresponding to the elements of `cov`. */
    
    unsigned int cellSize; /**< Number of pixels per cell in both dimensions used for training this background statistics. */


protected:

    /**
    * Initializes the `offsets` array.
    *
    * @param[in] maxOffset Maximum available offset in x or y direction of the corresponding autocorrelation function.
    * Determines the maximum size of the reconstructible covariance matrix, which will be `maxOffset + 1`.
    */
    void makeOffsetArray(const unsigned int maxOffset);

};

}

#endif
