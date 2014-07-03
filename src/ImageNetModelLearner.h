#ifndef ARTOS_IMAGENETMODELLEARNER_H
#define ARTOS_IMAGENETMODELLEARNER_H

#include <set>
#include "ModelLearner.h"
#include "ImageRepository.h"
#include "Synset.h"

namespace ARTOS
{

/**
* Handles learning of WHO (*Whitened Histogram of Orientations* [Hariharan et. al.]) models using
* Linear Discriminant Analysis (LDA) with positive and negative samples taken from an image repository.
*
* See the ModelLearner class for a short overview of our learning method.
*
* The typical **work-flow** for using this class is as follows:
*   1. Create a new ImageNetModelLearner and pass in and image repository and background statistics
*      as StationaryBackground object or file name.
*   2. Add positive samples from a specific synset using addPositiveSamplesFromSynset().
*   3. Call learn() to perform the actual learning step.
*   4. (*Optional*) Call optimizeThreshold() to determine an appropriate threshold for the learned model.
*   5. Write the model to file using save().
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ImageNetModelLearner : public ModelLearner
{

public:

    /**
    * Constructs an empty ImageNetModelLearner, which can not be used until some background statistics and an
    * image repository are given using setBackground() and setRepository().
    */
    ImageNetModelLearner() : ModelLearner(), m_repo(""), m_addedSynsets() { };
    
    /**
    * Constructs a new ImageNetModelLearner for a given image repository with given background statistics.
    *
    * @param[in] bg The stationary background statistics.
    *
    * @param[in] repo The image repository to take samples from.
    *
    * @param[in] loocv If set to true, *Leave-one-out-cross-validation* will be performed for threshold optimization.
    * This will increase memory usage, since the WHO features of all samples have to be stored for this, and will slow
    * down threshold optimization as well as the model learning step if only one WHO cluster is used. But it will guarantee
    * that no model is tested against a sample it has been learned from for threshold optimization.
    *
    * @param[in] verbose If set to true, debug and timing information will be printed to stderr.
    */
    ImageNetModelLearner(const StationaryBackground & bg, const ImageRepository & repo, const bool loocv = true, const bool verbose = false)
    : ModelLearner(bg, loocv, verbose), m_repo(repo), m_addedSynsets() { };
    
    /**
    * Constructs a new ImageNetModelLearner for a given image repository with given background statistics.
    *
    * @param[in] bgFile Path to a file containing background statistics.
    *
    * @param[in] repoDirectory Path of the image repository directory.
    *
    * @param[in] loocv If set to true, *Leave-one-out-cross-validation* will be performed for threshold optimization.
    * This will increase memory usage, since the WHO features of all samples have to be stored for this, and will slow
    * down threshold optimization as well as the model learning step if only one WHO cluster is used. But it will guarantee
    * that no model is tested against a sample it has been learned from for threshold optimization.
    *
    * @param[in] verbose If set to true, debug and timing information will be printed to stderr.
    */
    ImageNetModelLearner(const std::string & bgFile, const std::string & repoDirectory, const bool loocv = true, const bool verbose = false)
    : ModelLearner(bgFile, loocv, verbose), m_repo(repoDirectory), m_addedSynsets() { };
    
    /**
    * @return A reference to the image repository used by this learner.
    */
    const ImageRepository & getRepository() const { return this->m_repo; };
    
    /**
    * Changes the image repository, this learner takes samples from.
    *
    * @param[in] repo The new image repository.
    */
    void setRepository(const ImageRepository & repo) { this->m_repo = repo; };
    
    /**
    * Changes the image repository, this learner takes samples from.
    *
    * @param[in] repoDirectory Path of the new image repository directory.
    */
    void setRepository(const std::string & repoDirectory) { this->m_repo = ImageRepository(repoDirectory); };
    
    /**
    * @return A reference to a set with the IDs of synsets which positive samples have been added from.
    */
    const std::set<std::string> & getAddedSynsets() { return this->m_addedSynsets; };
    
    /**
    * Resets this learner to it's initial state and makes it forget all learned models, thresholds and added samples.
    * Only the given background statistics and the image repository will be left intact.
    */
    virtual void reset();
    
    /**
    * Extracts positive samples from a given synset using bounding box annotation data.
    *
    * @param[in] synsetId The id of the synset to take positive samples from (e. g. 'n02119789').
    *
    * @param[in] maxSamples Maximum number of images to extract from the synset. Set this to 0 to
    * extract all images.
    *
    * @return The number of added samples (in terms of object instances).
    *
    * @note An image repository must have been set before calling this function.
    */
    unsigned int addPositiveSamplesFromSynset(const std::string & synsetId, const unsigned int maxSamples = 0);
    
    /**
    * Extracts positive samples from a given synset using bounding box annotation data.
    *
    * @param[in] synset The synset to take positive samples from.
    *
    * @param[in] maxSamples Maximum number of images to extract from the synset. Set this to 0 to
    * extract all images.
    *
    * @return The number of added samples (in terms of object instances).
    *
    * @note An image repository must have been set before calling learnFromSynset().
    */
    virtual unsigned int addPositiveSamplesFromSynset(const Synset & synset, const unsigned int maxSamples = 0);
    
    /**
    * Finds the optimal combination of thresholds for the models learned previously with learn() by testing them
    * against the positive samples and, optionally, some additional negative samples to maximize the F-measure
    * of the combined model mixture:
    * \f[\frac{(1 + b^2) \cdot \mbox{precision} \cdot \mbox{recall}}{b^2 \cdot \mbox{precision} + \mbox{recall}}\f]
    *
    * @param[in] maxPositive Maximum number of positive samples to test the models against. Set this to 0
    * to run the detector against all samples.
    *
    * @param[in] numNegative Number of negative samples taken from different synsets of the image repository.
    * Every detection on one of these images will be considered as a false positive.
    *
    * @param[in] b The b parameter for F-measure calculation. This is used to weight precision against recall.
    * A value of 2 will rate recall twice as much as precision and a value of 0.5 will put more emphasis on
    * precision than on recall, for example. b must be greater than 0.
    *
    * @param[in] progressCB Optionally, a callback that is called after each run of the detector against a sample.
    * The first parameter to the callback will be the number of samples processed and the second parameter will be
    * the total number of samples to be processed. For example, the argument list (5, 10) means that the optimization
    * is half way done.  
    * The callback may return false to abort the operation. To continue, it must return true.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @return Vector with the calculated optimal thresholds, which will be stored internally and can also be
    * obtained later on using getThreshold().
    */
    virtual const std::vector<float> & optimizeThreshold(const unsigned int maxPositive = 0,
                                                         const unsigned int numNegative = 0,
                                                         const float b = 1.0f,
                                                         ProgressCallback progressCB = NULL, void * cbData = NULL);


protected:

    ImageRepository m_repo;
    
    std::set<std::string> m_addedSynsets; /**< Set with IDs of synsets which samples have been added from. */

};

}

#endif