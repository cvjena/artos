#ifndef ARTOS_MODELLEARNER_H
#define ARTOS_MODELLEARNER_H

#include <string>
#include <vector>
#include <utility>
#include "defs.h"
#include "StationaryBackground.h"
#include "ffld/JPEGImage.h"
#include "ffld/HOGPyramid.h"
#include "ffld/Rectangle.h"

namespace ARTOS
{


struct WHOSample : public Sample
{
    std::vector<FFLD::HOGPyramid::Level> whoFeatures; /**< WHO features of each object in this sample. */
    virtual ~WHOSample() { };
};


/**
* Handles learning of WHO (*Whitened Histogram of Orientations* [Hariharan et. al.]) models using
* Linear Discriminant Analysis (LDA).
*
* Here's our **learning method** in a nutshell: \f$m = \Sigma^{-1} \cdot (\mu_{pos} - \mu_{neg})\f$
*
* Average the HOG (*Histogram of Oriented Gradients* [Dalal & Triggs]) features of each positive sample,
* centre them by subtracting the previously learned negative mean `neg` and decorrelate ("whiten") them with
* the also previously learned covariance matrix `S`.
*
* The typical **work-flow** for using this class is as follows:
*   1. Create a new model learner and pass in background statistics as StationaryBackground object or file name.
*   2. Add some positive samples, i. e. JPEG images of the object to be learned.
*   3. Call learn() to perform the actual learning step.
*   4. (*Optional*) Call optimizeThreshold() to determine an appropriate threshold for the learned model.
*   5. Write the model to file using save().
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ModelLearner
{

protected:

    typedef struct { int width; int height; } Size;


public:

    /**
    * Constructs an empty ModelLearner, which can not be used until some background statistics are given
    * using setBackground().
    */
    ModelLearner()
    : m_verbose(false), m_loocv(true), m_bg(), m_samples(), m_numSamples(0), m_models(), m_thresholds(), m_clusterSizes(), m_normFactors() { };
    
    /**
    * Constructs a new ModelLearner with given background statistics.
    *
    * @param[in] bg The stationary background statistics.
    *
    * @param[in] loocv If set to true, *Leave-one-out-cross-validation* will be performed for threshold optimization.
    * This will increase memory usage, since the WHO features of all samples have to be stored for this, and will slow
    * down threshold optimization as well as the model learning step if only one WHO cluster is used. But it will guarantee
    * that no model is tested against a sample it has been learned from for threshold optimization.
    *
    * @param[in] verbose If set to true, debug and timing information will be printed to stderr.
    */
    ModelLearner(const StationaryBackground & bg, const bool loocv = true, const bool verbose = false)
    : m_verbose(verbose), m_loocv(loocv), m_bg(bg), m_samples(), m_numSamples(0), m_models(), m_thresholds(), m_clusterSizes(), m_normFactors() { };
    
    /**
    * Constructs a new ModelLearner with given background statistics.
    *
    * @param[in] bgFile Path to a file containing background statistics.
    *
    * @param[in] loocv If set to true, *Leave-one-out-cross-validation* will be performed for threshold optimization.
    * This will increase memory usage, since the WHO features of all samples have to be stored for this, and will slow
    * down threshold optimization as well as the model learning step if only one WHO cluster is used. But it will guarantee
    * that no model is tested against a sample it has been learned from for threshold optimization.
    *
    * @param[in] verbose If set to true, debug and timing information will be printed to stderr.
    */
    ModelLearner(const std::string & bgFile, const bool loocv = true, const bool verbose = false)
    : m_verbose(verbose), m_loocv(loocv), m_bg(bgFile), m_samples(), m_numSamples(0), m_models(), m_thresholds(), m_clusterSizes(), m_normFactors() { };
    
    /**
    * Changes the background statistics used by this ModelLearner for centring and whitening.
    *
    * @param[in] bg The new stationary background statistics.
    */
    void setBackground(const StationaryBackground & bg) { this->m_bg = bg; };
    
    /**
    * Changes the background statistics used by this ModelLearner for centring and whitening.
    *
    * @param[in] bgFile Path to a file containing the new background statistics.
    */
    void setBackground(const std::string & bgFile) { this->m_bg.readFromFile(bgFile); };
    
    /**
    * @return The StationaryBackground object used by this ModelLearner for centring and whitening.
    */
    StationaryBackground & getBackground() { return this->m_bg; };
    
    /**
    * @return The number of positive samples added to this ModelLearner before using addPositiveSample()
    * for example. Each bounding box on an image is counted as a single sample.
    */
    unsigned int getNumSamples() const { return this->m_numSamples; };
    
    /**
    * @return A reference to a vector with WHOSample structures representing the positive samples
    * added to this learner with addPositiveSample().
    */
    const std::vector<WHOSample> & getSamples() const { return this->m_samples; };
    
    /**
    * @return The models learned by learn(), which will be an empty vector if learn() hasn't been called yet
    * or has failed.
    */
    const std::vector<FFLD::HOGPyramid::Level> & getModels() { return this->m_models; };
    
    /**
    * @return The thresholds for the learned models determined by optimizeThreshold(), which will be 0 if
    * optimizeThreshold() hasn't been called yet or has failed.
    */
    const std::vector<float> & getThresholds() const { return this->m_thresholds; };
    
    /**
    * @return Reference to a vector with the number of samples, each model computed by learn() has been learned from.
    * An empty vector will be returned if no model has been learned yet.
    */
    const std::vector<unsigned int> & getClusterSizes() const { return this->m_clusterSizes; };
    
    /**
    * @return The factors `f` which have been used to normalize each model by `w = w/f`.
    * An empty vector will be returned if no model has been learned yet.
    */
    const std::vector<FFLD::HOGPyramid::Scalar> & getNormFactors() const { return this->m_normFactors; };
    
    /**
    * Resets this learner to it's initial state and makes it forget all learned models, thresholds and added samples.
    * Only the given background statistics will be left intact.
    */
    virtual void reset();
    
    /**
    * Adds a positive sample to learn from given by an image and a bounding box around the object on that image.
    *
    * @param[in] sample The image which contains the object.
    *
    * @param[in] boundingBox The bounding box around the object on the given image.
    * If the bounding box is empty (a rectangle with zero area), the entire image will be used as positive sample.
    */
    virtual void addPositiveSample(const FFLD::JPEGImage & sample, const FFLD::Rectangle & boundingBox);
    
    /**
    * Adds multiple positive samples to learn from on the same image given by bounding boxes around the objects.
    *
    * @param[in] sample The image which contains the objects.
    *
    * @param[in] boundingBoxes Vector of bounding boxes around each object on the given image. If just one of the
    * bounding boxes is empty (a rectangle with zero area) only one sample will be added, which is the entire image.
    */
    virtual void addPositiveSample(const FFLD::JPEGImage & sample, const std::vector<FFLD::Rectangle> & boundingBoxes);
    
    /**
    * Performs the actual learning step and stores the resulting models, so that they can be retrieved later on by getModels().
    *
    * Optionally, clustering can be performed before learning, first by aspect ratio, then by WHO features. A separate model
    * will be learned for each cluster. Thus, the maximum number of learned models will be `maxAspectClusters * maxWHOClusters`.
    *
    * The learned models will be provided with estimated thresholds, according to the following formula:
    * \f[\frac{\mu_0^T \Sigma^-1 \mu_0 - \mu_1^T \Sigma^-1 \mu_1}{2}\f]
    * That formula lacks an additive term of \f$\ln \left ( \frac{\varphi}{\varphi - 1} \right )\f$ involving a-priori probabilities
    * and, thus, is not optimal. Hence, you'll probably want to call optimizeThreshold() after learning the models.
    *
    * @param[in] maxAspectClusters Maximum number of clusters to form by the aspect ratio of the samples.
    *
    * @param[in] maxWHOClusters Maximum number of clusters to form by the WHO feature vectors of the samples
    * of a single aspect ratio cluster.
    *
    * @param[in] progressCB Optionally, a callback that is called to populate the progress of the procedure.
    * The first parameter to the callback will be the current step and the second parameter will be the total number
    * of steps. For example, the argument list (5, 10) means that the learning is half way done.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @return True if some models could be learned, false if learning failed completely.
    *
    * @note Background statistics must have been set and some samples must have been added before calling learn().
    */
    virtual bool learn(const unsigned int maxAspectClusters = 1, const unsigned int maxWHOClusters = 1,
                       ProgressCallback progressCB = NULL, void * cbData = NULL);
    
    /**
    * Finds the optimal thresholds for the models learned previously with learn() by testing them against the
    * positive samples and, optionally, some additional negative samples to maximize the F-measure of each model:
    * \f[\frac{(1 + b^2) \cdot \mbox{precision} \cdot \mbox{recall}}{b^2 \cdot \mbox{precision} + \mbox{recall}}\f]
    *
    * @param[in] maxPositive Maximum number of positive samples to test the models against. Set this to 0
    * to run the detector against all samples.
    *
    * @param[in] negative Pointer to a vector of JPEGImages which are negative samples and, thus, must not
    * contain any object of the class the learned model is trained for. Any detection on these images will
    * be considered as a false positive. This parameter is optional and may be `NULL`.
    *
    * @param[in] b The b parameter for F-measure calculation. This is used to weight precision against recall.
    * A value of 2 will rate recall twice as much as precision and a value of 0.5 will put more emphasis on
    * precision than on recall, for example. b must be greater than 0.
    *
    * @param[in] progressCB Optionally, a callback that is called after each run of the detector against a sample.
    * The first parameter to the callback will be the number of samples processed and the second parameter will be
    * the total number of samples to be processed. For example, the argument list (5, 10) means that the optimization
    * is half way done.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @return Vector with the calculated optimal thresholds, which will be stored internally and can also be
    * obtained later on using getThreshold().
    */
    virtual const std::vector<float> & optimizeThreshold(const unsigned int maxPositive = 0,
                                                         const std::vector<FFLD::JPEGImage> * negative = NULL,
                                                         const float b = 1.0f,
                                                         ProgressCallback progressCB = NULL, void * cbData = NULL);
    
    virtual const std::vector<float> & optimizeThresholdCombination(const unsigned int maxPositive = 0,
                                                         const std::vector<FFLD::JPEGImage> * negative = NULL,
                                                         int mode = 1, const float b = 1.0f,
                                                         ProgressCallback progressCB = NULL, void * cbData = NULL);
    
    /**
    * Writes the models learned by learn() to a file.
    *
    * @param[in] filename The path of the file to be written.
    *
    * @param[in] addToMixture If set to true and `filename` does already exist and is a valid mixture, the new models
    * will be appended to that mixture. If set to false, an existing file will be truncated and a new mixture will be created.
    *
    * @return True if the model file could be written successfully, false if no model has been learned yet or the specified file
    * is not accessible for writing.
    */
    virtual bool save(const std::string & filename, const bool addToMixture = true) const;


protected:

    bool m_verbose; /**< Specifies if debugging and timing information will be output to stderr. */

    bool m_loocv; /**< Specifies if *Leave-one-out-cross-validation* shall be used for threshold optimization. */

    StationaryBackground m_bg; /**< Stationary background statistics (to obtain negative mean and covariance). */

    std::vector<WHOSample> m_samples; /**< Positive samples consisting of images with given bounding boxes. */

    unsigned int m_numSamples; /**< Number of positive samples added (i. e. the number of bounding boxes summed over m_samples). */
    
    std::vector<FFLD::HOGPyramid::Level> m_models; /**< The models learned by `learn`. */
    
    std::vector<float> m_thresholds; /**< Optimal thresholds for the learned models computed by `optimizeThreshold`. */
    
    std::vector<unsigned int> m_clusterSizes; /**< Number of samples belonging to each model computed by `learn`. */
    
    std::vector<FFLD::HOGPyramid::Scalar> m_normFactors; /**< Vector with factors, each model has been divided by for normalization. */
    
    
    /**
    * Computes an optimal common number of cells in x and y direction for given widths and heights of samples.
    *
    * @param[in] widths Vector with the widths of the samples.
    *
    * @param[in] heights Vector with the heights of the samples, corresponding to `width`.
    *
    * @return Returns the optimal cell number in x and y direction as a struct with `width` and `height` fields.
    *
    * @note Background statistics must have been set for this function to work.
    */
    virtual Size computeOptimalCellNumber(const std::vector<int> & widths, const std::vector<int> & heights);

};

}

#endif