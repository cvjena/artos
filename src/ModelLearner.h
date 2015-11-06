#ifndef ARTOS_MODELLEARNER_H
#define ARTOS_MODELLEARNER_H

#include "ModelLearnerBase.h"
#include "StationaryBackground.h"

namespace ARTOS
{


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
class ModelLearner : public ModelLearnerBase
{

public:

    /**
    * Constructs an empty ModelLearner, which can not be used until some background statistics are given
    * using setBackground().
    */
    ModelLearner()
    : ModelLearnerBase(), m_loocv(true), m_bg(), m_normFactors() { };
    
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
    : ModelLearnerBase(verbose), m_loocv(loocv), m_bg(bg), m_normFactors() { };
    
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
    : ModelLearnerBase(verbose), m_loocv(loocv), m_bg(bgFile), m_normFactors() { };
    
    virtual ~ModelLearner() { this->reset(); };
    
    /**
    * Changes the background statistics used by this ModelLearner for centering and whitening.
    *
    * @param[in] bg The new stationary background statistics.
    */
    void setBackground(const StationaryBackground & bg) { this->m_bg = bg; };
    
    /**
    * Changes the background statistics used by this ModelLearner for centering and whitening.
    *
    * @param[in] bgFile Path to a file containing the new background statistics.
    */
    void setBackground(const std::string & bgFile) { this->m_bg.readFromFile(bgFile); };
    
    /**
    * @return The StationaryBackground object used by this ModelLearner for centering and whitening.
    */
    StationaryBackground & getBackground() { return this->m_bg; };
    
    /**
    * @return The factors `f` which have been used to normalize each model by `w = w/f`.
    * An empty vector will be returned if no model has been learned yet.
    */
    const std::vector<FeatureExtractor::Scalar> & getNormFactors() const { return this->m_normFactors; };
    
    /**
    * Resets this learner to it's initial state and makes it forget all learned models, thresholds and added samples.
    * Only the given background statistics will be left intact.
    */
    virtual void reset();
    
    /**
    * Adds a positive sample to learn from given by a SynsetImage object.
    *
    * @param[in] sample The SynsetImage object to be added as positive sample.
    *
    * @return True if the sample has been added, otherwise false.
    */
    virtual bool addPositiveSample(const SynsetImage & sample);
    
    /**
    * Adds a positive sample to learn from given by a SynsetImage object.
    *
    * @param[in] sample The SynsetImage object to be added as positive sample.
    *
    * @return True if the sample has been added, otherwise false.
    */
    virtual bool addPositiveSample(SynsetImage && sample);
    
    /**
    * Adds a positive sample to learn from given by an image and a bounding box around the object on that image.
    *
    * @param[in] sample The image which contains the object.
    *
    * @param[in] boundingBox The bounding box around the object on the given image.
    * If the bounding box is empty (a rectangle with zero area), the entire image will be used as positive sample.
    *
    * @return True if the sample has been added, otherwise false.
    */
    virtual bool addPositiveSample(const FFLD::JPEGImage & sample, const FFLD::Rectangle & boundingBox);
    
    /**
    * Adds a positive sample to learn from given by an image and a bounding box around the object on that image.
    *
    * @param[in] sample The image which contains the object.
    *
    * @param[in] boundingBox The bounding box around the object on the given image.
    * If the bounding box is empty (a rectangle with zero area), the entire image will be used as positive sample.
    *
    * @return True if the sample has been added, otherwise false.
    */
    virtual bool addPositiveSample(FFLD::JPEGImage && sample, const FFLD::Rectangle & boundingBox);
    
    /**
    * Adds multiple positive samples to learn from on the same image given by bounding boxes around the objects.
    *
    * @param[in] sample The image which contains the objects.
    *
    * @param[in] boundingBoxes Vector of bounding boxes around each object on the given image. If just one of the
    * bounding boxes is empty (a rectangle with zero area) only one sample will be added, which is the entire image.
    *
    * @return True if the sample has been added, otherwise false.
    */
    virtual bool addPositiveSample(const FFLD::JPEGImage & sample, const std::vector<FFLD::Rectangle> & boundingBoxes);
    
    /**
    * Adds multiple positive samples to learn from on the same image given by bounding boxes around the objects.
    *
    * @param[in] sample The image which contains the objects.
    *
    * @param[in] boundingBoxes Vector of bounding boxes around each object on the given image. If just one of the
    * bounding boxes is empty (a rectangle with zero area) only one sample will be added, which is the entire image.
    *
    * @return True if the sample has been added, otherwise false.
    */
    virtual bool addPositiveSample(FFLD::JPEGImage && sample, const std::vector<FFLD::Rectangle> & boundingBoxes);
    
    /**
    * Finds the optimal combination of thresholds for the models learned previously with learn() by testing them
    * against the positive samples and, optionally, some additional negative samples to maximize the F-measure
    * of the combined model mixture:
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
    * The callback may return false to abort the operation. To continue, it must return true.
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


protected:

    bool m_loocv; /**< Specifies if *Leave-one-out-cross-validation* shall be used for threshold optimization. */

    StationaryBackground m_bg; /**< Stationary background statistics (to obtain negative mean and covariance). */
    
    std::vector<FeatureExtractor::Scalar> m_normFactors; /**< Vector with factors, each model has been divided by for normalization. */


    /**
    * Performs the actual learning step and stores the resulting models, so that they can be retrieved later on by getModels().
    *
    * The learned models will be provided with estimated thresholds, according to the following formula:
    * \f[\frac{\mu_0^T \Sigma^-1 \mu_0 - \mu_1^T \Sigma^-1 \mu_1}{2}\f]
    * That formula lacks an additive term of \f$\ln \left ( \frac{\varphi}{\varphi - 1} \right )\f$ involving a-priori probabilities
    * and, thus, is not optimal. Hence, you'll probably want to call optimizeThreshold() after learning the models.
    *
    * @param[in] aspectClusterAssignment Vector which associates indexes of samples with aspect ratio cluster numbers.
    *
    * @param[in] samplesPerAspectCluster Vector with the number of samples in each aspect ratio cluster.
    *
    * @param[in] cellNumbers Vector with the preferred model dimensions for each aspect ratio cluster.
    *
    * @param[in] maxWHOClusters Maximum number of clusters to form by the WHO feature vectors of the samples
    * of a single aspect ratio cluster.
    *
    * @param[in] progressCB Optionally, a callback that is called to populate the progress of the procedure.
    * The first parameter to the callback will be the current step and the second parameter will be the total number
    * of steps. For example, the argument list (5, 10) means that the learning is half way done.  
    * The value returned by the callback will be ignored.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @return True if some models could be learned, false if learning failed completely.
    *
    * @note Background statistics must have been set and some samples must have been added before calling learn().
    */
    virtual void m_learn(Eigen::VectorXi & aspectClusterAssignment, std::vector<int> & samplesPerAspectCluster, std::vector<Size> & cellNumbers,
                         const unsigned int maxWHOClusters, ProgressCallback progressCB, void * cbData);
    
    
    /**
    * Called by learn() before anything is done to initialize the model learner.
    *
    * @return Returns true if learning may be performed or false if the model learner is not ready.
    */
    virtual bool learn_init();

};

}

#endif