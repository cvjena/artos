#ifndef ARTOS_MODELLEARNERBASE_H
#define ARTOS_MODELLEARNERBASE_H

#include <string>
#include <vector>
#include "defs.h"
#include "libartos_def.h"
#include "FeatureExtractor.h"
#include "SynsetImage.h"
#include "JPEGImage.h"
#include "Rectangle.h"

namespace ARTOS
{


/**
* Abstract base class for model learners.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ModelLearnerBase
{

public:

    /**
    * Default constructor.
    */
    ModelLearnerBase()
    : m_featureExtractor(FeatureExtractor::defaultFeatureExtractor()),
      m_verbose(false), m_samples(), m_numSamples(0), m_models(), m_thresholds(), m_clusterSizes() { };
    
    /**
    * Constructs a new ModelLearnerBase.
    *
    * @param[in] featureExtractor A shared pointer to the feature extractor to be used by the new model learner.
    *
    * @param[in] verbose If set to true, debug and timing information will be printed to stderr.
    */
    ModelLearnerBase(const std::shared_ptr<FeatureExtractor> & featureExtractor, const bool verbose = false)
    : m_featureExtractor((featureExtractor) ? featureExtractor : FeatureExtractor::defaultFeatureExtractor()),
      m_verbose(verbose), m_samples(), m_numSamples(0), m_models(), m_thresholds(), m_clusterSizes() { };
    
    virtual ~ModelLearnerBase() {};
    
    /**
    * @return The number of positive samples added to this model learner using addPositiveSample()
    * for example. Each bounding box on an image is counted as a single sample.
    */
    virtual unsigned int getNumSamples() const { return this->m_numSamples; };
    
    /**
    * @return A reference to a vector with Sample structures representing the positive samples
    * added to this learner with addPositiveSample().
    */
    virtual const std::vector<Sample> & getSamples() const { return this->m_samples; };
    
    /**
    * @return The models learned by learn(), which will be an empty vector if learn() hasn't been called yet
    * or has failed.
    */
    virtual const std::vector<FeatureMatrix> & getModels() { return this->m_models; };
    
    /**
    * @return The thresholds for the learned models determined by optimizeThreshold(), which will be 0 if
    * optimizeThreshold() hasn't been called yet or has failed.
    */
    virtual const std::vector<float> & getThresholds() const { return this->m_thresholds; };
    
    /**
    * @return Reference to a vector with the number of samples, each model computed by learn() has been learned from.
    * An empty vector will be returned if no model has been learned yet.
    */
    virtual const std::vector<unsigned int> & getClusterSizes() const { return this->m_clusterSizes; };
    
    /**
    * @return Returns a shared pointer to the FeatureExtractor used by this model learner.
    */
    virtual std::shared_ptr<FeatureExtractor> getFeatureExtractor() const { return this->m_featureExtractor; };
    
    /**
    * Changes the feature extractor used by this model learner.
    *
    * @param[in] featureExtractor A shared pointer to the new feature extractor to be used by this model learner.
    * If a nullptr is given, the default feature extractor will be used.
    *
    * @note This function also clears all models and thresholds learned so far.
    */
    virtual void setFeatureExtractor(const std::shared_ptr<FeatureExtractor> & featureExtractor);
    
    /**
    * Resets this learner to it's initial state and makes it forget all learned models, thresholds and added samples.
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
    virtual bool addPositiveSample(const JPEGImage & sample, const Rectangle & boundingBox);
    
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
    virtual bool addPositiveSample(JPEGImage && sample, const Rectangle & boundingBox);
    
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
    virtual bool addPositiveSample(const JPEGImage & sample, const std::vector<Rectangle> & boundingBoxes);
    
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
    virtual bool addPositiveSample(JPEGImage && sample, const std::vector<Rectangle> & boundingBoxes);
    
    /**
    * Performs the actual learning step and stores the resulting models, so that they can be retrieved later on by getModels().
    *
    * Optionally, clustering can be performed before learning, first by aspect ratio, then by features. A separate model
    * will be learned for each cluster. Thus, the maximum number of learned models will be `maxAspectClusters * maxWHOClusters`.
    *
    * @param[in] maxAspectClusters Maximum number of clusters to form by the aspect ratio of the samples.
    *
    * @param[in] maxFeatureClusters Maximum number of clusters to form by the feature vectors of the samples
    * of a single aspect ratio cluster.
    *
    * @param[in] progressCB Optionally, a callback that is called to populate the progress of the procedure.
    * The first parameter to the callback will be the current step and the second parameter will be the total number
    * of steps. For example, the argument list (5, 10) means that the learning is half way done.  
    * The value returned by the callback will be ignored.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @return Returns ARTOS_RES_OK if some models could be learned or an error code if learning failed completely.
    */
    virtual int learn(const unsigned int maxAspectClusters = 1, const unsigned int maxFeatureClusters = 1,
                      ProgressCallback progressCB = NULL, void * cbData = NULL);
    
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
                                                         const std::vector<JPEGImage> * negative = NULL,
                                                         const float b = 1.0f,
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

    std::shared_ptr<FeatureExtractor> m_featureExtractor; /**< The feature extractor used by this model learner. */

    bool m_verbose; /**< Specifies if debugging and timing information will be output to stderr. */

    std::vector<Sample> m_samples; /**< Positive samples consisting of images with given bounding boxes. */

    unsigned int m_numSamples; /**< Number of positive samples added (i. e. the number of bounding boxes summed over m_samples). */
    
    std::vector<FeatureMatrix> m_models; /**< The models learned by `learn`. */
    
    std::vector<float> m_thresholds; /**< Optimal thresholds for the learned models computed by `optimizeThreshold`. */
    
    std::vector<unsigned int> m_clusterSizes; /**< Number of samples belonging to each model computed by `learn`. */
    
    
    /**
    * This function is called by learn() to perform the actual learning. Implement it in derived classes.
    *
    * @param[in] aspectClusterAssignment Vector which associates indexes of samples with aspect ratio cluster numbers.
    *
    * @param[in] samplesPerAspectCluster Vector with the number of samples in each aspect ratio cluster.
    *
    * @param[in] cellNumbers Vector with the preferred model dimensions for each aspect ratio cluster.
    *
    * @param[in] maxFeatureClusters Maximum number of clusters to form by the feature vectors of the samples
    * of a single aspect ratio cluster.
    *
    * @param[in] progressCB Optionally, a callback that is called to populate the progress of the procedure.
    * The first parameter to the callback will be the current step and the second parameter will be the total number
    * of steps. For example, the argument list (5, 10) means that the learning is half way done.  
    * The value returned by the callback will be ignored.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @return Returns ARTOS_RES_OK if learning has been successful, otherwise an error code will be returned.
    */
    virtual int m_learn(Eigen::VectorXi & aspectClusterAssignment, std::vector<int> & samplesPerAspectCluster, std::vector<Size> & cellNumbers,
                        const unsigned int maxFeatureClusters, ProgressCallback progressCB, void * cbData) =0;
    
    
    /**
    * Called by learn() before anything is done to initialize the model learner.
    *
    * @return Returns ARTOS_RES_OK if learning may be performed or an error code if the model learner is not ready.
    */
    virtual int learn_init();
    
    
    /**
    * Specifies the highest possible size of a model learned by this model learner.
    * This information will be used to compute the actual model dimensions.
    *
    * @return Returns the maximum size of a model which can be learned by this model learner
    * in each dimension, given in cells. If any dimension is 0, its extension won't be limited.
    */
    virtual Size maximumModelSize() const { return Size(); };
    
    /**
    * Used by addPositiveSample() to initialize a given sample based on its m_simg field.
    *
    * @param[in,out] s The sample to be set up.
    */
    virtual void initSampleFromSynsetImage(Sample & s);

};

}

#endif