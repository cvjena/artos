#ifndef ARTOS_MODELLEARNERBASE_H
#define ARTOS_MODELLEARNERBASE_H

#include <string>
#include <vector>
#include "defs.h"
#include "FeatureExtractor.h"
#include "SynsetImage.h"
#include "ffld/JPEGImage.h"
#include "ffld/Rectangle.h"

namespace ARTOS
{


/**
* Abstract base class for model learners.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ModelLearnerBase
{

protected:

    typedef struct { int width; int height; } Size;


public:

    /**
    * Default constructor.
    */
    ModelLearnerBase()
    : m_verbose(false), m_samples(), m_numSamples(0), m_models(), m_thresholds(), m_clusterSizes() { };
    
    /**
    * Constructs a new ModelLearnerBase.
    *
    * @param[in] verbose If set to true, debug and timing information will be printed to stderr.
    */
    ModelLearnerBase(const bool verbose)
    : m_verbose(verbose), m_samples(), m_numSamples(0), m_models(), m_thresholds(), m_clusterSizes() { };
    
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
    virtual const std::vector<FeatureExtractor::FeatureMatrix> & getModels() { return this->m_models; };
    
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
    * @return True if some models could be learned, false if learning failed completely.
    */
    virtual bool learn(const unsigned int maxAspectClusters = 1, const unsigned int maxFeatureClusters = 1,
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
                                                         const std::vector<FFLD::JPEGImage> * negative = NULL,
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

    bool m_verbose; /**< Specifies if debugging and timing information will be output to stderr. */

    std::vector<Sample> m_samples; /**< Positive samples consisting of images with given bounding boxes. */

    unsigned int m_numSamples; /**< Number of positive samples added (i. e. the number of bounding boxes summed over m_samples). */
    
    std::vector<FeatureExtractor::FeatureMatrix> m_models; /**< The models learned by `learn`. */
    
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
    */
    virtual void m_learn(Eigen::VectorXi & aspectClusterAssignment, std::vector<int> & samplesPerAspectCluster, std::vector<Size> & cellNumbers,
                         const unsigned int maxFeatureClusters, ProgressCallback progressCB, void * cbData) =0;
    
    
    /**
    * Called by learn() before anything is done to initialize the model learner.
    *
    * @return Returns true if learning may be performed or false if the model learner is not ready.
    */
    virtual bool learn_init();
    
    
    /**
    * Computes an optimal common number of cells in x and y direction for given widths and heights of samples.
    *
    * @param[in] widths Vector with the widths of the samples.
    *
    * @param[in] heights Vector with the heights of the samples, corresponding to `width`.
    *
    * @return Returns the optimal cell number in x and y direction as a struct with `width` and `height` fields.
    *
    * @todo The implementation of this function is a legacy from the original WHO code and depends on the assumption
    * of having a cell size of 8 pixels and a maximum allowed model dimension of 20x20 cells.
    * It should be modified to be more generally applicable, so that it can be used with other feature extractors.
    */
    virtual Size computeOptimalCellNumber(const std::vector<int> & widths, const std::vector<int> & heights);
    
    /**
    * Used by addPositiveSample() to initialize a given sample based on its m_simg field.
    *
    * @param[in,out] s The sample to be set up.
    */
    virtual void initSampleFromSynsetImage(Sample & s);

};

}

#endif