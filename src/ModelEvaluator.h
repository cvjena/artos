#ifndef ARTOS_MODELEVALUATOR_H
#define ARTOS_MODELEVALUATOR_H

#include <string>
#include <vector>
#include "defs.h"
#include "DPMDetection.h"
#include "ffld/HOGPyramid.h"

namespace ARTOS
{

/**
* Class for evaluating models by testing it against some positive and negative samples.
* Will calculate measures like precision, recall and F-measure and, thus, may be used to
* evaluate a model's performance or to learn an optimal threshold for it.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ModelEvaluator : public DPMDetection
{

public:

    /**
    * Basic performance measures for a specific model using a specific threshold.
    */
    typedef struct {
        double threshold; /**< A detection threshold that rules the number of true and false positives. */
        unsigned int tp; /**< Number of true positives using that threshold. */
        unsigned int fp; /**< Number of false positives using that threshold. */
        unsigned int np; /**< Total number of positive samples. */ 
    } TestResult;
    
    typedef std::vector< std::pair<int, Detection> > SampleDetectionsVector;
    
    typedef FFLD::Mixture * (*LOOFunc)(const FFLD::Mixture * orig, const Sample * sample, const unsigned int objectIndex,
                                       const unsigned int numLeftOut, void * data);
    
    static const unsigned int PRECISION = 1;
    static const unsigned int RECALL = 2;
    static const unsigned int FMEASURE = 4;


    /**
    * Initializes a ModelEvaluator. Multiple models can be added later on using addModel() or addModels().
    *
    * @param[in] overlap Minimum overlap in non maxima suppression.
    *
    * @param[in] padding Amount of zero padding in HOG cells. Must be greater or equal to half the greatest filter dimension.
    *
    * @param[in] interval Number of levels per octave in the HOG pyramid.
    */
    ModelEvaluator(double overlap = 0.5, int padding = 12, int interval = 10)
    : DPMDetection(false, overlap, padding, interval), m_results() { }; 

    /** 
    * Initializes the ModelEvaluator and loads a single model from disk.  
    * This is equivalent to constructing the object with `ModelEvaluator(overlap, padding, interval)`
    * and then calling `addModel("single", modelfile)`.
    *
    * @param[in] modelfile The filename of the model to load.
    *
    * @param[in] overlap Minimum overlap in non maxima suppression.
    *
    * @param[in] padding Amount of zero padding in HOG cells. Must be greater or equal to half the greatest filter dimension.
    *
    * @param[in] interval Number of levels per octave in the HOG pyramid.
    */
    ModelEvaluator(const std::string & modelfile, double overlap = 0.5, int padding = 12, int interval = 10)
    : DPMDetection(modelfile, -100.0, false, overlap, padding, interval), m_results() { }; 
    
    /** 
    * Initializes the ModelEvaluator and adds a single model.
    * This is equivalent to constructing the object with `ModelEvaluator(overlap, padding, interval)`
    * and then calling `addModel("single", model)`.
    *
    * @param[in] model The model to be added as FFLD::Mixture object.
    *
    * @param[in] overlap Minimum overlap in non maxima suppression.
    *
    * @param[in] padding Amount of zero padding in HOG cells. Must be greater or equal to half the greatest filter dimension.
    *
    * @param[in] interval Number of levels per octave in the HOG pyramid.
    */
    ModelEvaluator(const FFLD::Mixture & model, double overlap = 0.5, int padding = 12, int interval = 10)
    : DPMDetection(model, -100.0, false, overlap, padding, interval), m_results() { }; 
    
    /**
    * Returns a reference to a vector with TestResult objects for a specific model computed by testModels().
    *
    * @param[in] modelIndex The index of the model to retrieve test results for.
    * No bounds checking will be performed on this index.
    *
    * @return Reference to a vector of TestResult structures, one for each possible threshold.
    */
    const std::vector<TestResult> & getResults(const unsigned int modelIndex = 0) const
    { return this->m_results[modelIndex]; };
    
    /**
    * Returns a vector with F-measure values for each threshold of a specific model based on the test results.
    * So, testModels() must have been called before.
    *
    * The F-measure is defined as:
    * \f[\frac{(1 + b^2) \cdot precision \cdot recall}{b^2 \cdot precision + recall}\f]
    *
    * @param[in] modelIndex The index of the model to retrieve F-measure values for.
    *
    * @param[in] b The b parameter for F-measure calculation. This is used to weight precision against recall.
    * A value of 2 will rate recall twice as much as precision and a value of 0.5 will put more emphasis on
    * precision than on recall, for example. b must be greater than 0.
    *
    * @return Vector with pairs of thresholds and F-measure values. If modelIndex is out of bounds or testModels()
    * hasn't been called yet, an empty vector will be returned.
    */
    std::vector< std::pair<float, float> > calculateFMeasures(const unsigned int modelIndex = 0, const float b = 1.0f) const;
    
    /**
    * Finds the threshold with the maximum F-measure value for a specific model.
    *
    * @see calculateFMeasures()
    *
    * @param[in] modelIndex The index of the model to retrieve the maximum F-measure value for.
    * No bounds checking will be performed on this index.
    *
    * @param[in] b The b parameter for F-measure calculation. This is used to weight precision against recall.
    * A value of 2 will rate recall twice as much as precision and a value of 0.5 will put more emphasis on
    * precision than on recall, for example. b must be greater than 0.
    *
    * @return Pair of threshold and F-measure value.
    */
    std::pair<float, float> getMaxFMeasure(const unsigned int modelIndex, const float b = 1.0f) const;
    
    /**
    * Computes the *interpolated average precision* for one of the tested models based on the test results.
    * So, testModels() must have been called before.
    *
    * According to the rules of the *PASCAL VOC Challenge*, the interpolated average precision is defined as the
    * area under the *precision-recall curve* `p(r)` in the interval [0,1], where `p(r)` is the maximum precision
    * at all recall levels greater than or equal to `r`.
    *
    * @param[in] modelIndex The index of the model to compute the average precision for.
    *
    * @return Average precision of the given model or 0 if `modelIndex` is out of bounds or testModels() hasn't
    * been called yet.
    */
    float computeAveragePrecision(const unsigned int modelIndex = 0) const;
    
    /**
    * Runs the detector on given images containing the positive samples and, optionally, some negative
    * images containing no instance of the object class to be detected and stores basic measures
    * (number of true and false positives) for each possible threshold for each model. Those results
    * may then be retrieved via getResults().
    *
    * @param[in] positive Vector with pointers to positive samples, given as Sample structures, which the
    * detector will be run against. The `modelAssoc` vector of those structures indicates, which one of the
    * models added to this evaluator (identified by it's index) is expected to detect that object.
    * For all over models, that sample will be ignored.
    *
    * @param[in] maxSamples Maximum number of positive samples to test _each_ model against. Set this to 0 to run the
    * detector against all samples belonging to the respective model.
    *
    * @param[in] negative Pointer to a vector of JPEGImages which are negative samples and, thus, must not
    * contain any object of the class to be detected by the models. This parameter is optional and may be `NULL`.
    *
    * @param[in] granularity Specifies the "resolution" or "precision" of the threshold scale.
    * The distance from one threshold to the next one will be 1/granularity.
    *
    * @param[in] progressCB Optionally, a callback that is called after each run of the detector against a sample.
    * The first parameter to the callback will be the number of samples processed and the second parameter will be
    * the total number of samples to be processed.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @param[in] looFunc A callback for *Leave-one-out-cross-validation* (LOOCV). Before the detector will be run
    * on each sample, that sample and the model belonging to it as well as the index of the object to "leave out"
    * in the `modelAssoc` vector of the sample and, finally, the number of already left out samples will be passed
    * to this function, which has to return a pointer to another model to be used instead. That way, you can remove
    * just that sample from your model.
    * The returned pointer is assumed to be newly allocated by the callback function and will be deleted after use.
    * The callback function may return NULL to leave the original model unchanged.
    *
    * @param[in] looData Will be passed to the `looFunc` callback as last parameter.
    */
    virtual void testModels(const std::vector<Sample*> & positive, unsigned int maxSamples = 0,
                            const std::vector<FFLD::JPEGImage> * negative = NULL,
                            const unsigned int granularity = 100,
                            ProgressCallback progressCB = NULL, void * cbData = NULL,
                            LOOFunc looFunc = NULL, void * looData = NULL);
    
    /**
    * Runs the detector on given images containing the positive samples and, optionally, some negative
    * images containing no instance of the object class to be detected.
    *
    * @param[out] detections A vector which will be filled with detection results. Each result is a pair of an integral
    * value and the actual detection. The integral value specifies, which image the object has been detected on, with
    * positive values being indices of the `positive` vector and negative values like -i meaning the i-th negative sample.
    *
    * @param[in] positive Vector with pointers to positive samples, given as Sample structures, which the
    * detector will be run against. The `modelAssoc` vector of those structures indicates, which one of the
    * models added to this evaluator (identified by it's index) is expected to detect that object.
    * For all over models, that sample will be ignored.
    *
    * @param[in] maxSamples Maximum number of positive samples to test _each_ model against. Set this to 0 to run the
    * detector against all samples belonging to the respective model.
    *
    * @param[in] negative Pointer to a vector of JPEGImages which are negative samples and, thus, must not
    * contain any object of the class to be detected by the models. This parameter is optional and may be `NULL`.
    *
    * @param[in] progressCB Optionally, a callback that is called after each run of the detector against a sample.
    * The first parameter to the callback will be the number of samples processed and the second parameter will be
    * the total number of samples to be processed.
    *
    * @param[in] cbData Will be passed to the `progressCB` callback as third parameter.
    *
    * @param[in] looFunc A callback for *Leave-one-out-cross-validation* (LOOCV). Before the detector will be run
    * on each sample, that sample and the model belonging to it as well as the index of the object to "leave out"
    * in the `modelAssoc` vector of the sample and, finally, the number of already left out samples will be passed
    * to this function, which has to return a pointer to another model to be used instead. That way, you can remove
    * just that sample from your model.
    * The returned pointer is assumed to be newly allocated by the callback function and will be deleted after use.
    * The callback function may return NULL to leave the original model unchanged.
    *
    * @param[in] looData Will be passed to the `looFunc` callback as last parameter.
    *
    * @return A vector with the number of positive detections which are required for a recall of 100% for each model.
    * That is the total number of objects (bounding boxes) on those images.
    */
    virtual std::vector<unsigned int> runDetector(SampleDetectionsVector & detections,
                                                  const std::vector<Sample*> & positive, unsigned int maxSamples = 0,
                                                  const std::vector<FFLD::JPEGImage> * negative = NULL,
                                                  ProgressCallback progressCB = NULL, void * cbData = NULL,
                                                  LOOFunc looFunc = NULL, void * looData = NULL);
    
    /**
    * Writes the test results to a CSV file.
    *
    * @param[in] filename The path of the file to be written.
    *
    * @param[in] modelIndex The index of the model to dump results for.
    * If set to a negative value, test results for all models will be written and, if there is more than just
    * one model, an additional column specifying the model index is prepended to the CSV rows.
    *
    * @param[in] headline If set to true, the first line in the resulting file will specify the names
    * of the columns.
    *
    * @param[in] measures Bit-wise combination of measures to be included in the output. By default, only
    * the basic measures (true positives, false positives, total number of positive samples) will be written.
    * Possible additional measures are: PRECISION, RECALL, FMEASURE (with default parameter b = 1).
    *
    * @param[in] separator The character used to separate the columns in the CSV file.
    *
    * @return Returns true if the file could be written successfully, otherwise false.
    */
    virtual bool dumpTestResults(const std::string & filename, const int modelIndex = -1,
                                 const bool headline = true, const unsigned int measures = 0,
                                 const char separator = ';') const;


protected:

    std::vector< std::vector<TestResult> > m_results; /**< Test results for each model and each threshold. */

};

}

#endif