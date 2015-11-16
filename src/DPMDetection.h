#ifndef ARTOS_DPMDETECTION_H
#define ARTOS_DPMDETECTION_H

#include <string>
#include <map>

#include "libartos_def.h"
#include "Mixture.h"
#include "Patchwork.h"
#include "JPEGImage.h"

namespace ARTOS
{

/**
* Single detection result, returned by DPMDetection::detect().
* Position and size of the rectangle specify the bounding box of the object.
*/
struct Detection : public Rectangle
{
    FeatureScalar score; /**< The detection score. */
    double scale; /**< The scale of the image where the object has been detected. */
    int x; /**< The x coordinate of the detection on the scaled sample. */
    int y; /**< The y coordinate of the detection on the scaled sample. */
    std::string classname; /**< The name of the detected class. */
    std::string synsetId; /**< Optionally, the ID of the ImageNet synset associated with the detected class. */
    unsigned int modelIndex; /**< The index of the model which caused the detection (for internal use). */
    
    Detection() : score(0), scale(0), x(0), y(0)
    {
    }
    
    Detection(FeatureScalar score, double scale, int x, int y, Rectangle bndbox,
              const std::string & classname, const std::string & synsetId = "", const unsigned int modelIndex = 0) :
    Rectangle(bndbox), score(score), scale(scale), x(x), y(y), classname(classname), synsetId(synsetId), modelIndex(modelIndex)
    {
    }
    
    /** Used for sorting the detection results in descending order. */
    bool operator<(const Detection & detection) const
    {
        return score > detection.score;
    }
};

/**
* Class for fast detection of objects on images using deformable part models, based on the FFLD library.
* @author Erik Rodner
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class DPMDetection
{

public:

    /**
    * Initialize detector. Multiple models can be added later on using addModel() or addModels().
    *
    * @param[in] verbose If set to true, debug and timing information will be logged to stderr.
    *
    * @param[in] overlap Minimum overlap in non maxima suppression.
    *
    * @param[in] interval Number of levels per octave in the feature pyramid.
    */
    DPMDetection ( bool verbose = false, double overlap = 0.5, int interval = 10 ); 

    /** 
    * Initializes the detector and loads a single model from disk.
    *
    * This is equivalent to constructing the detector with `DPMDetection(verbose, overlap, interval)`
    * and then calling `addModel("single", modelfile, threshold)`.
    *
    * @param[in] modelfile The filename of the model to load.
    *
    * @param[in] threshold The detection threshold for the model.
    *
    * @param[in] verbose If set to true, debug and timing information will be logged to stderr.
    *
    * @param[in] overlap Minimum overlap in non maxima suppression.
    *
    * @param[in] interval Number of levels per octave in the feature pyramid.
    */
    DPMDetection ( const std::string & modelfile, double threshold = 0.8, bool verbose = false, double overlap = 0.5, int interval = 10 );
    
    /** 
    * Initializes the detector and adds a single model.
    *
    * This is equivalent to constructing the detector with `DPMDetection(verbose, overlap, interval)`
    * and then calling `addModel("single", model, threshold)`.
    *
    * @param[in] model The model to be added as Mixture object.
    *
    * @param[in] threshold The detection threshold for the model.
    *
    * @param[in] verbose If set to true, debug and timing information will be logged to stderr.
    *
    * @param[in] overlap Minimum overlap in non maxima suppression.
    *
    * @param[in] interval Number of levels per octave in the feature pyramid.
    */
    DPMDetection ( const Mixture & model, double threshold = 0.8, bool verbose = false, double overlap = 0.5, int interval = 10 );
    
    /** 
    * Initializes the detector and adds a single model.
    *
    * This is equivalent to constructing the detector with `DPMDetection(verbose, overlap, interval)`
    * and then calling `addModel("single", model, threshold)`.
    *
    * @param[in] model The model to be added as Mixture object, whose contents will be moved.
    *
    * @param[in] threshold The detection threshold for the model.
    *
    * @param[in] verbose If set to true, debug and timing information will be logged to stderr.
    *
    * @param[in] overlap Minimum overlap in non maxima suppression.
    *
    * @param[in] interval Number of levels per octave in the feature pyramid.
    */
    DPMDetection ( Mixture && model, double threshold = 0.8, bool verbose = false, double overlap = 0.5, int interval = 10 );

    ~DPMDetection();

    /**
    * Detects objects in a given image which match one of the models added before using addModel() or addModels().
    *
    * @param[in] image The image.
    *
    * @param[out] detections A vector that will receive information about the detected objects.
    *
    * @return Returns zero on success, otherwise a negative error code.
    */
    int detect ( const JPEGImage & image, std::vector<Detection> & detections );

    /**
    * Detects only the highest scoring object in a given image which matches one of the models added before using addModel() or addModels().
    *
    * @param[in] image The image.
    *
    * @param[out] detection A detection object which will receive information about the highest scoring detection.
    *
    * @return Returns zero on success, otherwise a negative error code.
    */
    int detectMax ( const JPEGImage & image, Detection & detection );
    
    /**
    * Adds a model to the detection stack.
    *
    * @param[in] classname The name of the class ('bicycle' for example). It is used to name the objects detected in an image.
    * If there already is a model with the same class name, it will be replaced with this new one.
    *
    * @param[in] modelfile The filename of the model to load.
    *
    * @param[in] threshold The detection threshold for this model.
    *
    * @param[in] synsetId Optionally, the ID of the synset associated with the class. It will be present in the Detection structs.
    *
    * @return Returns zero on success, otherwise a negative error code if the model file could not be read or is invalid.
    *
    * @note Though mixing models which use different feature extractors is possible, it is not recommended, since a separate
    * feature pyramid would have to be built for every feature extractor, which will slow down detection significantly.
    */
    int addModel ( const std::string & classname, const std::string & modelfile, double threshold, const std::string & synsetId = "" );
    
    /**
    * Adds a model to the detection stack.
    *
    * @param[in] classname The name of the class ('bicycle' for example). It is used to name the objects detected in an image.
    * If there already is a model with the same class name, it will be replaced with this new one.
    *
    * @param[in] model The model to be added as Mixture object.
    *
    * @param[in] threshold The detection threshold for this model.
    *
    * @param[in] synsetId Optionally, the ID of the synset associated with the class. It will be present in the Detection structs.
    *
    * @return Returns zero on success, otherwise a negative error code.
    *
    * @note Though mixing models which use different feature extractors is possible, it is not recommended, since a separate
    * feature pyramid would have to be built for every feature extractor, which will slow down detection significantly.
    */
    int addModel ( const std::string & classname, const Mixture & model, double threshold, const std::string & synsetId = "" );
    
    /**
    * Adds a model to the detection stack.
    *
    * @param[in] classname The name of the class ('bicycle' for example). It is used to name the objects detected in an image.
    * If there already is a model with the same class name, it will be replaced with this new one.
    *
    * @param[in] model The model to be added as Mixture object, whose contents will be moved.
    *
    * @param[in] threshold The detection threshold for this model.
    *
    * @param[in] synsetId Optionally, the ID of the synset associated with the class. It will be present in the Detection structs.
    *
    * @return Returns zero on success, otherwise a negative error code.
    *
    * @note Though mixing models which use different feature extractors is possible, it is not recommended, since a separate
    * feature pyramid would have to be built for every feature extractor, which will slow down detection significantly.
    */
    int addModel ( const std::string & classname, Mixture && model, double threshold, const std::string & synsetId = "" );
    
    /**
    * Replaces the model at a specific position in the detection stack. The new model will use the same class name as the old one.
    *
    * @param[in] modelIndex The index of the model to be replaced.
    *
    * @param[in] model The new model as Mixture object.
    *
    * @param[in] threshold The detection threshold for the new model.
    *
    * @return Returns zero on success, otherwise a negative error code.
    * ARTOS_RES_INTERNAL_ERROR will be returned if there is no model with the given index.
    */
    int replaceModel ( const unsigned int modelIndex, const Mixture & model, double threshold );
    
    /**
    * Replaces the model at a specific position in the detection stack. The new model will use the same class name as the old one.
    *
    * @param[in] modelIndex The index of the model to be replaced.
    *
    * @param[in] model The new model as Mixture object, whose contents will be moved.
    *
    * @param[in] threshold The detection threshold for the new model.
    *
    * @return Returns zero on success, otherwise a negative error code.
    * ARTOS_RES_INTERNAL_ERROR will be returned if there is no model with the given index.
    */
    int replaceModel ( const unsigned int modelIndex, Mixture && model, double threshold );

    /**
    * Adds multiple models to the detection stack at once using information given in a file enumerating the models.
    *
    * The list file contains one model per line.  
    * Each model is described by
    * 1. the name of the class,
    * 2. the filename of the model file (relative paths are relative to the directory of the model list file),
    * 3. the detection threshold as double,
    * 4. optionally, the ID of the synset associated with the class.
    * Those 3 or 4 components are separated by spaces, while the class and file name can be surrounded by quotes to enable spaces within them.  
    * Lines starting with a hash-sign ('#') as well as empty lines will be ignored.
    *
    * @param[in] modellistfn The filename of the model list.
    *
    * @return Returns the number of successfully added models or -1 if the model list file could not be read.
    *
    * @note Though mixing models which use different feature extractors is possible, it is not recommended, since a separate
    * feature pyramid would have to be built for every feature extractor, which will slow down detection significantly.
    */
    int addModels ( const std::string & modellistfn );
    
    /**
    * @return The number of mixtures in the detection stack.
    */
    unsigned int getNumModels() const { return this->mixtures.size(); };
    
    /**
    * Returns a model added before using addModel() or addModels().
    *
    * @param[in] classname The class name of the model to be returned.
    *
    * @return Returns a pointer to a Mixture object or NULL if there is no model with that class name.
    */
    const Mixture * getModel(const std::string & classname) const;
    
    /**
    * Returns a model added before using addModel() or addModels().
    *
    * @param[in] modelIndex The index of the model to be returned.
    *
    * @return Returns a pointer to an Mixture object or NULL if the given index is out of bounds.
    */
    const Mixture * getModel(const unsigned int modelIndex) const;
    
    /**
    * Determines the classname of the model with a given index in the detection stack.
    *
    * @param[in] modelIndex The index of the model.
    *
    * @return Returns the classname of the model at the given index or an empty string if
    * that index is out of bounds.
    */
    std::string getClassnameFromIndex( const unsigned int modelIndex ) const;
    
    /**
    * @return Returns the minimum size of all models added to this detector.
    */
    Size minModelSize() const;
    
    /**
    * @return Returns the maximum size of all models added to this detector.
    */
    Size maxModelSize() const;
    
    /**
    * @return Returns the number of different feature extractors among those used by the models
    * added to this detector.
    *
    * @note Though mixing models which use different feature extractors is possible, it is not recommended, since a separate
    * feature pyramid would have to be built for every feature extractor, which will slow down detection significantly.
    */
    int differentFeatureExtractors() const { return this->featureExtractors.size(); };


protected:

    double overlap;
    int interval;
    bool verbose;
    unsigned int nextModelIndex;

    std::map<std::string, Mixture*> mixtures;
    std::map<std::string, double> thresholds;
    std::map<std::string, std::string> synsetIds;
    std::map<std::string, unsigned int> modelIndices;
    std::map<std::string, unsigned int> featureExtractorIndices;
    
    std::vector< std::shared_ptr<FeatureExtractor> > featureExtractors;
    
    int initPatchwork(unsigned int rows, unsigned int cols, unsigned int numFeatures);

    int addModelPointer ( const std::string & classname, Mixture * model, double threshold, const std::string & synsetId = "" );


private:

    void init ( bool verbose, double overlap, int interval );
    int detect( int width, int height, const FeaturePyramid & pyramid, unsigned int featureExtractorIndex, std::vector<Detection> & detections );

};


}

#endif
