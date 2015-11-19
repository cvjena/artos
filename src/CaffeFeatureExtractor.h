#ifndef ARTOS_CAFFEFEATUREEXTRACTOR_H
#define ARTOS_CAFFEFEATUREEXTRACTOR_H

#include "FeatureExtractor.h"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>

namespace ARTOS
{

/**
* Uses Caffe to extract image features from a specific layer of a given
* Convolutional Neural Network (CNN).
*
* **Parameters of this feature extractor:**
*     - *netFile* (`string`) - path to the protobuf file specifying the network structure.
*     - *weightsFile* (`string`) - path to the file with the pre-trained weights for the network.
*     - *meanFile* (`string`) - path to a mean image file which has to be subtracted from each
*       sample before propagating it through the network.
*     - *layerName* (`string`) - the name of the layer in the network to extract features from.
*     - *maxImgSize* (`int`) - maximum size of input images (may be limited to save time and memory).
*       0 means no limit.
*
* @see http://caffe.berkeleyvision.org/
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class CaffeFeatureExtractor : public FeatureExtractor
{

public:

    /**
    * Constructs an empty CaffeFeatureExtractor which is not yet ready to be used.
    * The parameters "netFile" and "weightsFile" have to be set before use.
    */
    CaffeFeatureExtractor();
    
    /**
    * Constructs a CaffeFeatureExtractor for a given pre-trained network.
    *
    * @param[in] netFile Path to the protobuf file specifying the network structure.
    *
    * @param[in] weightsFile Path to the file with the pre-trained weights for the network.
    *
    * @param[in] meanFile Optionally, path to a mean image file which has to be subtracted from each
    * sample before propagating it through the network.
    *
    * @param[in] layerName The name of the layer in the network to extract features from. If an empty
    * string is given, the last layer before the first fully connected layer will be selected.
    */
    CaffeFeatureExtractor(const std::string & netFile, const std::string & weightsFile,
                          const std::string & meanFile = "", const std::string & layerName = "");

    virtual ~CaffeFeatureExtractor() {};

    /**
    * @return Returns the unique identifier of this kind of feature extractor. That type specifier
    * must consist of alphanumeric characters + dashes + underscores only and must begin with a letter.
    */
    virtual const char * type() const override { return "Caffe"; };
    
    /**
    * @return Human-readable name of this feature extractor.
    */
    virtual const char * name() const override { "CNN Features (Caffe)" };

    /**
    * @return Returns the number of features this feature extractor extracts from each cell.
    */
    virtual int numFeatures() const override;
    
    /**
    * @return Returns the size of the cells used by this feature extractor in x and y direction.
    */
    virtual Size cellSize() const override;
    
    /**
    * Specifies the size of the border along the x and y dimension of an image, which gets lost
    * during feature extraction. This may be due to unpadded convolutions, for instance.  
    * For example, a border of size (4, 2) would indicate, that only the region between (4, 2) and
    * (width - 5, height - 3) of an image would be transformed into features.
    *
    * @return Returns the size of the border along each image dimension in pixels.
    */
    virtual Size borderSize() const override;
    
    /**
    * @return Returns a Size struct with the maximum sizes for image in x and y direction which
    * can be processed by this feature extractor. If any dimension is 0, the size of the image
    * along that dimension does not need to be limited.
    */
    virtual Size maxImageSize() const override { return Size(this->getIntParam("maxImgSize")); };
    
    /**
    * @return Returns true if it is considered reasonable to process feature extraction of multiple
    * scales of an image by patchworking them together, so that multiple scales are processed at
    * once on a single plane, which will have the size of the largest scale.
    * The patchworkPadding() reported by the feature extractor will be used as padding between images
    * on the same plane.
    */
    virtual bool patchworkProcessing() const override { return false; };
    
    /**
    * Specifies the amount of padding which should be added between images on the same plane when
    * features are extracted using patchworking (see patchworkProcessing()).
    *
    * @return Returns the amount of padding to add between two images.
    */
    virtual Size patchworkPadding() const override { return this->borderSize(); };
    
    /**
    * Computes features for a given image.
    *
    * @param[in] img The image to compute features for.
    *
    * @param[out] feat Destination matrix to store the extracted features in.
    * It will be resized to fit the number of cells in the given image.
    *
    * @note This function must be thread-safe.
    */
    virtual void extract(const JPEGImage & img, FeatureMatrix & feat) const override;
    
    /**
    * Changes the value of an integer parameter specific to the concrete feature extraction method.
    *
    * @param[in] val The new value for the parameter.
    *
    * @throws UnknownParameterException There is no string parameter with the given name.
    *
    * @throws std::invalid_argument The given value is not allowed for the given parameter.
    */
    virtual void setParam(const std::string & paramName, int32_t val) override;
    
    /**
    * Changes the value of a string parameter specific to this algorithm.
    *
    * @param[in] val The new value for the parameter.
    *
    * @throws UnknownParameterException There is no string parameter with the given name.
    *
    * @throws std::invalid_argument The given value is not allowed for the given parameter.
    */
    virtual void setParam(const std::string & paramName, const std::string & val) override;


protected:

    std::shared_ptr< Caffe::Net<float> > m_net; /**< The network. */
    cv::Scalar m_mean; /**< Image mean. */
    int m_lastLayer; /**< Index of the last convolutional layer in the network before the fully connected network. */
    int m_numChannels; /**< Number of input channels of the network. */
    int m_layerIndex; /**< Index of the layer to extract features from. */
    Size m_cellSize; /**< Cell size derived from the network structure. */
    Size m_borderSize; /**< Border size derived from the network structure. */
    
    /**
    * Tries to load the network using the current parameters of this feature extractor.
    * Nothing will be done if the parameters are not yet set up.
    *
    * @throws std::invalid_argument All required parameters have been set up, but the network could not be loaded.
    */
    void loadNetwork();
    
    /**
    * Tries to load the image mean from the file specified in the parameter "meanFile".
    *
    * @throws std::invalid_argument The mean file could not be loaded.
    */
    void loadMean();
    
    /**
    * Caches information about the layer specified in the parameter "layerName" if the net has already been loaded
    * and sets m_cellSize and m_borderSize accordingly.
    *
    * If the specified layer could not be found, the last convolutional layer in the network will be used.
    */
    void loadLayerInfo();
    
    /**
    * Wraps a cv::Mat object around each channel of the input layer and adds those wrappers to @p input_channels.
    */
    void wrapInputLayers(std::vector<cv::Mat> & input_channels);
    
    /**
    * Preprocesses a given image and stores the result in given channels of the input layer.
    *
    * @param[in] img The image.
    *
    * @param[out] input_channels Vector of cv::Mat objects which will retrieve the data of each channel
    * of the preprocessed image.
    */
    void preprocess(const JPEGImage & img, std::vector<cv::Mat> & input_channels);
    
    /**
    * Cache of networks which have already been loaded to be used by different feature extractor instances.
    * Their key is a pair of the protobuf filename and the weights filename.
    */
    static std::map< std::pair<std::string, std::string>, std::shared_ptr< Caffe::Net<float> > > netPool;

};

}

#endif