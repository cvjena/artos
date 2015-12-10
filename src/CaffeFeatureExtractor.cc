#include "CaffeFeatureExtractor.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <sstream>
using namespace ARTOS;
using namespace caffe;
using namespace std;


std::map< std::pair<std::string, std::string>, std::weak_ptr< caffe::Net<float> > > CaffeFeatureExtractor::netPool;

bool CaffeFeatureExtractor::initializedCaffe = false;


CaffeFeatureExtractor::CaffeFeatureExtractor() : m_net(nullptr), m_mean(0)
{
    if (!CaffeFeatureExtractor::initializedCaffe)
    {
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        Caffe::set_mode(Caffe::GPU);
#endif
        ::google::InitGoogleLogging("CaffeFeatureExtractor");
        CaffeFeatureExtractor::initializedCaffe = true;
    }
    this->m_stringParams["netFile"] = "";
    this->m_stringParams["weightsFile"] = "";
    this->m_stringParams["meanFile"] = "";
    this->m_stringParams["layerName"] = "";
    this->m_intParams["maxImgSize"] = 0;
};


CaffeFeatureExtractor::CaffeFeatureExtractor(const string & netFile, const string & weightsFile,
                                             const string & meanFile, const string & layerName)
: CaffeFeatureExtractor()
{
    this->setParam("netFile", netFile);
    this->setParam("weightsFile", weightsFile);
    if (!meanFile.empty())
        this->setParam("meanFile", meanFile);
    if (!layerName.empty())
        this->setParam("layerName", layerName);
};


int CaffeFeatureExtractor::numFeatures() const
{
    if (!this->m_net)
        throw UseBeforeSetupException("netFile and weightsFile have to be set before CaffeFeatureExtractor may be used.");
    
    return this->m_net->blob_by_name(this->m_net->layer_names()[this->m_layerIndex])->channels();
}


Size CaffeFeatureExtractor::cellSize() const
{
    if (!this->m_net)
        throw UseBeforeSetupException("netFile and weightsFile have to be set before CaffeFeatureExtractor may be used.");
    
    return this->m_cellSize;
}


Size CaffeFeatureExtractor::borderSize() const
{
    if (!this->m_net)
        throw UseBeforeSetupException("netFile and weightsFile have to be set before CaffeFeatureExtractor may be used.");
    
    return this->m_borderSize;
}


void CaffeFeatureExtractor::setParam(const string & paramName, int32_t val)
{
    if (paramName == "maxImgSize" && val < 0)
        val = 0;
    
    FeatureExtractor::setParam(paramName, val);
}


void CaffeFeatureExtractor::setParam(const string & paramName, const std::string & val)
{
    if (paramName == "layerName" && this->m_net && !this->m_net->has_blob(val))
        throw std::invalid_argument("CNN layer not found: " + val);
    
    FeatureExtractor::setParam(paramName, val);
    
    if (paramName == "netFile" || paramName == "weightsFile")
    {
        this->m_net.reset();
        this->loadNetwork();
    }
    else if (paramName == "meanFile")
        this->loadMean();
    else if (paramName == "layerName")
        this->loadLayerInfo();
}


void CaffeFeatureExtractor::extract(const JPEGImage & img, FeatureMatrix & feat) const
{
    if (!this->m_net)
        throw UseBeforeSetupException("netFile and weightsFile have to be set before CaffeFeatureExtractor may be used.");
    
    Blob<float> * input_layer = this->m_net->input_blobs()[0];
    if (input_layer->num() != 1 || input_layer->height() != img.height() || input_layer->width() != img.width())
    {
        input_layer->Reshape(1, this->m_numChannels, img.height(), img.width());
        // Forward dimension change to all convolutional layers
        for (int i = 0; i <= this->m_layerIndex; i++)
            this->m_net->layers()[i]->Reshape(this->m_net->bottom_vecs()[i], this->m_net->top_vecs()[i]);
    }

    // Create cv::Mat wrapper around input layer
    vector<cv::Mat> input_channels;
    this->wrapInputLayers(input_channels);

    // Preprocess image and copy its channels to input_channels
    this->preprocess(img, input_channels);

    // Forward the net until the last layer we need
    this->m_net->ForwardTo(this->m_layerIndex);

    // Extract features from the given layer
    const boost::shared_ptr< Blob<float> > feature_layer = this->m_net->blob_by_name(this->m_net->layer_names()[this->m_layerIndex]);
    int w = feature_layer->width(), h = feature_layer->height();
    feat.resize(h, w, feature_layer->channels());
    const float * feature_data = feature_layer->cpu_data();
    for (int c = 0; c < feature_layer->channels(); c++)
    {
        feat.channel(c) = Eigen::Map< const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(
            feature_data, h, w
        ).cast<FeatureScalar>();
        feature_data += w * h;
    }
}


void CaffeFeatureExtractor::wrapInputLayers(vector<cv::Mat> & input_channels) const
{
    Blob<float> * input_layer = this->m_net->input_blobs()[0];
    int width = input_layer->width(), height = input_layer->height();
    float * input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        input_channels.push_back(cv::Mat(height, width, CV_32FC1, input_data));
        input_data += width * height;
    }
}


void CaffeFeatureExtractor::preprocess(const JPEGImage & img, vector<cv::Mat> & input_channels) const
{
    // Image should never have a depth differing from 3 or 1
    if (img.depth() != 3 && img.depth() != 1)
        throw std::invalid_argument("Images must either be RGB or grayscale for CNN feature extraction.");
    
    // Convert the input image to the input image format of the network
    const cv::Mat cvImg(
        img.height(), img.width(),
        (img.depth() == 3) ? CV_8UC3 : CV_8UC1,
        reinterpret_cast<void*>(const_cast<uint8_t*>(img.bits()))
    );
    cv::Mat sample;
    if (cvImg.channels() == 3 && this->m_numChannels == 1)
        cv::cvtColor(cvImg, sample, CV_RGB2GRAY);
    else if (cvImg.channels() == 1 && this->m_numChannels == 3)
        cv::cvtColor(cvImg, sample, CV_GRAY2BGR);
    else
        cv::cvtColor(cvImg, sample, CV_RGB2BGR);

    cv::Mat sample_float;
    if (this->m_numChannels == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);

    sample_float -= this->m_mean;

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_float, input_channels);
}


void CaffeFeatureExtractor::loadNetwork()
{
    string netFile = this->getStringParam("netFile");
    string weightsFile = this->getStringParam("weightsFile");
    if (!netFile.empty() && !weightsFile.empty())
    {
        // Search for cached network
        auto cacheIt = CaffeFeatureExtractor::netPool.find(make_pair(netFile, weightsFile));
        if (cacheIt != CaffeFeatureExtractor::netPool.end() && (this->m_net = cacheIt->second.lock()))
            this->m_numChannels = this->m_net->input_blobs()[0]->channels();
        else
        {
        
            // Initialize network structure
            {
                NetParameter param;
                if (!ReadProtoFromTextFile(netFile, &param))
                    throw std::invalid_argument("Could not load network structure from " + netFile);
                UpgradeNetAsNeeded(netFile, &param);
                param.mutable_state()->set_phase(TEST);
                this->m_net.reset(new Net<float>(param));
            }
            
            // Load network weights
            {
                NetParameter param;
                if (!ReadProtoFromBinaryFile(weightsFile, &param))
                    throw std::invalid_argument("Could not load pre-trained network weights from " + weightsFile);
                UpgradeNetAsNeeded(weightsFile, &param);
                this->m_net->CopyTrainedLayersFrom(param);
            }
            
            // Check network parameters
            if (this->m_net->num_inputs() != 1)
                throw std::invalid_argument("Network should have exactly one input.");
        
            this->m_numChannels = this->m_net->input_blobs()[0]->channels();
            if (this->m_numChannels != 1 && this->m_numChannels != 3)
                throw std::invalid_argument("Input layer must have 1 or 3 channels.");
            
            // Add network to cache
            CaffeFeatureExtractor::netPool[make_pair(netFile, weightsFile)] = this->m_net;
        }

        // Find last layer before fully connected subnet
        this->m_lastLayer = -1;
        for (size_t layer_id = 0; layer_id < this->m_net->layers().size()
                && string(this->m_net->layers()[layer_id]->type()) != string("InnerProduct"); ++layer_id)
            ++this->m_lastLayer;
        
        this->loadLayerInfo();
    }
}


void CaffeFeatureExtractor::loadMean()
{
    this->m_mean = cv::Scalar(0);
    string meanFile = this->getStringParam("meanFile");
    
    if (!meanFile.empty())
    {
        BlobProto blob_proto;
        if (!ReadProtoFromBinaryFile(meanFile.c_str(), &blob_proto))
            throw std::invalid_argument("Mean file could not be loaded: " + meanFile);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        if (this->m_net && mean_blob.channels() != this->m_numChannels)
            throw std::invalid_argument("Number of channels of mean file doesn't match input layer.");

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        vector<cv::Mat> channels;
        float * data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < mean_blob.channels(); ++i)
        {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value per channel. */
        this->m_mean = cv::mean(mean);
    }
}


void CaffeFeatureExtractor::loadLayerInfo()
{
    if (!this->m_net)
        return;
   
    // Find layer 
    string layerName;
    if (this->m_stringParams["layerName"].empty())
    {
        this->m_stringParams["layerName"] = this->m_net->layer_names()[this->m_lastLayer];
        this->m_layerIndex = this->m_lastLayer;
        layerName = this->m_net->layer_names()[this->m_layerIndex];
    }
    else
    {
        string layerName = this->m_stringParams["layerName"];
        for (this->m_layerIndex = 0; this->m_layerIndex < this->m_lastLayer
             && this->m_net->layer_names()[this->m_layerIndex] != layerName; ++this->m_layerIndex);
        
        if (this->m_layerIndex == this->m_lastLayer && this->m_net->layer_names()[this->m_layerIndex] != layerName)
            throw runtime_error("CNN layer not found or behind fully-connected layer: " + layerName);
    }
    
    // Determine cell size and border size
    this->m_cellSize = Size(1, 1);
    this->m_borderSize = Size(0, 0);
    for (int l = 0; l <= this->m_layerIndex; l++)
    {
        const LayerParameter & layerParam = this->m_net->layers()[l]->layer_param();
        if (layerParam.has_convolution_param())
        {
            const ConvolutionParameter cp = layerParam.convolution_param();
            
            if (cp.stride_size() > 0)
                this->m_cellSize *= Size(cp.stride(0));
            else
            {
                if (cp.has_stride_w())
                    this->m_cellSize.width *= cp.stride_w();
                if (cp.has_stride_h())
                    this->m_cellSize.height *= cp.stride_h();
            }
            
            if (cp.kernel_size_size() > 0)
                this->m_borderSize += Size(cp.kernel_size(0)) / 2;
            else
            {
                if (cp.has_kernel_w())
                    this->m_borderSize.width += cp.kernel_w() / 2;
                if (cp.has_kernel_h())
                    this->m_borderSize.height += cp.kernel_h() / 2;
            }
            
            if (cp.pad_size() > 0)
                this->m_borderSize -= Size(cp.pad(0));
            else
            {
                if (cp.has_pad_w())
                    this->m_borderSize.width -= cp.pad_w();
                if (cp.has_pad_h())
                    this->m_borderSize.height -= cp.pad_h();
            }
            
        }
        else if (layerParam.has_pooling_param())
        {
            const PoolingParameter pp = layerParam.pooling_param();
            
            if (pp.has_stride())
                this->m_cellSize *= Size(pp.stride());
            else
            {
                if (pp.has_stride_w())
                    this->m_cellSize.width *= pp.stride_w();
                if (pp.has_stride_h())
                    this->m_cellSize.height *= pp.stride_h();
            }
            
            if (pp.has_kernel_size())
                this->m_borderSize += Size(pp.kernel_size()) / 2;
            else
            {
                if (pp.has_kernel_w())
                    this->m_borderSize.width += pp.kernel_w() / 2;
                if (pp.has_kernel_h())
                    this->m_borderSize.height += pp.kernel_h() / 2;
            }
            
            if (pp.has_pad())
                this->m_borderSize -= Size(pp.pad());
            else
            {
                if (pp.has_pad_w())
                    this->m_borderSize.width -= pp.pad_w();
                if (pp.has_pad_h())
                    this->m_borderSize.height -= pp.pad_h();
            }
            
        }
    }
}
