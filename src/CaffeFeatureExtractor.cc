#include "CaffeFeatureExtractor.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cassert>
#include "portable_endian.h"
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
    this->m_stringParams["scalesFile"] = "";
    this->m_stringParams["pcaFile"] = "";
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


const char * CaffeFeatureExtractor::name() const
{
    return "CNN Features (Caffe)";
}


int CaffeFeatureExtractor::numFeatures() const
{
    if (!this->m_net)
        throw UseBeforeSetupException("netFile and weightsFile have to be set before CaffeFeatureExtractor may be used.");
    
    int numFeat = this->m_net->top_vecs()[this->m_layerIndex][0]->channels();
    return (this->m_pcaMean.size() == numFeat) ? this->m_pcaTransform.cols() : numFeat;
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


Size CaffeFeatureExtractor::maxImageSize() const
{
    return Size(this->getIntParam("maxImgSize"));
}


bool CaffeFeatureExtractor::supportsMultiThread() const
{
    return false;
}


bool CaffeFeatureExtractor::patchworkProcessing() const
{
    return false;
}


Size CaffeFeatureExtractor::patchworkPadding() const
{
    return this->borderSize();
}


Size CaffeFeatureExtractor::cellsToPixels(const Size & cells) const
{
    // Initial estimate
    Size pixels = FeatureExtractor::cellsToPixels(cells);
    
    // Adjustment
    Size adjustment = this->cellSize();
    Size c = this->pixelsToCells(pixels);
    while (c != cells)
    {
        adjustment = max(Size(1), adjustment / 2);
        if (c.width != cells.width)
            pixels.width += adjustment.width * ((c.width < cells.width) ? 1 : -1);
        if (c.height != cells.height)
            pixels.height += adjustment.height * ((c.height < cells.height) ? 1 : -1);
        c = this->pixelsToCells(pixels);
    }
    
    return pixels;
}


Size CaffeFeatureExtractor::pixelsToCells(const Size & pixels) const
{
    Size cells = pixels;
    LayerParams layerParams;
    for (int l = 0; l <= this->m_layerIndex; l++)
    {
        this->getLayerParams(l, layerParams);
        
        switch (layerParams.layerType)
        {
            case LayerType::CONV:
                // According to Caffe::ConvolutionLayer::compute_output_shape()
                cells = (cells + layerParams.padding * 2 - layerParams.kernelSize) / layerParams.stride + 1;
                break;
            
            case LayerType::POOL:
                // According to Caffe::PoolingLayer::Reshape()
                Size prevSize = cells;
                cells.width = static_cast<int>(ceil(static_cast<float>(cells.width + 2 * layerParams.padding.width - layerParams.kernelSize.width)
                                / layerParams.stride.width)) + 1;
                cells.height = static_cast<int>(ceil(static_cast<float>(cells.height + 2 * layerParams.padding.height - layerParams.kernelSize.height)
                                / layerParams.stride.height)) + 1;
                if (layerParams.padding.width || layerParams.padding.height)
                {
                    if ((cells.width - 1) * layerParams.stride.width >= prevSize.width + layerParams.padding.width)
                        cells.width--;
                    if ((cells.height - 1) * layerParams.stride.height >= prevSize.height + layerParams.padding.height)
                        cells.height--;
                }
                break;
        }
    }
    return cells;
}


void CaffeFeatureExtractor::setParam(const string & paramName, int32_t val)
{
    if (paramName == "maxImgSize" && val < 0)
        val = 0;
    
    FeatureExtractor::setParam(paramName, val);
}


void CaffeFeatureExtractor::setParam(const string & paramName, const std::string & val)
{
    if (paramName == "layerName" && this->m_net && !this->m_net->has_layer(val))
        throw std::invalid_argument("CNN layer not found: " + val);
    
    FeatureExtractor::setParam(paramName, val);
    
    if (paramName == "netFile" || paramName == "weightsFile")
    {
        this->m_net.reset();
        this->loadNetwork();
    }
    else if (paramName == "meanFile")
        this->loadMean();
    else if (paramName == "scalesFile")
        this->loadScales();
    else if (paramName == "pcaFile")
        this->loadPCAParams();
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
    const Blob<float> * feature_layer = this->m_net->top_vecs()[this->m_layerIndex][0];
    int w = feature_layer->width(), h = feature_layer->height();
    assert(Size(w, h) == this->pixelsToCells(Size(img.width(), img.height())));
    feat.resize(h, w, feature_layer->channels());
    const float * feature_data = feature_layer->cpu_data();
    for (int c = 0; c < feature_layer->channels(); c++)
    {
        feat.channel(c) = Eigen::Map< const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(
            feature_data, h, w
        ).cast<FeatureScalar>();
        feature_data += w * h;
    }
    
    // Scale features
    if (this->m_scales.size() == feat.channels())
        feat /= this->m_scales;
    
    // Apply dimensionality reduction
    if (this->m_pcaMean.size() == feat.channels())
    {
        FeatureMatrix reducedFeat(feat.rows(), feat.cols(), this->m_pcaTransform.cols());
        feat -= this->m_pcaMean;
        reducedFeat.asCellMatrix().noalias() = feat.asCellMatrix() * this->m_pcaTransform;
        feat = move(reducedFeat);
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
        this->loadScales();
        this->loadPCAParams();
    }
}


void CaffeFeatureExtractor::loadMean()
{
    this->m_mean = cv::Scalar(0);
    string meanFile = this->getStringParam("meanFile");
    
    if (!meanFile.empty())
    {
        // Try to read binaryproto as mean image
        BlobProto blob_proto;
        if (ReadProtoFromBinaryFile(meanFile.c_str(), &blob_proto))
        {

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
        else
        {
            
            // Try to read mean channel values from text file
            ifstream mFile(meanFile);
            float dummy;
            if (!mFile.is_open())
                throw std::invalid_argument("Mean file could not be loaded: " + meanFile);
            mFile >> this->m_mean[0] >> this->m_mean[1] >> this->m_mean[2];
            if (mFile.fail() || (mFile >> dummy))
                throw std::invalid_argument("Mean file could not be loaded: " + meanFile);
            
        }
    }
}


void CaffeFeatureExtractor::loadScales()
{
    this->m_scales = FeatureCell();
    string scalesFilename = this->getStringParam("scalesFile");

    if (this->m_net && !scalesFilename.empty())
    {
        FeatureCell scales(this->numFeatures());
        
        ifstream scalesFile(scalesFilename);
        if (!scalesFile.is_open())
            throw std::invalid_argument("Scales file could not be loaded: " + scalesFilename);
        
        for (FeatureCell::Index i = 0; i < scales.size(); i++)
        {
            if (scalesFile.eof())
                throw std::invalid_argument("Wrong number of channels in scales file: " + scalesFilename);
            scalesFile >> scales(i);
            if (scalesFile.bad())
                throw std::invalid_argument("Scales file could not be read: " + scalesFilename);
            else if (scalesFile.fail())
                throw std::invalid_argument("Invalid scales file: " + scalesFilename);
        }
        
        this->m_scales = scales;
    }
}


void CaffeFeatureExtractor::loadPCAParams()
{
    this->m_pcaMean = FeatureCell();
    this->m_pcaTransform = ScalarMatrix();
    string pcaFilename = this->getStringParam("pcaFile");
    
    if (this->m_net && !pcaFilename.empty())
    {
        ifstream pcaFile(pcaFilename, ios_base::in | ios_base::binary);
        if (!pcaFile.is_open())
            throw std::invalid_argument("PCA file could not be loaded: " + pcaFilename);
        
        // Read metadata
        uint32_t numRows, numCols;
        pcaFile.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));
        pcaFile.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));
        numRows = le32toh(numRows);
        numCols = le32toh(numCols);
        if (numRows != this->numFeatures())
            throw std::invalid_argument("Wrong number of features in PCA file: " + pcaFilename);
        if (numCols > numRows)
            throw std::invalid_argument("Reduced is larger than original dimensionality in PCA file: " + pcaFilename);
        
        // Read mean
        FeatureCell mean(numRows);
        float buf;
        for (FeatureCell::Index i = 0; i < numRows; i++)
        {
            if (pcaFile.eof())
                throw std::invalid_argument("Unexpected end of PCA file while reading mean: " + pcaFilename);
            pcaFile.read(reinterpret_cast<char*>(&buf), sizeof(float));
            if (pcaFile.bad())
                throw std::invalid_argument("PCA file could not be read: " + pcaFilename);
            mean(i) = le32toh(buf);
        }
        
        // Read transformation matrix
        ScalarMatrix transform(numRows, numCols);
        for (ScalarMatrix::Index i = 0; i < transform.size(); i++)
        {
            if (pcaFile.eof())
                throw std::invalid_argument("Unexpected end of PCA file while reading matrix: " + pcaFilename);
            pcaFile.read(reinterpret_cast<char*>(&buf), sizeof(float));
            if (pcaFile.bad())
                throw std::invalid_argument("PCA file could not be read: " + pcaFilename);
            transform(i) = le32toh(buf);
        }
        
        this->m_pcaMean = mean;
        this->m_pcaTransform = transform;
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
    LayerParams layerParams;
    for (int l = 0; l <= this->m_layerIndex; l++)
    {
        this->getLayerParams(l, layerParams);
        this->m_cellSize *= layerParams.stride;
        this->m_borderSize += layerParams.kernelSize / 2;
        this->m_borderSize -= layerParams.padding;
    }
}


void CaffeFeatureExtractor::getLayerParams(int layerIndex, CaffeFeatureExtractor::LayerParams & params) const
{
    // Initialize to default values
    params.layerType = LayerType::OTHER;
    params.kernelSize = Size(1);
    params.padding = Size(0);
    params.stride = Size(1);
    
    // Retrieve parameters
    const LayerParameter & layerParam = this->m_net->layers()[layerIndex]->layer_param();
    if (layerParam.has_convolution_param())
    {
        // Handle convolutional parameters
        const ConvolutionParameter cp = layerParam.convolution_param();
        
        params.layerType = LayerType::CONV;
        
        if (cp.kernel_size_size() > 0)
            params.kernelSize = Size(cp.kernel_size(0));
        else
        {
            if (cp.has_kernel_w())
                params.kernelSize.width = cp.kernel_w();
            if (cp.has_kernel_h())
                params.kernelSize.height = cp.kernel_h();
        }
        
        if (cp.pad_size() > 0)
            params.padding = Size(cp.pad(0));
        else
        {
            if (cp.has_pad_w())
                params.padding.width = cp.pad_w();
            if (cp.has_pad_h())
                params.padding.height = cp.pad_h();
        }
        
        if (cp.stride_size() > 0)
            params.stride = Size(cp.stride(0));
        else
        {
            if (cp.has_stride_w())
                params.stride.width = cp.stride_w();
            if (cp.has_stride_h())
                params.stride.height = cp.stride_h();
        }
        
    }
    else if (layerParam.has_pooling_param())
    {
        // Handle pooling parameters
        const PoolingParameter pp = layerParam.pooling_param();
        
        params.layerType = LayerType::POOL;
        
        if (pp.has_kernel_size())
            params.kernelSize = Size(pp.kernel_size());
        else
        {
            if (pp.has_kernel_w())
                params.kernelSize.width = pp.kernel_w();
            if (pp.has_kernel_h())
                params.kernelSize.height = pp.kernel_h();
        }
        
        if (pp.has_pad())
            params.padding = Size(pp.pad());
        else
        {
            if (pp.has_pad_w())
                params.padding.width = pp.pad_w();
            if (pp.has_pad_h())
                params.padding.height = pp.pad_h();
        }
        
        if (pp.has_stride())
            params.stride = Size(pp.stride());
        else
        {
            if (pp.has_stride_w())
                params.stride.width = pp.stride_w();
            if (pp.has_stride_h())
                params.stride.height = pp.stride_h();
        }
        
    }
}
