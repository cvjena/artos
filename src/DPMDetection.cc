#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>

#include "DPMDetection.h"
#include "sysutils.h"
#include "Intersector.h"
#include "timingtools.h"

using namespace ARTOS;
using namespace std;

DPMDetection::DPMDetection ( bool verbose, double overlap, int interval )
{
    init ( verbose, overlap, interval );
}

DPMDetection::DPMDetection ( const std::string & modelfile, double threshold, bool verbose, double overlap, int interval ) 
{
    init ( verbose, overlap, interval );
    addModel ( "single", modelfile, threshold );
}

DPMDetection::DPMDetection ( const Mixture & model, double threshold, bool verbose, double overlap, int interval ) 
{
    init ( verbose, overlap, interval );
    addModel ( "single", model, threshold );
}

DPMDetection::DPMDetection ( Mixture && model, double threshold, bool verbose, double overlap, int interval ) 
{
    init ( verbose, overlap, interval );
    addModel ( "single", move(model), threshold );
}

void DPMDetection::init ( bool verbose, double overlap, int interval )
{
    this->overlap = overlap;
    this->interval = interval;
    this->verbose = verbose;
    this->nextModelIndex = 0;
}


int DPMDetection::addModel ( const std::string & classname, const std::string & modelfile, double threshold, const std::string & synsetId )
{
    // Try to open the mixture
    ifstream in(modelfile.c_str(), ios::binary);
    
    if (!in.is_open()) {
        if (this->verbose)
            cerr << "\nInvalid model file " << modelfile << endl;
        return ARTOS_DETECT_RES_INVALID_MODEL_FILE;
    }
    
    Mixture * mixture = new Mixture();
    try {
        in >> (*mixture);
    } catch (const Exception & e) {
        if (this->verbose)
            cerr << "Invalid model file: " << modelfile << " (" << e.what() << ")" << endl;
        delete mixture;
        return ARTOS_DETECT_RES_INVALID_MODEL_FILE;
    } catch (const std::invalid_argument & e) {
        if (this->verbose)
            cerr << "Invalid model file: " << modelfile << " (" << e.what() << ")" << endl;
        delete mixture;
        return ARTOS_DETECT_RES_INVALID_MODEL_FILE;
    }

    return this->addModelPointer(classname, mixture, threshold, synsetId);
}

int DPMDetection::addModel ( const std::string & classname, const Mixture & model, double threshold, const std::string & synsetId )
{
    Mixture * mixture = new Mixture(model);
    return this->addModelPointer(classname, mixture, threshold, synsetId);
}


int DPMDetection::addModel ( const std::string & classname, Mixture && model, double threshold, const std::string & synsetId )
{
    Mixture * mixture = new Mixture(move(model));
    return this->addModelPointer(classname, mixture, threshold, synsetId);
}

int DPMDetection::addModelPointer ( const std::string & classname, Mixture * mixture, double threshold, const std::string & synsetId )
{
    pair< map<std::string, Mixture*>::iterator, bool > insertResult = mixtures.insert ( pair<std::string, Mixture*> ( classname, mixture ) );
    if (insertResult.second)
        modelIndices[classname] = this->nextModelIndex++;
    else
    {
        delete insertResult.first->second;
        insertResult.first->second = mixture;
    }
    thresholds[classname] = threshold;
    synsetIds[classname] = synsetId;
    
    int feIndex = -1;
    for (int i = 0; i < featureExtractors.size(); i++)
        if (*(featureExtractors[i]) == *(mixture->featureExtractor()))
        {
            feIndex = i;
            break;
        }
    if (feIndex < 0)
    {
        feIndex = featureExtractors.size();
        featureExtractors.push_back(mixture->featureExtractor());
    }
    featureExtractorIndices[classname] = static_cast<unsigned int>(feIndex);

    return ARTOS_RES_OK;
}

int DPMDetection::replaceModel ( const unsigned int modelIndex, const Mixture & model, double threshold )
{
    string classname = this->getClassnameFromIndex(modelIndex);
    return (classname.empty()) ? ARTOS_RES_INTERNAL_ERROR : this->addModel(classname, model, threshold);
}

int DPMDetection::replaceModel ( const unsigned int modelIndex, Mixture && model, double threshold )
{
    string classname = this->getClassnameFromIndex(modelIndex);
    return (classname.empty()) ? ARTOS_RES_INTERNAL_ERROR : this->addModel(classname, move(model), threshold);
}

const Mixture * DPMDetection::getModel(const string & classname) const
{
    map<string, Mixture*>::const_iterator it = mixtures.find(classname);
    return (it != mixtures.end()) ? it->second : NULL;
}

const Mixture * DPMDetection::getModel(const unsigned int modelIndex) const
{
    string classname = this->getClassnameFromIndex(modelIndex);
    return (!classname.empty()) ? this->getModel(classname) : NULL;
}

std::string DPMDetection::getClassnameFromIndex( const unsigned int modelIndex ) const
{
    for (map<string, unsigned int>::const_iterator it = modelIndices.begin(); it != modelIndices.end(); it++)
        if (it->second == modelIndex)
            return it->first;
    return "";
}

Size DPMDetection::minModelSize() const
{
    Size s;
    for (map<std::string, Mixture*>::const_iterator m = this->mixtures.begin(); m != this->mixtures.end(); ++m)
        s = (s.min() == 0) ? m->second->minSize() : min(s, m->second->minSize());
    return s;
}

Size DPMDetection::maxModelSize() const
{
    Size s;
    for (map<std::string, Mixture*>::const_iterator m = this->mixtures.begin(); m != this->mixtures.end(); ++m)
        s = max(s, m->second->maxSize());
    return s;
}

DPMDetection::~DPMDetection()
{
    for ( map<std::string, Mixture *>::iterator i = mixtures.begin(); i != mixtures.end(); i++ )
    {
        Mixture *mixture = i->second;
        if ( mixture != NULL ) 
            delete mixture;
    }
    mixtures.clear();
}

int DPMDetection::detect ( const JPEGImage & image, vector<Detection> & detections )
{
    if ( mixtures.size() == 0 )
        return ARTOS_DETECT_RES_NO_MODELS;

    int errcode;
    unsigned int minLevelSize = min(5, this->minModelSize().min());
    
    // Separate detection for every unique feature extractor
    for (unsigned int feIndex = 0; feIndex < this->featureExtractors.size(); feIndex++)
    {
    
        // Compute the features
        if (this->verbose)
            start();

        FeaturePyramid pyramid(image, this->featureExtractors[feIndex], this->interval, minLevelSize);

        if (pyramid.empty())
        {
            if (this->verbose)
                cerr << "\nCould not create feature pyramid! Image may be invalid." << endl;
            return ARTOS_DETECT_RES_INVALID_IMAGE;
        }

        if (this->verbose)
        {
            cerr << "Computed " << pyramid.featureExtractor()->type() << " features in " << stop() << " ms for an image of size " <<
                    image.width() << " x " << image.height() << endl;
        }

        errcode = this->initPatchwork(pyramid.levels()[0].rows(), pyramid.levels()[0].cols(), pyramid.levels()[0].channels());
        if (errcode != ARTOS_RES_OK)
            return errcode;

        if ( this->verbose )
            start();

        errcode = this->detect( image.width(), image.height(), pyramid, feIndex, detections);
        if (errcode != ARTOS_RES_OK)
            return errcode;
     
        if (this->verbose)
            cerr << "Computed the convolutions and distance transforms in " << stop() << " ms" << endl;
    
    }

    return errcode;
}

int DPMDetection::detect(int width, int height, const FeaturePyramid & pyramid, unsigned int featureExtractorIndex, vector<Detection> & detections)
{
    for ( map<std::string, Mixture *>::iterator m = this->mixtures.begin(); m != this->mixtures.end(); m++ )
        if (this->featureExtractorIndices[m->first] == featureExtractorIndex)
        {
            Mixture * mixture = m->second;
            const std::string & classname = m->first;
            double threshold = thresholds[classname];
            const std::string & synsetId = synsetIds[classname];
            unsigned int modelIndex = modelIndices[classname];

            // Compute the scores
            if (this->verbose)
                cerr << "Running detector for " << classname << endl;
            vector<ScalarMatrix> scores;
            vector<Mixture::Indices> argmaxes;
            vector<Detection> single_detections;
            mixture->convolve(pyramid, scores, argmaxes);
            
            // Cache the size of the models
            vector<Size> sizes(mixture->models().size());
            for (int i = 0; i < sizes.size(); ++i)
                sizes[i] = mixture->models()[i].rootSize();
            
            // For each scale
            for (int i = 0; i < scores.size(); ++i)
            {
                const double scale = pyramid.scales()[i];
              
                const int rows = scores[i].rows();
                const int cols = scores[i].cols();
              
                for (int y = 0; y < rows; ++y)
                {
                    for (int x = 0; x < cols; ++x)
                    {
                        const float score = scores[i](y, x);
                  
                        if (score > threshold)
                        {
                            if (((y == 0) || (x == 0) || (score > scores[i](y - 1, x - 1))) &&
                              ((y == 0) || (score > scores[i](y - 1, x))) &&
                              ((y == 0) || (x == cols - 1) || (score > scores[i](y - 1, x + 1))) &&
                              ((x == 0) || (score > scores[i](y, x - 1))) &&
                              ((x == cols - 1) || (score > scores[i](y, x + 1))) &&
                              ((y == rows - 1) || (x == 0) || (score > scores[i](y + 1, x - 1))) &&
                              ((y == rows - 1) || (score > scores[i](y + 1, x))) &&
                              ((y == rows - 1) || (x == cols - 1) || (score > scores[i](y + 1, x + 1))))
                            {
                                const Size pos = pyramid.featureExtractor()->cellCoordsToPixels(Size(x / scale + 0.5, y / scale + 0.5));
                                const Size size = pyramid.featureExtractor()->cellsToPixels(Size(
                                        sizes[argmaxes[i](y, x)].width / scale + 0.5,
                                        sizes[argmaxes[i](y, x)].height / scale + 0.5
                                ));
                                Rectangle bndbox(pos.width, pos.height, size.width, size.height);
                      
                                // Truncate the object
                                bndbox.setX(max(bndbox.x(), 0));
                                bndbox.setY(max(bndbox.y(), 0));
                                bndbox.setWidth(min(bndbox.width(), width - bndbox.x()));
                                bndbox.setHeight(min(bndbox.height(), height - bndbox.y()));
                                  
                                if (!bndbox.empty())
                                    single_detections.push_back(Detection(score, scale, x, y, bndbox, classname, synsetId, modelIndex));

                            }
                        }
                    }
                }
            }

            if (this->verbose)
                cerr << "Number of detections before non-maximum suppression: " << single_detections.size() << endl;

            // Non maxima suppression
            sort(single_detections.begin(), single_detections.end());
            
            for (int i = 1; i < single_detections.size(); ++i)
                single_detections.resize(remove_if(single_detections.begin() + i, single_detections.end(),
                        Intersector(single_detections[i - 1], this->overlap, true)) -
                        single_detections.begin());

            if (this->verbose)
                cerr << "Number of detections after non-maximum suppression: " << single_detections.size() << endl;

            detections.insert ( detections.begin(), single_detections.begin(), single_detections.end() );
        }
    return ARTOS_RES_OK;
}

int DPMDetection::detectMax ( const JPEGImage & image, Detection & detection )
{
    if ( mixtures.size() == 0 )
        return ARTOS_DETECT_RES_NO_MODELS;

    // Separate detection for every unique feature extractor
    for (unsigned int feIndex = 0; feIndex < this->featureExtractors.size(); feIndex++)
    {
        
        // Compute the features
        if (this->verbose)
            start();
        
        FeaturePyramid pyramid(image, this->featureExtractors[feIndex], this->interval);

        if (pyramid.empty())
        {
            if (this->verbose)
                cerr << "\nCould not create feature pyramid! Image may be invalid." << endl;
            return ARTOS_DETECT_RES_INVALID_IMAGE;
        }

        if (this->verbose) {
            cerr << "Computed " << pyramid.featureExtractor()->type() << " features in " << stop() << " ms for an image of size " <<
                    image.width() << " x " << image.height() << endl;
        }

        int errcode = this->initPatchwork(pyramid.levels()[0].rows(), pyramid.levels()[0].cols(), pyramid.levels()[0].channels());
        if (errcode != ARTOS_RES_OK)
            return errcode;

        if ( this->verbose )
            start();

        FeatureScalar score, maxScore = -1 * numeric_limits<FeatureScalar>::infinity();
        int y, x;
        for ( map<std::string, Mixture *>::iterator m = this->mixtures.begin(); m != this->mixtures.end(); m++ )
            if (this->featureExtractorIndices[m->first] == feIndex)
            {
                Mixture * mixture = m->second;
                const std::string & classname = m->first;
                const std::string & synsetId = synsetIds[classname];
                unsigned int modelIndex = modelIndices[classname];

                // Compute the scores
                if (this->verbose)
                    cerr << "Running detector for " << classname << endl;
                vector<ScalarMatrix> scores;
                vector<Mixture::Indices> argmaxes;
                mixture->convolve(pyramid, scores, argmaxes);
                
                // Cache the size of the models
                vector<Size> sizes(mixture->models().size());
                for (int i = 0; i < sizes.size(); ++i)
                    sizes[i] = mixture->models()[i].rootSize();
                
                // For each scale
                for (int i = 0; i < scores.size(); ++i)
                {
                    const double scale = pyramid.scales()[i];
                  
                    score = scores[i].maxCoeff(&y, &x);
                    if (score > maxScore)
                    {
                        const Size pos = pyramid.featureExtractor()->cellCoordsToPixels(Size(x / scale + 0.5, y / scale + 0.5));
                        const Size size = pyramid.featureExtractor()->cellsToPixels(Size(
                                sizes[argmaxes[i](y, x)].width / scale + 0.5,
                                sizes[argmaxes[i](y, x)].height / scale + 0.5
                        ));
                        Rectangle bndbox(pos.width, pos.height, size.width, size.height);
                          
                        // Truncate the object
                        bndbox.setX(max(bndbox.x(), 0));
                        bndbox.setY(max(bndbox.y(), 0));
                        bndbox.setWidth(min(bndbox.width(), image.width() - bndbox.x()));
                        bndbox.setHeight(min(bndbox.height(), image.height() - bndbox.y()));
                          
                        if (!bndbox.empty())
                        {
                            detection = Detection(score, scale, x, y, bndbox, classname, synsetId, modelIndex);
                            maxScore = score;
                        }
                    }
                }

            }
     
        if (this->verbose)
            cerr << "Computed the convolutions and distance transforms in " << stop() << " ms" << endl;

    }
    
    return ARTOS_RES_OK;
}

int DPMDetection::initPatchwork(unsigned int rows, unsigned int cols, unsigned int numFeatures)
{
    // Initialize the Patchwork class (only when necessary)
    const Size maxFilterSize = this->maxModelSize(); // the Mixture class will add padding according to the filter size
    int w = (rows + maxFilterSize.height + 2 + 15) & ~15;
    int h = (cols + maxFilterSize.width + 2 + 15) & ~15;
    if ( w > Patchwork::MaxCols() || h > Patchwork::MaxRows() || numFeatures != Patchwork::NumFeatures() )
    {
        w = max(w, Patchwork::MaxCols());
        h = max(h, Patchwork::MaxRows());
        if (this->verbose) {
            cerr << "Init values for Patchwork: " << w << " x " << h << " x " << numFeatures << endl;
            start();
        }

        if (!Patchwork::Init(w, h, numFeatures)) {
            if (this->verbose)
                cerr << "\nCould not initialize the Patchwork class" << endl;
            return ARTOS_RES_INTERNAL_ERROR;
        }
        if (this->verbose) {
            cerr << "Initialized FFTW in " << stop() << " ms" << endl;
            start();
        }
        
        // Cache filters
        for ( map<std::string, Mixture *>::iterator i = this->mixtures.begin(); i != this->mixtures.end(); i++ )
            i->second->cacheFilters();
        if (this->verbose) 
            cerr << "Transformed the filters in " << stop() << " ms" << endl;
    }
    return ARTOS_RES_OK;
}

int DPMDetection::addModels ( const std::string & modellistfn )
{
    ifstream ifs ( modellistfn.c_str(), ifstream::in);

    if ( !ifs.good() )
    {
        if (this->verbose)
            cerr << "Unable to open " << modellistfn << endl;
        return ARTOS_DETECT_RES_INVALID_MODEL_LIST_FILE;
    }
    
    // Change working directory to the location of the list file to allow relative paths
    string wd = get_cwd();
    change_cwd(extract_dirname(real_path(modellistfn)));

    int numAdded = 0;
    while ( !ifs.eof() )
    {
        string classname;
        string modelfile;
        string synsetId;
        string buf;
        double threshold;
        if ( !( ifs >> classname ) ) break;
        if ( classname[0] == '#')
        {
            ifs.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }
        if ( classname[0] == '"' )
        {
            do
            {
                if ( !( ifs >> buf ) ) break;
                classname += " " + buf;
            }
            while ( classname[classname.length() - 1] != '"');
            classname = classname.substr(1, classname.length() - 2);
        }
        if ( !( ifs >> modelfile ) ) break;
        if ( modelfile[0] == '"' )
        {
            do
            {
                if ( !( ifs >> buf ) ) break;
                modelfile += " " + buf;
            }
            while ( modelfile[modelfile.length() - 1] != '"');
            modelfile = modelfile.substr(1, modelfile.length() - 2);
        }
        if ( !( ifs >> threshold ) ) break;
        while (!ifs.eof() && (ifs.peek() == ' ' || ifs.peek() == '\t' || ifs.peek() == '\r'))
            ifs.ignore();
        if (ifs.eof() || ifs.peek() == '\n')
            synsetId = "";
        else
            ifs >> synsetId;
        if (this->verbose)
            cerr << "Adding a model for " << classname << " with threshold " << threshold << endl;
        if (addModel ( classname, modelfile, threshold, synsetId ) == 0)
            numAdded++;
    }

    change_cwd(wd); // Restore working directory
    ifs.close();

    return numAdded; 
}
