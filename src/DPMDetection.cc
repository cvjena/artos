#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits>

#include "DPMDetection.h"
#include "sysutils.h"
#include "ffld/Intersector.h"
#include "ffld/timingtools.h"

using namespace ARTOS;
using namespace FFLD;
using namespace std;

DPMDetection::DPMDetection ( bool verbose, double overlap, int padding, int interval )
{
    init ( verbose, overlap, padding, interval );
}

DPMDetection::DPMDetection ( const std::string & modelfile, double threshold, bool verbose, double overlap, int padding, int interval ) 
{
    init ( verbose, overlap, padding, interval );
    addModel ( "single", modelfile, threshold );
}

DPMDetection::DPMDetection ( const Mixture & model, double threshold, bool verbose, double overlap, int padding, int interval ) 
{
    init ( verbose, overlap, padding, interval );
    addModel ( "single", model, threshold );
}

DPMDetection::DPMDetection ( Mixture && model, double threshold, bool verbose, double overlap, int padding, int interval ) 
{
    init ( verbose, overlap, padding, interval );
    addModel ( "single", move(model), threshold );
}

void DPMDetection::init ( bool verbose, double overlap, int padding, int interval )
{
    this->overlap = overlap;
    this->padding = padding;
    this->interval = interval;
    this->verbose = verbose;
    this->initw = -1;
    this->inith = -1;
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
    
    Mixture *mixture = new Mixture();
    in >> (*mixture);

    if (mixture->empty()) {
        if (this->verbose)
            cerr << "\nInvalid model file " << modelfile << endl;
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
    
    this->initw = -1;
    this->inith = -1;

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

    // Compute the HOG features
    if (this->verbose)
        start();

    HOGPyramid pyramid(image, this->padding, this->padding, this->interval);

    if (pyramid.empty()) {
        if (this->verbose)
            cerr << "\nInvalid image!" << endl;
        return ARTOS_DETECT_RES_INVALID_IMAGE;
    }

    if (this->verbose) {
        cerr << "Computed HOG features in " << stop() << " ms for an image of size " <<
                image.width() << " x " << image.height() << endl;
    }

    int errcode = this->initPatchwork(pyramid.levels()[0].rows(), pyramid.levels()[0].cols());
    if (errcode != ARTOS_RES_OK)
        return errcode;

    if ( this->verbose )
        start();

    errcode = this->detect( image.width(), image.height(), pyramid, detections);
 
    if (this->verbose)
        cerr << "Computed the convolutions and distance transforms in " << stop() << " ms" << endl;

    return errcode;
}

int DPMDetection::detect(int width, int height, const HOGPyramid & pyramid, vector<Detection> & detections)
{
    for ( map<std::string, Mixture *>::iterator i = this->mixtures.begin(); i != this->mixtures.end(); i++ )
    {
        Mixture *mixture = i->second;
        const std::string & classname = i->first;
        double threshold = thresholds[classname];
        const std::string & synsetId = synsetIds[classname];
        unsigned int modelIndex = modelIndices[classname];

        // Compute the scores
        vector<HOGPyramid::Matrix> scores;
        vector<Mixture::Indices> argmaxes;
        vector<Detection> single_detections;
        
        // there is also a version which allows obtaining the part positions, but who cares :)
        // see the original ffld code for this call
        mixture->convolve(pyramid, scores, argmaxes);
        
        if (this->verbose)
            cerr << "Running detector for " << classname << endl;
        
        // Cache the size of the models
        vector<pair<int, int> > sizes(mixture->models().size());
        
        for (int i = 0; i < sizes.size(); ++i)
            sizes[i] = mixture->models()[i].rootSize();
        
        // For each scale
        for (int i = pyramid.interval(); i < scores.size(); ++i)
        {
            // Scale = 8 / 2^(1 - i / interval)
            const double scale = pow(2.0, static_cast<double>(i) / pyramid.interval() + 2.0);
          
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
                            FFLD::Rectangle bndbox((int)((x - pyramid.padx()) * scale + 0.5),
                                       (int)((y - pyramid.pady()) * scale + 0.5),
                                       (int)(sizes[argmaxes[i](y, x)].second * scale + 0.5),
                                       (int)(sizes[argmaxes[i](y, x)].first * scale + 0.5));
                  
                            // Truncate the object
                            bndbox.setX(max(bndbox.x(), 0));
                            bndbox.setY(max(bndbox.y(), 0));
                            bndbox.setWidth(min(bndbox.width(), width - bndbox.x()));
                            bndbox.setHeight(min(bndbox.height(), height - bndbox.y()));
                              
                            if (!bndbox.empty())
                                single_detections.push_back(Detection(score, i, x, y, bndbox, classname, synsetId, modelIndex));

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

    // Compute the HOG features
    if (this->verbose)
        start();

    HOGPyramid pyramid(image, this->padding, this->padding, this->interval);

    if (pyramid.empty()) {
        if (this->verbose)
            cerr << "\nInvalid image!" << endl;
        return ARTOS_DETECT_RES_INVALID_IMAGE;
    }

    if (this->verbose) {
        cerr << "Computed HOG features in " << stop() << " ms for an image of size " <<
                image.width() << " x " << image.height() << endl;
    }

    int errcode = this->initPatchwork(pyramid.levels()[0].rows(), pyramid.levels()[0].cols());
    if (errcode != ARTOS_RES_OK)
        return errcode;

    if ( this->verbose )
        start();

    HOGPyramid::Scalar score, maxScore = -1 * numeric_limits<HOGPyramid::Scalar>::infinity();
    int y, x;
    for ( map<std::string, Mixture *>::iterator i = this->mixtures.begin(); i != this->mixtures.end(); i++ )
    {
        Mixture *mixture = i->second;
        const std::string & classname = i->first;
        const std::string & synsetId = synsetIds[classname];
        unsigned int modelIndex = modelIndices[classname];

        // Compute the scores
        vector<HOGPyramid::Matrix> scores;
        vector<Mixture::Indices> argmaxes;
        
        // there is also a version which allows obtaining the part positions, but who cares :)
        // see the original ffld code for this call
        mixture->convolve(pyramid, scores, argmaxes);
        
        if (this->verbose)
            cerr << "Running detector for " << classname << endl;
        
        // Cache the size of the models
        vector<pair<int, int> > sizes(mixture->models().size());
        
        for (int i = 0; i < sizes.size(); ++i)
            sizes[i] = mixture->models()[i].rootSize();
        
        // For each scale
        for (int i = pyramid.interval(); i < scores.size(); ++i)
        {
            // Scale = 8 / 2^(1 - i / interval)
            const double scale = pow(2.0, static_cast<double>(i) / pyramid.interval() + 2.0);
          
            score = scores[i].maxCoeff(&y, &x);
            if (score > maxScore)
            {
                FFLD::Rectangle bndbox((int)((x - pyramid.padx()) * scale + 0.5),
                                       (int)((y - pyramid.pady()) * scale + 0.5),
                                       (int)(sizes[argmaxes[i](y, x)].second * scale + 0.5),
                                       (int)(sizes[argmaxes[i](y, x)].first * scale + 0.5));
                  
                // Truncate the object
                bndbox.setX(max(bndbox.x(), 0));
                bndbox.setY(max(bndbox.y(), 0));
                bndbox.setWidth(min(bndbox.width(), image.width() - bndbox.x()));
                bndbox.setHeight(min(bndbox.height(), image.height() - bndbox.y()));
                  
                if (!bndbox.empty())
                {
                    detection = Detection(score, i, x, y, bndbox, classname, synsetId, modelIndex);
                    maxScore = score;
                }
            }
        }

    }
 
    if (this->verbose)
        cerr << "Computed the convolutions and distance transforms in " << stop() << " ms" << endl;

    return ARTOS_RES_OK;
}

int DPMDetection::initPatchwork(unsigned int rows, unsigned int cols)
{
    // Initialize the Patchwork class (only when necessary)
    int w = (rows - this->padding + 15) & ~15;
    int h = (cols - this->padding + 15) & ~15;
    if ( w != initw || h != inith )
    {
        if (this->verbose) {
            cerr << "Init values for Patchwork: " << w << " x " << h << endl;
            start();
        }

        if (!Patchwork::Init(w,h)) {
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
        {
            Mixture *mixture = i->second;
            mixture->cacheFilters();
        }
        if (this->verbose) 
            cerr << "Transformed the filters in " << stop() << " ms" << endl;

        initw = w;
        inith = h;
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
