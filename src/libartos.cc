#include "libartos.h"
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include "DPMDetection.h"
#include "ImageNetModelLearner.h"
#include "ImageRepository.h"
#include "sysutils.h"
using namespace std;
using namespace FFLD;
using namespace ARTOS;


//-------------------------------------------------------------------
//---------------------------- Detecting ----------------------------
//-------------------------------------------------------------------


vector<DPMDetection*> detectors;

int detect_jpeg(const unsigned int detector, const JPEGImage & img, FlatDetection * detection_buf, unsigned int * detection_buf_size);
void write_results_to_buffer(const vector<Detection> & detections, FlatDetection * detection_buf, unsigned int * detection_buf_size);
bool is_valid_detector_handle(const unsigned int detector);


unsigned int create_detector(const double overlap, const int padding, const int interval, const bool debug)
{
    DPMDetection * newDetector = 0;
    try
    {
        newDetector = new DPMDetection(debug, overlap, padding, interval);
        detectors.push_back(newDetector);
        return detectors.size(); // return handle of the new detector
    }
    catch (exception e)
    {
        if (newDetector != 0)
            delete newDetector;
        return 0; // allocation error
    }
}

void destroy_detector(const unsigned int detector)
{
    if (is_valid_detector_handle(detector))
        try
        {
            delete detectors[detector - 1];
            detectors[detector - 1] = NULL;
        }
        catch (exception e) { }
}
    
int add_model(const unsigned int detector, const char * classname, const char * modelfile, const double threshold, const char * synset_id)
{
    if (is_valid_detector_handle(detector))
        return detectors[detector - 1]->addModel(classname, modelfile, threshold, (synset_id != NULL) ? synset_id : "");
    else
        return ARTOS_RES_INVALID_HANDLE;
}

int add_models(const unsigned int detector, const char * modellistfile)
{
    if (is_valid_detector_handle(detector))
        return detectors[detector - 1]->addModels(modellistfile);
    else
        return ARTOS_RES_INVALID_HANDLE;
}

int detect_file_jpeg(const unsigned int detector,
                             const char * imagefile,
                             FlatDetection * detection_buf, unsigned int * detection_buf_size)
{
    return detect_jpeg(detector, JPEGImage(imagefile), detection_buf, detection_buf_size);
}

int detect_raw(const unsigned int detector,
                       const unsigned char * img_data, const unsigned int img_width, const unsigned int img_height, const bool grayscale,
                       FlatDetection * detection_buf, unsigned int * detection_buf_size)
{
    return detect_jpeg(detector, JPEGImage(img_width, img_height, (grayscale) ? 1 : 3, img_data), detection_buf, detection_buf_size);
}


int detect_jpeg(const unsigned int detector, const JPEGImage & img, FlatDetection * detection_buf, unsigned int * detection_buf_size)
{
    if (is_valid_detector_handle(detector))
    {
        if (img.empty())
            return ARTOS_DETECT_RES_INVALID_IMG_DATA;
        vector<Detection> detections;
        int result = detectors[detector - 1]->detect(img, detections);
        if (result == ARTOS_RES_OK)
        {
            sort(detections.begin(), detections.end());
            write_results_to_buffer(detections, detection_buf, detection_buf_size);
        }
        else
            *detection_buf_size = 0;
        return result;
    }
    else
        return ARTOS_RES_INVALID_HANDLE;
}

void write_results_to_buffer(const vector<Detection> & detections, FlatDetection * detection_buf, unsigned int * detection_buf_size)
{
    vector<Detection>::const_iterator dit; // iterator over detection results
    unsigned int bi; // index for accessing the detection buffer
    for (dit = detections.begin(), bi = 0; dit < detections.end() && bi < *detection_buf_size; dit++, bi++, detection_buf++)
    {
        // Write detection data to the detection buffer
        memset(detection_buf->classname, 0, sizeof(detection_buf->classname));
        dit->classname.copy(detection_buf->classname, sizeof(detection_buf->classname) - 1);
        memset(detection_buf->synset_id, 0, sizeof(detection_buf->synset_id));
        dit->synsetId.copy(detection_buf->synset_id, sizeof(detection_buf->synset_id) - 1);
        detection_buf->score = static_cast<float>(dit->score);
        detection_buf->left = dit->left();
        detection_buf->top = dit->top();
        detection_buf->right = dit->right() + 1;
        detection_buf->bottom = dit->bottom() + 1;
    }
    *detection_buf_size = bi; // store number of detections written to the buffer
}

bool is_valid_detector_handle(const unsigned int detector)
{
    return (detector > 0 && detector <= detectors.size() && detectors[detector - 1] != NULL);
}


//------------------------------------------------------------------
//---------------------------- Training ----------------------------
//------------------------------------------------------------------


typedef struct {
    overall_progress_cb_t cb;
    unsigned int overall_step;
} progress_params;

void populate_progress(unsigned int cur, unsigned int total, void * data)
{
    progress_params * params = reinterpret_cast<progress_params*>(data);
    params->cb(params->overall_step, 3, cur, total);
}


int learn_imagenet(const char * repo_directory, const char * synset_id, const char * bg_file, const char * modelfile,
                            const bool add, const unsigned int max_aspect_clusters, const unsigned int max_who_clusters,
                            const unsigned int th_opt_num_positive, const unsigned int th_opt_num_negative, const bool th_opt_loocv,
                            overall_progress_cb_t progress_cb, const bool debug)
{
    // Check repository
    if (!ImageRepository::hasRepositoryStructure(repo_directory))
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    // Search synset
    ImageRepository repo = ImageRepository(repo_directory);
    Synset synset = repo.getSynset(synset_id);
    if (synset.id.empty())
        return ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND;
    // Load background statistics
    StationaryBackground bg(bg_file);
    if (bg.empty())
        return ARTOS_LEARN_RES_INVALID_BG_FILE;
    
    // Setup some stuff for progress callback
    progress_params progParams;
    progParams.cb = progress_cb;
    progParams.overall_step = 0;
    if (progress_cb != NULL)
        progress_cb(0, 3, 0, 0);
    
    // Learn model
    ImageNetModelLearner learner(bg, repo, th_opt_loocv, debug);
    learner.addPositiveSamplesFromSynset(synset);
    progParams.overall_step++;
    if (!learner.learn(max_aspect_clusters, max_who_clusters, (progress_cb != NULL) ? &populate_progress : NULL, reinterpret_cast<void*>(&progParams)))
        return ARTOS_LEARN_RES_FAILED;
    progParams.overall_step++;
    learner.optimizeThreshold(th_opt_num_positive, th_opt_num_negative, 1.0f,
                              (progress_cb != NULL) ? &populate_progress : NULL, reinterpret_cast<void*>(&progParams));
    
    // Save model
    if (!learner.save(modelfile, add))
        return ARTOS_RES_FILE_ACCESS_DENIED;
    
    if (progress_cb != NULL)
        progress_cb(3, 3, 0, 0);
    return ARTOS_RES_OK;
}

int learn_files_jpeg(const char ** imagefiles, const unsigned int num_imagefiles, const FlatBoundingBox * bounding_boxes,
                            const char * bg_file, const char * modelfile, const bool add,
                            const unsigned int max_aspect_clusters, const unsigned int max_who_clusters,
                            const bool th_opt_loocv,
                            overall_progress_cb_t progress_cb, const bool debug)
{
    // Load background statistics
    StationaryBackground bg(bg_file);
    if (bg.empty())
        return ARTOS_LEARN_RES_INVALID_BG_FILE;
    
    // Setup some stuff for progress callback
    progress_params progParams;
    progParams.cb = progress_cb;
    progParams.overall_step = 0;
    if (progress_cb != NULL)
        progress_cb(0, 3, 0, 0);
    
    // Add samples
    ModelLearner learner(bg, th_opt_loocv, debug);
    FFLD::Rectangle bbox; // empty bounding box
    const FlatBoundingBox * flat_bbox;
    for (unsigned int i = 0; i < num_imagefiles; i++)
    {
        JPEGImage img(imagefiles[i]);
        if (!img.empty())
        {
            if (bounding_boxes != NULL)
            {
                flat_bbox = bounding_boxes + i;
                bbox = FFLD::Rectangle(flat_bbox->left, flat_bbox->top, flat_bbox->width, flat_bbox->height);
            }
            learner.addPositiveSample(img, bbox);
        }
    }
    progParams.overall_step++;
    
    // Learn model
    if (!learner.learn(max_aspect_clusters, max_who_clusters, (progress_cb != NULL) ? &populate_progress : NULL, reinterpret_cast<void*>(&progParams)))
        return ARTOS_LEARN_RES_FAILED;
    progParams.overall_step++;
    if (progress_cb == NULL)
        learner.optimizeThreshold();
    else
        learner.optimizeThreshold(0, NULL, 1.0f, &populate_progress, reinterpret_cast<void*>(&progParams));
    
    // Save model
    if (!learner.save(modelfile, add))
        return ARTOS_RES_FILE_ACCESS_DENIED;
    
    if (progress_cb != NULL)
        progress_cb(3, 3, 0, 0);
    return ARTOS_RES_OK;
}


vector<ImageNetModelLearner*> learners;

bool is_valid_learner_handle(const unsigned int learner);
int learner_add_jpeg(const unsigned int learner, const JPEGImage & img, const FlatBoundingBox * bboxes, const unsigned int num_bboxes);
void progress_proxy(unsigned int current, unsigned int total, void * data);

unsigned int create_learner(const char * bg_file, const char * repo_directory, const bool th_opt_loocv, const bool debug)
{
    if (!ImageRepository::hasRepositoryStructure(repo_directory))
        repo_directory = "";
    ImageNetModelLearner * newLearner = new ImageNetModelLearner(bg_file, repo_directory, th_opt_loocv, debug);
    if (newLearner->getBackground().empty())
    {
        delete newLearner;
        return 0;
    }
    learners.push_back(newLearner);
    return learners.size(); // return handle of the new detector
}

void destroy_learner(const unsigned int learner)
{
    if (is_valid_learner_handle(learner))
        try
        {
            delete learners[learner - 1];
            learners[learner - 1] = NULL;
        }
        catch (exception e) { }
}

int learner_add_synset(const unsigned int learner, const char * synset_id, const unsigned int max_samples)
{
    if (!is_valid_learner_handle(learner))
        return ARTOS_RES_INVALID_HANDLE;
    ImageRepository repo = learners[learner - 1]->getRepository();
    if (repo.getRepoDirectory().empty())
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    // Search synset
    Synset synset = repo.getSynset(synset_id);
    if (synset.id.empty())
        return ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND;
    learners[learner - 1]->addPositiveSamplesFromSynset(synset, max_samples);
    return ARTOS_RES_OK;
}

int learner_add_file_jpeg(const unsigned int learner, const char * imagefile,
                          const FlatBoundingBox * bboxes, const unsigned int num_bboxes)
{
    return learner_add_jpeg(learner, JPEGImage(imagefile), bboxes, num_bboxes);
}

int learner_add_raw(const unsigned int learner,
                    const unsigned char * img_data, const unsigned int img_width, const unsigned int img_height, const bool grayscale,
                    const FlatBoundingBox * bboxes, const unsigned int num_bboxes)
{
    return learner_add_jpeg(learner, JPEGImage(img_width, img_height, (grayscale) ? 1 : 3, img_data), bboxes, num_bboxes);
}

int learner_run(const unsigned int learner, const unsigned int max_aspect_clusters, const unsigned int max_who_clusters, progress_cb_t progress_cb)
{
    if (!is_valid_learner_handle(learner))
        return ARTOS_RES_INVALID_HANDLE;
    ImageNetModelLearner * l = learners[learner - 1];
    if (l->getNumSamples() == 0)
        return ARTOS_LEARN_RES_NO_SAMPLES;
    ProgressCallback progressCB = (progress_cb != NULL) ? &progress_proxy : NULL;
    void * cbData = (progress_cb != NULL) ? reinterpret_cast<void*>(progress_cb) : NULL;
    return (l->learn(max_aspect_clusters, max_who_clusters, progressCB, cbData)) ? ARTOS_RES_OK : ARTOS_LEARN_RES_FAILED;
}

int learner_optimize_th(const unsigned int learner, const unsigned int max_positive, const unsigned int num_negative, progress_cb_t progress_cb)
{
    if (!is_valid_learner_handle(learner))
        return ARTOS_RES_INVALID_HANDLE;
    ImageNetModelLearner * l = learners[learner - 1];
    if (l->getModels().empty())
        return ARTOS_LEARN_RES_MODEL_NOT_LEARNED;
    if (num_negative > 0 && l->getRepository().getRepoDirectory().empty())
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    ProgressCallback progressCB = (progress_cb != NULL) ? &progress_proxy : NULL;
    void * cbData = (progress_cb != NULL) ? reinterpret_cast<void*>(progress_cb) : NULL;
    l->optimizeThreshold(max_positive, num_negative, 1.0f, progressCB, cbData);
    return ARTOS_RES_OK;
}

int learner_save(const unsigned int learner, const char * modelfile, const bool add)
{
    if (!is_valid_learner_handle(learner))
        return ARTOS_RES_INVALID_HANDLE;
    ImageNetModelLearner * l = learners[learner - 1];
    if (l->getModels().empty())
        return ARTOS_LEARN_RES_MODEL_NOT_LEARNED;
    return (l->save(modelfile, add)) ? ARTOS_RES_OK : ARTOS_RES_FILE_ACCESS_DENIED;
}

int learner_reset(const unsigned int learner)
{
    if (!is_valid_learner_handle(learner))
        return ARTOS_RES_INVALID_HANDLE;
    learners[learner - 1]->reset();
    return ARTOS_RES_OK;
}


bool is_valid_learner_handle(const unsigned int learner)
{
    return (learner > 0 && learner <= learners.size() && learners[learner - 1] != NULL);
}

int learner_add_jpeg(const unsigned int learner, const JPEGImage & img, const FlatBoundingBox * bboxes, const unsigned int num_bboxes)
{
    if (!is_valid_learner_handle(learner))
        return ARTOS_RES_INVALID_HANDLE;
    if (img.empty())
        return ARTOS_LEARN_RES_INVALID_IMG_DATA;
    vector<FFLD::Rectangle> _bboxes;
    if (bboxes != NULL)
        for (const FlatBoundingBox * flat_bbox = bboxes; flat_bbox < bboxes + num_bboxes; flat_bbox++)
            _bboxes.push_back(FFLD::Rectangle(flat_bbox->left, flat_bbox->top, flat_bbox->width, flat_bbox->height));
    learners[learner - 1]->addPositiveSample(img, _bboxes);
    return ARTOS_RES_OK;
}

void progress_proxy(unsigned int current, unsigned int total, void * data)
{
    progress_cb_t cb = reinterpret_cast<progress_cb_t>(data);
    cb(current, total);
}



//------------------------------------------------------------------
//---------------------------- ImageNet ----------------------------
//------------------------------------------------------------------


int list_synsets(const char * repo_directory, SynsetSearchResult * synset_buf, unsigned int * synset_buf_size)
{
    if (!ImageRepository::hasRepositoryStructure(repo_directory))
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    ImageRepository repo(repo_directory);
    if (synset_buf == NULL || *synset_buf_size == 0)
        *synset_buf_size = repo.getNumSynsets();
    else
    {
        vector<string> ids, descriptions;
        repo.listSynsets(&ids, &descriptions);
        size_t i;
        for (i = 0; i < ids.size() && i < *synset_buf_size; i++, synset_buf++)
        {
            memset(synset_buf->synsetId, 0, sizeof(synset_buf->synsetId));
            ids[i].copy(synset_buf->synsetId, sizeof(synset_buf->synsetId) - 1);
            memset(synset_buf->description, 0, sizeof(synset_buf->description));
            descriptions[i].copy(synset_buf->description, sizeof(synset_buf->description) - 1);
            synset_buf->score = 0;
        }
        *synset_buf_size = i;
    }
    return ARTOS_RES_OK;
}

int search_synsets(const char * repo_directory, const char * phrase, SynsetSearchResult * result_buf, unsigned int * result_buf_size)
{
    if (!ImageRepository::hasRepositoryStructure(repo_directory))
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    ImageRepository repo(repo_directory);
    vector<Synset> results;
    vector<float> scores;
    repo.searchSynsets(phrase, results, *result_buf_size, &scores);
    size_t i;
    for (i = 0; i < results.size() && i < *result_buf_size; i++, result_buf++)
    {
        memset(result_buf->synsetId, 0, sizeof(result_buf->synsetId));
        results[i].id.copy(result_buf->synsetId, sizeof(result_buf->synsetId) - 1);
        memset(result_buf->description, 0, sizeof(result_buf->description));
        results[i].description.copy(result_buf->description, sizeof(result_buf->description) - 1);
        result_buf->score = scores[i];
    }
    *result_buf_size = i;
    return ARTOS_RES_OK;
}

int extract_images_from_synset(const char * repo_directory, const char * synset_id, const char * out_directory, unsigned int * num_images)
{
    // Check repository
    if (!ImageRepository::hasRepositoryStructure(repo_directory))
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    // Check output directory
    if (!is_dir(out_directory))
        return ARTOS_RES_DIRECTORY_NOT_FOUND;
    // Check pointer parameters
    if (num_images == NULL)
        return ARTOS_RES_OK; // extract nothing
    
    // Search synset
    Synset synset = ImageRepository(repo_directory).getSynset(synset_id);
    if (synset.id.empty())
        return ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND;
        
    // Extract
    SynsetImageIterator imgIt = synset.getImageIterator();
    for (; imgIt.ready() && (unsigned int) imgIt < *num_images; ++imgIt)
    {
        SynsetImage simg = *imgIt;
        JPEGImage img = simg.getImage();
        if (!img.empty())
            img.save(join_path(2, out_directory, (simg.getFilename() + ".jpg").c_str()));
    }
    *num_images = imgIt.pos();
    return ARTOS_RES_OK;
}

int extract_samples_from_synset(const char * repo_directory, const char * synset_id, const char * out_directory, unsigned int * num_samples)
{
    // Check repository
    if (!ImageRepository::hasRepositoryStructure(repo_directory))
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    // Check output directory
    if (!is_dir(out_directory))
        return ARTOS_RES_DIRECTORY_NOT_FOUND;
    // Check pointer parameters
    if (num_samples == NULL)
        return ARTOS_RES_OK; // extract nothing
    
    // Search synset
    Synset synset = ImageRepository(repo_directory).getSynset(synset_id);
    if (synset.id.empty())
        return ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND;
        
    // Extract
    SynsetImageIterator imgIt = synset.getImageIterator(true);
    unsigned int count = 0;
    char extBuf[10];
    vector<JPEGImage> samples;
    for (; imgIt.ready() && count < *num_samples; ++imgIt)
    {
        SynsetImage simg = *imgIt;
        samples.clear();
        simg.getSamplesFromBoundingBoxes(samples);
        for (size_t i = 0; i < samples.size() && count < *num_samples; i++)
        {
            sprintf(extBuf, "_%lu.jpg", i + 1);
            samples[i].save(join_path(2, out_directory, (simg.getFilename() + extBuf).c_str()));
            count++;
        }
    }
    *num_samples = count;
    return ARTOS_RES_OK;
}

int extract_mixed_images(const char * repo_directory, const char * out_directory, const unsigned int num_images, const unsigned int per_synset)
{
    // Check repository
    if (!ImageRepository::hasRepositoryStructure(repo_directory))
        return ARTOS_IMGREPO_RES_INVALID_REPOSITORY;
    // Check output directory
    string out_dir(out_directory);
    if (!is_dir(out_dir))
        return ARTOS_RES_DIRECTORY_NOT_FOUND;
    
    // Extract
    MixedImageIterator imgIt = ImageRepository(repo_directory).getMixedIterator(per_synset);
    for (; imgIt.ready() && (unsigned int) imgIt < num_images; ++imgIt)
        imgIt.extract(out_dir);
    return ARTOS_RES_OK;
}
