/**
* @file
*
* Procedural C-style interface to ARTOS, the Adaptive Real-Time Object Detection system.
*
* This file provides a procedural interface to the object detection, whitened HOG model learning
* and image retrieval functionalities of ARTOS for integrating libartos into other projects
* created with any programming language.
*
* The **detection interface** is handle-controlled. First, you create a detector instance by calling
* `create_detector`, which returns a (numerical) handle. Any other detection related functions take
* this handle as first parameter and will operate on that detector instance.  
* Add some models to the detector using either `add_model` or `add_models` and run the detector on
* your images with one of the `detect_*` functions.  
* Don't forget to destroy the detector instance using `destroy_detector` if you don't need it any more.
*
* To **learn new models** either the all-in-one short-cut functions like `learn_imagenet` or `learn_files_jpeg`
* can be used or the handle-controlled learner interface, which is more complex, but gives you more control
* over the details of the model learning process. A learner instance is created by calling `create_learner`.
* After that, samples can be added either from ImageNet, from file or from raw pixel data using one of the
* `learner_add_*` functions. The actual learning step is performed by `learner_run` and, optionally,
* `learner_optimize_th`. Finally, the learned model can be saved with `learner_save`.  
* The all-in-one short-cut functions mentioned above handle all those steps with just one function call.
*
* The **ImageNet interface** provides functionality for searching synsets and extracting whole images
* as well as samples of objects from them to disk. In-memory extraction as provided by the object-oriented
* interface, `ImageRepository` and it's related classes, is not supported in this C-style library interface.  
* In the default implementation, all images and annotations from ImageNet have to be downloaded and made
* available on the filesystem (that are some terabytes of data). See the documentation of the `ImageRepository`
* class for how to structure your local copy of ImageNet.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/

#ifndef LIBARTOS_H
#define LIBARTOS_H

#include "libartos_def.h"

#ifdef __cplusplus
extern "C"
{
#endif


//-------------------
//     Detection
//-------------------

/**
* Holds information about a detection on an image, such as the name of the detected class, the ID of the
* associated synset the detection score and the bounding box, while `left` and `top` are inside,
* but `right` and `bottom` just outside the rectangle, so that it's width can be calculated as `right - left`.
*/
typedef struct {
    char classname[44];
    char synset_id[16];
    float score;
    int left;
    int top;
    int right;
    int bottom;
} FlatDetection;

/**
* Creates a new detector instance.
* @param[in] overlap Minimum overlap in non maxima suppression.
* @param[in] padding Amount of zero padding in HOG cells. Must be greater or equal to half the greatest filter dimension.
* @param[in] interval Number of levels per octave in the HOG pyramid.
* @param[in] debug If this is set to true, debug and performance information will be printed to stderr.
* @return Handle to the new detector instance or 0 if the memory for the instance could not be allocated.
*/
unsigned int create_detector(const double overlap = 0.5, const int padding = 12, const int interval = 10, const bool debug = false);

/**
* Frees a detector instance.
* @param[in] detector The handle of the detector instance obtained by create_detector().
*/
void destroy_detector(const unsigned int detector);

/**
* Adds a model to the detection stack of a detector instance.
* @param[in] detector The handle of the detector instance obtained by create_detector().
* @param[in] classname The name of the class ('bicycle' for example). It is used to name the objects detected in an image.
* @param[in] modelfile The filename of the model to load.
* @param[in] threshold The detection threshold for this model.
* @param[in] synset_id Optionally, the ID of the ImageNet synset associated with this model. May be NULL.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*           - `ARTOS_RES_INVALID_HANDLE`
*           - `ARTOS_DETECT_RES_INVALID_MODEL_FILE`
*/
int add_model(const unsigned int detector, const char * classname, const char * modelfile, const double threshold, const char * synset_id = 0);

/**
* Adds multiple models to the detection stack of a detector instance at once using information given in a file enumerating the models.
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
* @param[in] detector The handle of the detector instance obtained by create_detector().
* @param[in] modellistfile The filename of the model list.
* @return Returns the (non-negative) number of successfully added models or one of the following (negative) error codes on failure:
*           - `ARTOS_RES_INVALID_HANDLE`
*           - `ARTOS_DETECT_RES_INVALID_MODEL_LIST_FILE`
*/
int add_models(const unsigned int detector, const char * modellistfile);

/**
* Detects objects in a JPEG image file which match one of the models added before using add_model() or add_models().
* @param[in] detector The handle of the detector instance obtained by create_detector().
* @param[in] imagefile The filename of the JPEG image.
* @param[out] detection_buf A beforehand allocated buffer array of FlatDetection structs, that will be filled up with the
*                           detection results ordered descending by their detection score.
* @param[in,out] detection_buf_size The number of allocated array slots of `detection_buf`.
*                                   In turn, the number of actually stored results will be written to this pointer's location.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*           - `ARTOS_RES_INVALID_HANDLE`
*           - `ARTOS_DETECT_RES_NO_MODELS`
*           - `ARTOS_DETECT_RES_INVALID_IMG_DATA` (image couldn't be read)
*           - `ARTOS_DETECT_RES_INVALID_IMAGE` (image couldn't be processed)
*           - `ARTOS_RES_INTERNAL_ERROR`
*/
int detect_file_jpeg(const unsigned int detector,
                             const char * imagefile,
                             FlatDetection * detection_buf, unsigned int * detection_buf_size);

/**
* Detects objects in an RGB or grayscale image given by raw pixel data in a buffer,
* which match one of the models added before using add_model() or add_models().
* @param[in] detector The handle of the detector instance obtained by create_detector().
* @param[in] img_data The pixel data of the image in row-major order. Row 0: R,G,B,R,G,B,...; Row 1: R,G,B,R,G,B,...; ...
* @param[in] img_width The width of the image.
* @param[in] img_height The height of the image.
* @param[in] grayscale If set to true, a bit depth of 1 byte per pixel is assumed (intensity), otherwise bit depth is set to 3 (RGB).
* @param[out] detection_buf A beforehand allocated buffer array of FlatDetection structs, that will be filled up with the
*                           detection results ordered descending by their detection score.
* @param[in,out] detection_buf_size The number of allocated array slots of `detection_buf`.
*                                   In turn, the number of actually stored results will be written to this pointer's location.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*           - `ARTOS_RES_INVALID_HANDLE`
*           - `ARTOS_DETECT_RES_NO_MODELS`
*           - `ARTOS_DETECT_RES_INVALID_IMG_DATA` (image couldn't be read)
*           - `ARTOS_DETECT_RES_INVALID_IMAGE` (image couldn't be processed)
*           - `ARTOS_RES_INTERNAL_ERROR`
*/
int detect_raw(const unsigned int detector,
                       const unsigned char * img_data, const unsigned int img_width, const unsigned int img_height, const bool grayscale,
                       FlatDetection * detection_buf, unsigned int * detection_buf_size);


//------------------
//     Learning
//------------------

/**
* A simple bounding box around an object on an image.
*/
typedef struct {
    unsigned int left;
    unsigned int top;
    unsigned int width;
    unsigned int height;
} FlatBoundingBox;

typedef void (*progress_cb_t)(unsigned int, unsigned int);
typedef void (*overall_progress_cb_t)(unsigned int, unsigned int, unsigned int, unsigned int);


/**
* Learns a new model from the image and bounding box annotation data of a synset of an image repository.
* @param[in] repo_directory The path to the image repository directory.
* @param[in] synset_id The ID of the synset.
* @param[in] modelfile Path to file file which the newly learned models will be written to
*                      (the file will be created if it does not exist).
* @param[in] bg_file Path to a file containing the stationary background statistics for whitening (negative mean and covariance).
* @param[in] add If set to true, the new models will be added as additional mixture components if the model file
*                does already exist, otherwise the model file will be overwritten with just the new mixture.
* @param[in] max_aspect_clusters Maximum number of clusters formed by considering the aspect ratio of the input samples.
* @param[in] max_who_clusters Maximum number of clusters formed by considering the WHO vectors of samples in the same
*                             aspect ratio cluster.
* @param[in] th_opt_num_positive Maximum number of positive samples to test the model against for threshold optimization.
*                                Set this to 0 to run the detector against all samples from the synset.
* @param[in] th_opt_num_negative Maximum number of negative samples to test the model against for threshold optimization.
*                                Negative samples will be taken from different synsets.
* @param[in] th_opt_loocv If set to true, *Leave-one-out-cross-validation* will be performed for threshold optimization.
*                         This will increase memory usage, since the WHO features of all samples have to be stored for this,
*                         and will slow down threshold optimization as well as the model learning step if only one WHO cluster
*                         is used. But it will guarantee that no model is tested against a sample it has been learned from
*                         for threshold optimization.
* @param[in] progress_cb Optionally, a callback which is called between the steps of the learning process to populate the progress.  
*                        The first parameter to the callback will be the number of steps performed in the entire process,
*                        the second one will be the total number of steps. To date, the entire process is divided into three steps:
*                        image extraction, model creation and threshold optimization.  
*                        The third and fourth parameters will be the number of performed steps and the total number of steps of
*                        the current sub-procedure.
* @param[in] debug If this is set to true, debug and performance information will be printed to stderr.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_IMGREPO_RES_INVALID_REPOSITORY`
*                   - `ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND`
*                   - `ARTOS_LEARN_RES_FAILED`
*                   - `ARTOS_LEARN_RES_INVALID_BG_FILE`
*                   - `ARTOS_RES_FILE_ACCESS_DENIED` (if could not write model file)
* @note Due to clustering, the result of the learning process may be multiple models which will be combined in the same
*       model file as mixture components. The maximum number of components is given by `max_aspect_clusters * max_who_clusters`.
*/
int learn_imagenet(const char * repo_directory, const char * synset_id, const char * bg_file, const char * modelfile,
                            const bool add = true, const unsigned int max_aspect_clusters = 2, const unsigned int max_who_clusters = 3,
                            const unsigned int th_opt_num_positive = 0, const unsigned int th_opt_num_negative = 0, const bool th_opt_loocv = true,
                            overall_progress_cb_t progress_cb = 0, const bool debug = false);

/**
* Learns a new model from given JPEG files using the entire images as bounding boxes.
* @param[in] imagefiles Array of file names as null-terminated strings of the JPEG files showing the objects.
* @param[in] num_imagefiles Number of file names in `imagefiles`.
* @param[in] bounding_boxes Array of bounding boxes around objects to learn corresponding to the given images.
*                           Must have exactly `num_imagefiles` entries or must be set to NULL. In the latter case,
*                           the entire images will be considered as showing the object of interest. The same can be
*                           done for a single image by specifying an empty bounding box (i. e. a rectangle with no area).
* @param[in] modelfile Path to file file which the newly learned models will be written to
*                      (the file will be created if it does not exist).
* @param[in] bg_file Path to a file containing the stationary background statistics for whitening (negative mean and covariance).
* @param[in] add If set to true, the new models will be added as additional mixture components if the model file
*                does already exist, otherwise the model file will be overwritten with just the new mixture.
* @param[in] max_aspect_clusters Maximum number of clusters formed by considering the aspect ratio of the input samples.
* @param[in] max_who_clusters Maximum number of clusters formed by considering the WHO vectors of samples in the same
*                             aspect ratio cluster.
* @param[in] th_opt_loocv If set to true, *Leave-one-out-cross-validation* will be performed for threshold optimization.
*                         This will increase memory usage, since the WHO features of all samples have to be stored for this,
*                         and will slow down threshold optimization as well as the model learning step if only one WHO cluster
*                         is used. But it will guarantee that no model is tested against a sample it has been learned from
*                         for threshold optimization.
* @param[in] progress_cb Optionally, a callback which is called between the steps of the learning process to populate the progress.  
*                        The first parameter to the callback will be the number of steps performed in the entire process,
*                        the second one will be the total number of steps. To date, the entire process is divided into three steps:
*                        image reading & decoding, model creation and threshold optimization.  
*                        The third and fourth parameters will be the number of performed steps and the total number of steps of
*                        the current sub-procedure.
* @param[in] debug If this is set to true, debug and performance information will be printed to stderr.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_LEARN_RES_FAILED`
*                   - `ARTOS_LEARN_RES_INVALID_BG_FILE`
*                   - `ARTOS_RES_FILE_ACCESS_DENIED` (if could not write model file)
* @note Due to clustering, the result of the learning process may be multiple models which will be combined in the same
*       model file as mixture components. The maximum number of components is given by `max_aspect_clusters * max_who_clusters`.
*/
int learn_files_jpeg(const char ** imagefiles, const unsigned int num_imagefiles, const FlatBoundingBox * bounding_boxes,
                            const char * bg_file, const char * modelfile, const bool add = true,
                            const unsigned int max_aspect_clusters = 2, const unsigned int max_who_clusters = 3,
                            const bool th_opt_loocv = true,
                            overall_progress_cb_t progress_cb = 0, const bool debug = false);

/**
* Creates a new learner object, which gives you more control over the model learning process than the all-in-one short-cut
* functions like `learn_imagenet` or `learn_files_jpeg` (but those are easier to use).
*
* The positive samples can be added then by using the `learner_add_*` functions with the learner handle returned by
* this function and the actual learning step will be performed by `learner_run` and `learner_optimize_th`.
*
* @param[in] bg_file Path to a file containing the stationary background statistics for whitening (negative mean and covariance).
* @param[in] repo_directory Optionally, a path to an image repository. Required if you want to use ImageNet functions with
*                           this learner.
* @param[in] th_opt_loocv If set to true, *Leave-one-out-cross-validation* will be performed for threshold optimization.
*                         This will increase memory usage, since the WHO features of all samples have to be stored for this,
*                         and will slow down threshold optimization as well as the model learning step if only one WHO cluster
*                         is used. But it will guarantee that no model is tested against a sample it has been learned from
*                         for threshold optimization.
* @param[in] debug If this is set to true, debug and performance information will be printed to stderr.
* @return Handle to the new learner instance or 0 if the given background statistics file is invalid.
*/
unsigned int create_learner(const char * bg_file, const char * repo_directory = "", const bool th_opt_loocv = true, const bool debug = false);

/**
* Frees a learner instance.
* @param[in] learner The handle of the learner instance obtained by create_learner().
*/
void destroy_learner(const unsigned int learner);

/**
* Adds positive samples from a synset of an image repository to a learner instance.
* The bounding box annotation data of the synset is used to extract the samples from the images.
* @param[in] learner The handle of the learner instance obtained by create_learner().
* @param[in] synset_id The ID of the synset.
* @param[in] max_samples Maximum number images to extract from the synset. Set this to 0, to extract
*                        all images of the synset.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_RES_INVALID_HANDLE`
*                   - `ARTOS_IMGREPO_RES_INVALID_REPOSITORY`
*                   - `ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND`
*/
int learner_add_synset(const unsigned int learner, const char * synset_id, const unsigned int max_samples = 0);

/**
* Adds a JPEG file as positive sample to a learner instance.
* @param[in] learner The handle of the learner instance obtained by create_learner().
* @param[in] imagefile Path of the JPEG image file to be added.
* @param[in] bboxes Pointer to an array of FlatBoundingBox structs giving the bounding box(es)
*                   around the objects of interest on the given image. If this is set to NULL or
*                   one of the given bounding boxes is empty (a rectangle with zero area), the entire
*                   image will be considered to show the object.
* @param[in] num_bboxes Number of entries in the `bboxes` array. Is ignored, if `bboxes` is NULL.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_RES_INVALID_HANDLE`
*                   - `ARTOS_LEARN_RES_INVALID_IMG_DATA` (file could not be found or read)
*/
int learner_add_file_jpeg(const unsigned int learner, const char * imagefile,
                          const FlatBoundingBox * bboxes = 0, const unsigned int num_bboxes = 1);

/**
* Adds an RGB or grayscale image given by raw pixel data in a buffer as positive sample to a learner instance.
* @param[in] learner The handle of the learner instance obtained by create_learner().
* @param[in] img_data The pixel data of the image in row-major order. Row 0: R,G,B,R,G,B,...; Row 1: R,G,B,R,G,B,...; ...
* @param[in] img_width The width of the image.
* @param[in] img_height The height of the image.
* @param[in] grayscale If set to true, a bit depth of 1 byte per pixel is assumed (intensity), otherwise bit depth is set to 3 (RGB).
* @param[in] bboxes Pointer to an array of FlatBoundingBox structs giving the bounding box(es)
*                   around the objects of interest on the given image. If this is set to NULL or
*                   one of the given bounding boxes is empty (a rectangle with zero area), the entire
*                   image will be considered to show the object.
* @param[in] num_bboxes Number of entries in the `bboxes` array. Is ignored, if `bboxes` is NULL.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_RES_INVALID_HANDLE`
*                   - `ARTOS_LEARN_RES_INVALID_IMG_DATA`
*/
int learner_add_raw(const unsigned int learner,
                    const unsigned char * img_data, const unsigned int img_width, const unsigned int img_height, const bool grayscale = false,
                    const FlatBoundingBox * bboxes = 0, const unsigned int num_bboxes = 1);

/**
* Performs the actual learning step for a given learner instance.
* For an overview of our model learning method, refer to the documentation of the ModelLearner class.
* @see ModelLearner
* @param[in] learner The handle of the learner instance obtained by create_learner().
* @param[in] max_aspect_clusters Maximum number of clusters formed by considering the aspect ratio of the input samples.
* @param[in] max_who_clusters Maximum number of clusters formed by considering the WHO vectors of samples in the same
*                             aspect ratio cluster.
* @param[in] progress_cb Optionally, a callback that is called to populate the progress of the procedure.
*                        The first parameter to the callback will be the current step and the second parameter will
*                        be the total number of steps. For example, the argument list (5, 10) means that the learning
*                        is half way done.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_RES_INVALID_HANDLE`
*                   - `ARTOS_LEARN_RES_NO_SAMPLES`
*                   - `ARTOS_LEARN_RES_FAILEd`
*/
int learner_run(const unsigned int learner, const unsigned int max_aspect_clusters = 2, const unsigned int max_who_clusters = 3,
                progress_cb_t progress_cb = 0);

/**
* Tries to find the optimal thresholds for the models learned before by learner_run().
* @param[in] learner The handle of the learner instance obtained by create_learner().
* @param[in] max_positive Maximum number of positive samples to test the models against. Set this to 0
*                        to run the detector against all samples.
* @param[in] num_negative Number of negative samples taken from different synsets if an image repository has been set.
*                        Every detection on one of these images will be considered as a false positive.
* @param[in] progress_cb Optionally, a callback that is called after each run of the detector against a sample.
*                        The first parameter to the callback will be the number of samples processed and the second
*                        parameter will be the total number of samples to be processed. For example, the argument list
*                        (5, 10) means that the optimization is half way done.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_RES_INVALID_HANDLE`
*                   - `ARTOS_LEARN_RES_MODEL_NOT_LEARNED`
*                   - `ARTOS_IMGREPO_RES_INVALID_REPOSITORY`
*/
int learner_optimize_th(const unsigned int learner, const unsigned int max_positive = 0, const unsigned int num_negative = 0,
                        progress_cb_t progress_cb = 0);

/**
* Writes to models learned before by a learner instance to a mixture file.
* @param[in] learner The handle of the learner instance obtained by create_learner().
* @param[in] modelfile Path to the file which the newly learned models will be written to
*                      (the file will be created if it does not exist).
* @param[in] add If set to true, the new models will be added as additional mixture components if the model file
*                does already exist, otherwise the model file will be overwritten with just the new mixture.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_RES_INVALID_HANDLE`
*                   - `ARTOS_LEARN_RES_MODEL_NOT_LEARNED`
*                   - `ARTOS_RES_FILE_ACCESS_DENIED`
*/
int learner_save(const unsigned int learner, const char * modelfile, const bool add = true);

/**
* Resets a model learner instance to it's initial state by forgetting all positive samples,
* learned models and thresholds.
* @param[in] learner The handle of the learner instance obtained by create_learner().
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_RES_INVALID_HANDLE`
*/
int learner_reset(const unsigned int learner);


//------------------
//     ImageNet
//------------------

typedef struct {
    char synsetId[32]; /**< ID of the synset, e. g. 'n02119789'. */
    char description[220]; /**< Words and phrases describing the synset, e. g. 'kit fox, Vulpes macrotis'. */
    float score; /**< Score rating the precision of this synset as result for a search query. */
} SynsetSearchResult;

/**
* Lists all synsets in the image repository at a given location.
* @param[in] repo_directory The path to the repository directory.
* @param[in] synset_buf A beforehand allocated buffer array of SynsetSearchResult structs, that will be filled up with the
*                       list of synsets. The `score` field of each struct will be zero, since no search is performed.
*                       The order of the synsets corresponds to their order in the synset list file.
*                       This parameter may be NULL.
* @param[in,out] synset_buf_size The number of allocated array slots of `synset_buf`.
*                                In turn, the number of actually stored records will be written to this pointer's location.
*                                If `synset_buf` is NULL, this will be set to the number of records otherwise stored, i. e.
*                                the number of synsets available.            
* @return Returns `ARTOS_RES_OK` on success or `ARTOS_IMGREPO_RES_INVALID_REPOSITORY` if the given directory doesn't point to
*         a valid image repository.
*/
int list_synsets(const char * repo_directory, SynsetSearchResult * synset_buf, unsigned int * synset_buf_size);

/**
* Searches for synsets matching a given search phrase in the image repository at a given location.
* @param[in] repo_directory The path to the repository directory.
* @param[in] phrase The search phrase: a string of space-separated keywords.
* @param[in] result_buf A beforehand allocated buffer array of SynsetSearchResult structs, that will be filled up with the
*                       search results ordered descending by a score which specifies how well this synset's description
*                       matches the search phrase (higher is better).
* @param[in,out] result_buf_size The number of allocated array slots of `synset_buf`.
*                                In turn, the number of actually stored results will be written to this pointer's location.
* @return Returns `ARTOS_RES_OK` on success or `ARTOS_IMGREPO_RES_INVALID_REPOSITORY` if the given directory doesn't point to
*         a valid image repository.
*/
int search_synsets(const char * repo_directory, const char * phrase, SynsetSearchResult * result_buf, unsigned int * result_buf_size);

/**
* Extracts images from a given synset in a given image repository to a given directory.
* @param[in] repo_directory The path to the repository directory.
* @param[in] synset_id The ID of the synset.
* @param[in] out_directory The path to the directory where the extracted images are to be stored.
* @param[in,out] num_images Pointer to the maximum number of images to extract from the synset.
*                           In turn, the number of actually extracted images will be written to this pointer's location.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_IMGREPO_RES_INVALID_REPOSITORY`
*                   - `ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND`
*                   - `ARTOS_RES_DIRECTORY_NOT_FOUND` (if `out_directory` does not exist)
*/
int extract_images_from_synset(const char * repo_directory, const char * synset_id, const char * out_directory, unsigned int * num_images);

/**
* Extracts samples from images in a given synset of a given image repository, cropped to the actual object on that image
* using bounding box annotations. Thus, only samples from images which bounding box annotations are available for will
* be extracted to the given directory.
* @param[in] repo_directory The path to the repository directory.
* @param[in] synset_id The ID of the synset.
* @param[in] out_directory The path to the directory where the extracted samples are to be stored.
* @param[in,out] num_samples Pointer to the maximum number of samples to extract from the synset.
*                            In turn, the number of actually extracted samples will be written to this pointer's location.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_IMGREPO_RES_INVALID_REPOSITORY`
*                   - `ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND`
*                   - `ARTOS_RES_DIRECTORY_NOT_FOUND` (if `out_directory` does not exist)
*/
int extract_samples_from_synset(const char * repo_directory, const char * synset_id, const char * out_directory, unsigned int * num_samples);

/**
* Extracts images from various synsets of a given image repository to a given directory.
*
* The number of images taken from each synset can be specified. After that number has been extracted
* from the first synset, the next set of images will be taken from the second synset and so on. When the last
* synset has been processed, the next bunch of images will be taken from the first.
*
* @param[in] repo_directory The path to the repository directory.
* @param[in] out_directory The path to the directory where the extracted images are to be stored.
* @param[in] num_images The maximum number of images to extract.
* @param[in] per_synset The number of images taken from each synset in a row.
* @return Returns `ARTOS_RES_OK` on success or one of the following error codes on failure:
*                   - `ARTOS_IMGREPO_RES_INVALID_REPOSITORY`
*                   - `ARTOS_RES_DIRECTORY_NOT_FOUND` (if `out_directory` does not exist)
*/
int extract_mixed_images(const char * repo_directory, const char * out_directory, const unsigned int num_images, const unsigned int per_synset = 1);


#ifdef __cplusplus
}
#endif

#endif
