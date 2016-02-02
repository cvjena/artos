"""Wrapper for the shared ARTOS (Adaptive Real-time Object detection System) library.

This module provides a low-level interface to the ARTOS library as an instance of a wrapper class
called _LibARTOS. That instance is stored in this module's dictionary's field 'libartos', which may
be None if the library could not be found or loaded.
The _LibARTOS class provides exactly the same functions as libartos.
It is used by detecting.Detector, learning.ModelLearner and imagenet.ImageRepository, which provide
an object-oriented high-level API to libartos.
"""

from ctypes import *
from ctypes import util
from .utils import basedir
from .config import config
import os.path


# libartos parameter constants
THOPT_NONE = 0
THOPT_OVERLAPPING = 1
THOPT_LOOCV = 2

PARAM_TYPE_INT = 0
PARAM_TYPE_SCALAR = 1
PARAM_TYPE_STRING = 2

# libartos result codes
RES_OK = 0
RES_INVALID_HANDLE = -1
RES_DIRECTORY_NOT_FOUND = -2
RES_FILE_NOT_FOUND = -3
RES_FILE_ACCESS_DENIED = -4
RES_ABORTED = -5
RES_INDEX_OUT_OF_BOUNDS = -6
RES_INTERNAL_ERROR = -999
DETECT_RES_INVALID_IMG_DATA = -101
DETECT_RES_INVALID_MODEL_FILE = -102
DETECT_RES_INVALID_MODEL_LIST_FILE = -103
DETECT_RES_NO_MODELS = -104
DETECT_RES_INVALID_IMAGE = -105
DETECT_RES_NO_IMAGES = -106
DETECT_RES_NO_RESULTS = -107
DETECT_RES_INVALID_ANNOTATIONS = -108
DETECT_RES_TOO_MANY_MODELS = -109
LEARN_RES_FAILED = -201
LEARN_RES_INVALID_BG_FILE = -202
LEARN_RES_INVALID_IMG_DATA = -203
LEARN_RES_NO_SAMPLES = -204
LEARN_RES_MODEL_NOT_LEARNED = -205
LEARN_RES_FEATURE_EXTRACTOR_NOT_READY = -206
IMGREPO_RES_INVALID_REPOSITORY = -301
IMGREPO_RES_SYNSET_NOT_FOUND = -302
IMGREPO_RES_EXTRACTION_FAILED = -303
SETTINGS_RES_UNKNOWN_FEATURE_EXTRACTOR = -401
SETTINGS_RES_UNKNOWN_PARAMETER = -402
SETTINGS_RES_INVALID_PARAMETER_VALUE = -403



# FlatDetection structure definition according to libartos.h
class FlatDetection(Structure):
    _fields_ = [('classname', c_char * 44),
                ('synset_id', c_char * 16),
                ('score', c_float),
                ('left', c_int),
                ('top', c_int),
                ('right', c_int),
                ('bottom', c_int)]


# FlatBoundingBox structure definition according to libartos.h
class FlatBoundingBox(Structure):
    _fields_ = [('left', c_uint),
                ('top', c_uint),
                ('width', c_uint),
                ('height', c_uint)]


# RawTestResult structure definition according to libartos.h
class RawTestResult(Structure):
    _fields_ = [('threshold', c_double),
                ('tp', c_uint),
                ('fp', c_uint),
                ('np', c_uint)]


# SynsetSearchResult structure definition according to libartos.h
class SynsetSearchResult(Structure):
    _fields_ = [('synsetId', c_char * 32),
                ('description', c_char * 220),
                ('score', c_float)]


# FeatureExtractorInfo structure definition according to libartos.h
class FeatureExtractorInfo(Structure):
    _fields_ = [('type', c_char * 28),
                ('name', c_char * 100)]


# FeatureExtractorParameterValue union definition according to libartos.h
class FeatureExtractorParameterValue(Union):
    _fields_ = [('intVal', c_int),
                ('scalarVal', c_float),
                ('stringVal', c_char_p)]


# FeatureExtractorParameter structure definition according to libartos.h
class FeatureExtractorParameter(Structure):
    _fields_ = [('name', c_char * 52),
                ('type', c_uint),
                ('val', FeatureExtractorParameterValue)]



# Callback types
progress_cb_t = CFUNCTYPE(c_bool, c_uint, c_uint)
overall_progress_cb_t = CFUNCTYPE(c_bool, c_uint, c_uint, c_uint, c_uint)



# Pointer types
c_ubyte_p = POINTER(c_ubyte)
c_uint_p = POINTER(c_uint)
c_float_p = POINTER(c_float)
FlatDetection_p = POINTER(FlatDetection)
FlatBoundingBox_p = POINTER(FlatBoundingBox)
RawTestResult_p = POINTER(RawTestResult)
SynsetSearchResult_p = POINTER(SynsetSearchResult)
FeatureExtractorInfo_p = POINTER(FeatureExtractorInfo)
FeatureExtractorParameterValue_p = POINTER(FeatureExtractorParameterValue)
FeatureExtractorParameter_p = POINTER(FeatureExtractorParameter)



class _LibARTOS(object):

    def __init__(self, library):
        """Sets up the function prototypes of the library."""
        
        object.__init__(self)
        self._lib = library
        
        # create_detector function
        self._register_func('create_detector',
            (c_uint, c_double, c_int, c_bool),
            ((1, 'overlap', 0.5), (1, 'interval', 10), (1, 'debug', False)),
            self._errcheck_create_detector
        )
        
        # destroy_detector function
        self._register_func('destroy_detector',
            (c_void_p, c_uint),
            ((1, 'detector'),)
        )
        
        # add_model function
        self._register_func('add_model',
            (c_int, c_uint, c_char_p, c_char_p, c_double, c_char_p),
            ((1, 'detector'), (1, 'classname'), (1, 'modelfile'), (1, 'threshold'), (1, 'synset_id', None))
        )
        
        # add_models function
        self._register_func('add_models',
            (c_int, c_uint, c_char_p),
            ((1, 'detector'), (1, 'modellistfile'))
        )
        
        # num_feature_extractors_in_detector function
        self._register_func('num_feature_extractors_in_detector',
            (c_int, c_uint),
            ((1, 'detector'), ),
            self._errcheck_num_fe
        )
        
        # detect_file_jpeg function
        self._register_func('detect_file_jpeg',
            (c_int, c_uint, c_char_p, FlatDetection_p, c_uint_p),
            ((1, 'detector'), (1, 'imagefile'), (1, 'detection_buf'), (1, 'detection_buf_size'))
        )
        
        # detect_raw function
        self._register_func('detect_raw',
            (c_int, c_uint, c_ubyte_p, c_uint, c_uint, c_bool, FlatDetection_p, c_uint_p),
            ((1, 'detector'), (1, 'img_data'), (1, 'img_width'), (1, 'img_height'), (1, 'grayscale'),
             (1, 'detection_buf'), (1, 'detection_buf_size'))
        )
        
        # learn_imagenet function
        self._register_func('learn_imagenet',
            (c_int, c_char_p, c_char_p, c_char_p, c_char_p, c_bool, c_uint, c_uint, c_uint, c_uint, c_uint, overall_progress_cb_t, c_bool),
            ((1, 'repo_directory'), (1, 'synset_id'), (1, 'bg_file'), (1, 'modelfile'),
             (1, 'add', True), (1, 'max_aspect_clusters', 2), (1, 'max_who_clusters', 3),
             (1, 'th_opt_num_positive', 0), (1, 'th_opt_num_negative', 0), (1, 'th_opt_mode', THOPT_LOOCV),
             (1, 'progress_cb', cast(None, progress_cb_t)), (1, 'debug', False))
        )
        
        # learn_files_jpeg function
        self._register_func('learn_files_jpeg',
            (c_int, POINTER(c_char_p), c_uint, FlatBoundingBox_p, c_char_p, c_char_p, c_bool, c_uint, c_uint, c_uint, overall_progress_cb_t, c_bool),
            ((1, 'imagefiles'), (1, 'num_imagefiles'), (1, 'bounding_boxes'), (1, 'bg_file'), (1, 'modelfile'),
             (1, 'add', True), (1, 'max_aspect_clusters', 2), (1, 'max_who_clusters', 3), (1, 'th_opt_mode', THOPT_LOOCV),
             (1, 'progress_cb', cast(None, progress_cb_t)), (1, 'debug', False))
        )
        
        # create_learner function
        self._register_func('create_learner',
            (c_uint, c_char_p, c_char_p, c_bool, c_bool),
            ((1, 'bg_file'), (1, 'repo_directory', ''), (1, 'th_opt_loocv', True), (1, 'debug', False)),
            self._errcheck_create_learner
        )
        
        # destroy_learner function
        self._register_func('destroy_learner',
            (c_void_p, c_uint),
            ((1, 'learner'),)
        )
        
        # learner_add_synset function
        self._register_func('learner_add_synset',
            (c_int, c_uint, c_char_p, c_uint),
            ((1, 'learner'), (1, 'synset_id'), (1, 'max_samples', 0))
        )
        
        # learner_add_file_jpeg function
        self._register_func('learner_add_file_jpeg',
            (c_int, c_uint, c_char_p, FlatBoundingBox_p, c_uint),
            ((1, 'learner'), (1, 'imagefile'), (1, 'bboxes', None), (1, 'num_bboxes', 1))
        )
        
        # learner_add_raw function
        self._register_func('learner_add_raw',
            (c_int, c_uint, c_ubyte_p, c_uint, c_uint, c_bool, FlatBoundingBox_p, c_uint),
            ((1, 'learner'), (1, 'img_data'), (1, 'img_width'), (1, 'img_height'), (1, 'grayscale', False),
             (1, 'bboxes', None), (1, 'num_bboxes', 1))
        )
        
        # learner_run function
        self._register_func('learner_run',
            (c_int, c_uint, c_uint, c_uint, progress_cb_t),
            ((1, 'learner'), (1, 'max_aspect_clusters', 2), (1, 'max_who_clusters', 3), (1, 'progress_cb', cast(None, progress_cb_t)))
        )
        
        # learner_optimize_th function
        self._register_func('learner_optimize_th',
            (c_int, c_uint, c_uint, c_uint, progress_cb_t),
            ((1, 'learner'), (1, 'max_positive', 0), (1, 'num_negative', 0), (1, 'progress_cb', cast(None, progress_cb_t)))
        )
        
        # learner_save function
        self._register_func('learner_save',
            (c_int, c_uint, c_char_p, c_bool),
            ((1, 'learner'), (1, 'modelfile'), (1, 'add', True))
        )
        
        # learner_reset function
        self._register_func('learner_reset',
            (c_int, c_uint),
            ((1, 'learner'), )
        )
        
        # learn_bg function
        self._register_func('learn_bg',
            (c_int, c_char_p, c_char_p, c_uint, c_uint, overall_progress_cb_t, c_bool),
            ((1, 'repo_directory'), (1, 'bg_file'), (1, 'num_images'), (1, 'max_offset', 19),
             (1, 'progress_cb', cast(None, progress_cb_t)), (1, 'accurate_autocorrelation', False))
        )
        
        # evaluator_add_samples_from_synset function
        self._register_func('evaluator_add_samples_from_synset',
            (c_int, c_uint, c_char_p, c_char_p, c_uint),
            ((1, 'detector'), (1, 'repo_directory'), (1, 'synset_id'), (1, 'num_negative', 0))
        )
        
        # evaluator_add_positive_file function
        self._register_func('evaluator_add_positive_file',
            (c_int, c_uint, c_char_p, c_char_p),
            ((1, 'detector'), (1, 'imagefile'), (1, 'annotation_file'))
        )
        
        # evaluator_add_positive_file_jpeg function
        self._register_func('evaluator_add_positive_file_jpeg',
            (c_int, c_uint, c_char_p, FlatBoundingBox_p, c_uint),
            ((1, 'detector'), (1, 'imagefile'), (1, 'bboxes', None), (1, 'num_bboxes', 1))
        )
        
        # evaluator_add_positive_raw function
        self._register_func('evaluator_add_positive_raw',
            (c_int, c_uint, c_ubyte_p, c_uint, c_uint, c_bool, FlatBoundingBox_p, c_uint),
            ((1, 'detector'), (1, 'img_data'), (1, 'img_width'), (1, 'img_height'), (1, 'grayscale', False),
             (1, 'bboxes', None), (1, 'num_bboxes', 1))
        )
        
        # evaluator_add_negative_file_jpeg function
        self._register_func('evaluator_add_negative_file_jpeg',
            (c_int, c_uint, c_char_p),
            ((1, 'detector'), (1, 'imagefile'))
        )
        
        # evaluator_add_negative_raw function
        self._register_func('evaluator_add_negative_raw',
            (c_int, c_uint, c_ubyte_p, c_uint, c_uint, c_bool),
            ((1, 'detector'), (1, 'img_data'), (1, 'img_width'), (1, 'img_height'), (1, 'grayscale', False))
        )
        
        # evaluator_run function
        self._register_func('evaluator_run',
            (c_int, c_uint, c_uint, progress_cb_t),
            ((1, 'detector'), (1, 'granularity', 100), (1, 'progress_cb', cast(None, progress_cb_t)))
        )
        
        # evaluator_get_raw_results function
        self._register_func('evaluator_get_raw_results',
            (c_int, c_uint, RawTestResult_p, c_uint_p, c_uint),
            ((1, 'detector'), (1, 'result_buf'), (1, 'result_buf_size'), (1, 'model_index', 0))
        )
        
        # evaluator_get_max_fmeasure function
        self._register_func('evaluator_get_max_fmeasure',
            (c_int, c_uint, c_float_p, c_float_p, c_uint),
            ((1, 'detector'), (1, 'fmeasure'), (1, 'threshold', None), (1, 'model_index', 0))
        )
        
        # evaluator_get_fmeasure_at function
        self._register_func('evaluator_get_fmeasure_at',
            (c_int, c_uint, c_float, c_float_p, c_uint),
            ((1, 'detector'), (1, 'threshold'), (1, 'fmeasure'), (1, 'model_index', 0))
        )
        
        # evaluator_get_ap function
        self._register_func('evaluator_get_ap',
            (c_int, c_uint, c_float_p, c_uint),
            ((1, 'detector'), (1, 'ap'), (1, 'model_index', 0))
        )
        
        # evaluator_dump_results function
        self._register_func('evaluator_dump_results',
            (c_int, c_uint, c_char_p),
            ((1, 'detector'), (1, 'dump_file'))
        )
        
        # change_feature_extractor function
        self._register_func('change_feature_extractor',
            (c_int, c_char_p),
            ((1, 'type'), )
        )
        
        # feature_extractor_get_info function
        self._register_func('feature_extractor_get_info',
            (c_int, FeatureExtractorInfo_p),
            ((1, 'info'), )
        )
        
        # list_feature_extractors function
        self._register_func('list_feature_extractors',
            (c_int, FeatureExtractorInfo_p, c_uint_p),
            ((1, 'info_buf'), (1, 'info_buf_size'))
        )
        
        # list_feature_extractor_params function
        self._register_func('list_feature_extractor_params',
            (c_int, c_char_p, FeatureExtractorParameter_p, c_uint_p),
            ((1, 'type'), (1, 'param_buf'), (1, 'param_buf_size'))
        )
        
        # feature_extractor_list_params function
        self._register_func('feature_extractor_list_params',
            (c_int, FeatureExtractorParameter_p, c_uint_p),
            ((1, 'param_buf'), (1, 'param_buf_size'))
        )
        
        # feature_extractor_set_int_param function
        self._register_func('feature_extractor_set_int_param',
            (c_int, c_char_p, c_int),
            ((1, 'param_name'), (1, 'value'))
        )
        
        # feature_extractor_set_scalar_param function
        self._register_func('feature_extractor_set_scalar_param',
            (c_int, c_char_p, c_float),
            ((1, 'param_name'), (1, 'value'))
        )
        
        # feature_extractor_set_string_param function
        self._register_func('feature_extractor_set_string_param',
            (c_int, c_char_p, c_char_p),
            ((1, 'param_name'), (1, 'value'))
        )
        
        # list_synsets function
        self._register_func('list_synsets',
            (c_int, c_char_p, SynsetSearchResult_p, c_uint_p),
            ((1, 'repo_directory'), (1, 'synset_buf'), (1, 'synset_buf_size'))
        )
        
        # search_synsets function
        self._register_func('search_synsets',
            (c_int, c_char_p, c_char_p, SynsetSearchResult_p, c_uint_p),
            ((1, 'repo_directory'), (1, 'phrase'), (1, 'result_buf'), (1, 'result_buf_size'))
        )
        
        # extract_images_from_synset and extract_samples_from_synset function
        prototype = (c_int, c_char_p, c_char_p, c_char_p, c_uint_p)
        paramflags = (1, 'repo_directory'), (1, 'synset_id'), (1, 'out_directory'), (1, 'num_images')
        self._register_func('extract_images_from_synset', prototype, paramflags)
        self._register_func('extract_samples_from_synset', prototype, paramflags)
        
        # extract_mixed_images function
        self._register_func('extract_mixed_images',
            (c_int, c_char_p, c_char_p, c_uint, c_uint),
            ((1, 'repo_directory'), (1, 'out_directory'), (1, 'num_images'), (1, 'per_synset', 1))
        )
    
    
    def _register_func(self, funcName, paramtypes, paramflags, errcheck = None):
        
        prototype = CFUNCTYPE(*paramtypes)
        self.__dict__[funcName] = prototype((funcName, self._lib), paramflags)
        if (paramtypes[0] != c_void_p) or (errcheck is not None):
            self.__dict__[funcName].errcheck = errcheck if errcheck is not None else self._errcheck_common


    @staticmethod
    def _errcheck_create_detector(result, func, args):
        if (result == 0):
            raise MemoryError('[libartos] Not enough memory to create new detector')
        return args
    
    
    @staticmethod
    def _errcheck_create_learner(result, func, args):
        if (result == 0):
            raise LibARTOSException(LEARN_RES_INVALID_BG_FILE)
        return args
    
    
    @staticmethod
    def _errcheck_num_fe(result, func, args):
        if (result < 0):
            raise LibARTOSException(RES_INVALID_HANDLE)
        return args


    @staticmethod
    def _errcheck_common(result, func, args):
        if (result < 0):
            raise LibARTOSException(result)
        return args



class LibARTOSException(Exception):

    errmsgs = {
        RES_INVALID_HANDLE                      : 'Invalid handle given',
        RES_DIRECTORY_NOT_FOUND                 : 'Could not find the given directory',
        RES_FILE_NOT_FOUND                      : 'File not found',
        RES_FILE_ACCESS_DENIED                  : 'Access to file denied',
        RES_ABORTED                             : 'Operation aborted by user',
        RES_INDEX_OUT_OF_BOUNDS                 : 'Index out of bounds',
        RES_INTERNAL_ERROR                      : 'Internal error',
        DETECT_RES_INVALID_IMG_DATA             : 'Invalid image',
        DETECT_RES_INVALID_MODEL_FILE           : 'Model file could not be read or parsed',
        DETECT_RES_INVALID_MODEL_LIST_FILE      : 'Model list file could not be read or parsed',
        DETECT_RES_NO_MODELS                    : 'Models must be added to the detector before detecting',
        DETECT_RES_INVALID_IMAGE                : 'Image could not be processed',
        DETECT_RES_NO_IMAGES                    : 'No test samples have been added to the detector',
        DETECT_RES_NO_RESULTS                   : 'The detector has not been run yet',
        DETECT_RES_INVALID_ANNOTATIONS          : 'Invalid annotation file',
        DETECT_RES_TOO_MANY_MODELS              : 'Evaluating multiple models at once is not supported',
        LEARN_RES_FAILED                        : 'Learning the model failed for some reason',
        LEARN_RES_INVALID_BG_FILE               : 'Given background statistics file is invalid',
        LEARN_RES_INVALID_IMG_DATA              : 'Invalid image',
        LEARN_RES_NO_SAMPLES                    : 'No positive sample has been added yet',
        LEARN_RES_MODEL_NOT_LEARNED             : 'No model has been learned yet',
        LEARN_RES_FEATURE_EXTRACTOR_NOT_READY   : 'The feature extractor has not been set up properly',
        IMGREPO_RES_INVALID_REPOSITORY          : 'Given path doesn\'t point to a valid image repository',
        IMGREPO_RES_SYNSET_NOT_FOUND            : 'Synset not found',
        IMGREPO_RES_EXTRACTION_FAILED           : 'Could not extract images from synset. Please check your image repository and make sure, ' \
                                                  'that both the image and the annotation archive are there.',
        SETTINGS_RES_UNKNOWN_FEATURE_EXTRACTOR  : 'Unknown feature extractor',
        SETTINGS_RES_UNKNOWN_PARAMETER          : 'Parameter is not known by the current feature extractor',
        SETTINGS_RES_INVALID_PARAMETER_VALUE    : 'Invalid value for feature extractor parameter given'
    }


    def __init__(self, errcode):
        Exception.__init__(self)
        self.errcode = errcode
        self.errmsg = self.__class__.errmsgs[errcode]
    
    
    def __str__(self):
        return self.errmsg


    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.errcode)



def _search_libartos():
    search_names = (config.get('libartos', 'library_path'), 'artos', 'libartos', 'libartos.so', os.path.join('.', 'libartos.so'), \
                    os.path.join('bin', 'artos'), os.path.join('bin', 'libartos'), os.path.join('bin', 'libartos.so'), \
                    os.path.join(basedir, 'artos'), os.path.join(basedir, 'libartos'), os.path.join(basedir, 'libartos.so'), \
                    os.path.join(basedir, 'bin', 'artos'), os.path.join(basedir, 'bin', 'libartos'), os.path.join(basedir, 'bin', 'libartos.so'), \
                    util.find_library('artos'))
    for n in search_names:
        if not n is None:
            try:
                lib = CDLL(n)
                return _LibARTOS(lib)
            except (OSError, TypeError):
                pass
    return None


# Search library
libartos = _search_libartos()
