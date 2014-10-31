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

# libartos result codes
RES_OK = 0
RES_INVALID_HANDLE = -1
RES_DIRECTORY_NOT_FOUND = -2
RES_FILE_NOT_FOUND = -3
RES_FILE_ACCESS_DENIED = -4
RES_ABORTED = -5
RES_INTERNAL_ERROR = -999
DETECT_RES_INVALID_IMG_DATA = -101
DETECT_RES_INVALID_MODEL_FILE = -102
DETECT_RES_INVALID_MODEL_LIST_FILE = -103
DETECT_RES_NO_MODELS = -104
DETECT_RES_INVALID_IMAGE = -105
LEARN_RES_FAILED = -201
LEARN_RES_INVALID_BG_FILE = -202
LEARN_RES_INVALID_IMG_DATA = -203
LEARN_RES_NO_SAMPLES = -204
LEARN_RES_MODEL_NOT_LEARNED = -205
IMGREPO_RES_INVALID_REPOSITORY = -301
IMGREPO_RES_SYNSET_NOT_FOUND = -302
IMGREPO_RES_EXTRACTION_FAILED = -303



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


# SynsetSearchResult structure definition according to libartos.h
class SynsetSearchResult(Structure):
    _fields_ = [('synsetId', c_char * 32),
                ('description', c_char * 220),
                ('score', c_float)]



# Callback types
progress_cb_t = CFUNCTYPE(c_bool, c_uint, c_uint)
overall_progress_cb_t = CFUNCTYPE(c_bool, c_uint, c_uint, c_uint, c_uint)



# Pointer types
c_ubyte_p = POINTER(c_ubyte)
c_uint_p = POINTER(c_uint)
c_float_p = POINTER(c_float)
FlatDetection_p = POINTER(FlatDetection)
FlatBoundingBox_p = POINTER(FlatBoundingBox)
SynsetSearchResult_p = POINTER(SynsetSearchResult)



class _LibARTOS(object):

    def __init__(self, library):
        """Sets up the function prototypes of the library."""
        
        object.__init__(self)
        self._lib = library
        
        # create_detector function
        prototype = CFUNCTYPE(c_uint, c_double, c_int, c_int, c_bool)
        paramflags = (1, 'overlap', 0.5), (1, 'padding', 12), (1, 'interval', 10), (1, 'debug', False)
        self.create_detector = prototype(('create_detector', self._lib), paramflags)
        self.create_detector.errcheck = self._errcheck_create_detector
        
        # destroy_detector function
        prototype = CFUNCTYPE(c_void_p, c_uint)
        paramflags = (1, 'detector'),
        self.destroy_detector = prototype(('destroy_detector', self._lib), paramflags)
        
        # add_model function
        prototype = CFUNCTYPE(c_int, c_uint, c_char_p, c_char_p, c_double, c_char_p)
        paramflags = (1, 'detector'), (1, 'classname'), (1, 'modelfile'), (1, 'threshold'), (1, 'synset_id', None)
        self.add_model = prototype(('add_model', self._lib), paramflags)
        self.add_model.errcheck = self._errcheck_common
        
        # add_models function
        prototype = CFUNCTYPE(c_int, c_uint, c_char_p)
        paramflags = (1, 'detector'), (1, 'modellistfile')
        self.add_models = prototype(('add_models', self._lib), paramflags)
        self.add_models.errcheck = self._errcheck_common
        
        # detect_file_jpeg function
        prototype = CFUNCTYPE(c_int, c_uint, c_char_p, FlatDetection_p, c_uint_p)
        paramflags = (1, 'detector'), (1, 'imagefile'), (1, 'detection_buf'), (1, 'detection_buf_size')
        self.detect_file_jpeg = prototype(('detect_file_jpeg', self._lib), paramflags)
        self.detect_file_jpeg.errcheck = self._errcheck_common
        
        # detect_raw function
        prototype = CFUNCTYPE(c_int, c_uint, c_ubyte_p, c_uint, c_uint, c_bool, FlatDetection_p, c_uint_p)
        paramflags = (1, 'detector'), (1, 'img_data'), (1, 'img_width'), (1, 'img_height'), (1, 'grayscale'), (1, 'detection_buf'), (1, 'detection_buf_size')
        self.detect_raw = prototype(('detect_raw', self._lib), paramflags)
        self.detect_raw.errcheck = self._errcheck_common
        
        # learn_imagenet function
        prototype = CFUNCTYPE(c_int, c_char_p, c_char_p, c_char_p, c_char_p, c_bool, c_uint, c_uint, c_uint, c_uint, c_uint, overall_progress_cb_t, c_bool)
        paramflags = (1, 'repo_directory'), (1, 'synset_id'), (1, 'bg_file'), (1, 'modelfile'), \
                     (1, 'add', True), (1, 'max_aspect_clusters', 2), (1, 'max_who_clusters', 3), \
                     (1, 'th_opt_num_positive', 0), (1, 'th_opt_num_negative', 0), (1, 'th_opt_mode', THOPT_LOOCV), \
                     (1, 'progress_cb', None), (1, 'debug', False)
        self.learn_imagenet = prototype(('learn_imagenet', self._lib), paramflags)
        self.learn_imagenet.errcheck = self._errcheck_common
        
        # learn_files_jpeg function
        prototype = CFUNCTYPE(c_int, POINTER(c_char_p), c_uint, FlatBoundingBox_p, c_char_p, c_char_p, c_bool, c_uint, c_uint, c_uint, \
                              overall_progress_cb_t, c_bool)
        paramflags = (1, 'imagefiles'), (1, 'num_imagefiles'), (1, 'bounding_boxes'), (1, 'bg_file'), (1, 'modelfile'), \
                     (1, 'add', True), (1, 'max_aspect_clusters', 2), (1, 'max_who_clusters', 3), (1, 'th_opt_mode', THOPT_LOOCV), \
                     (1, 'progress_cb', None), (1, 'debug', False)
        self.learn_files_jpeg = prototype(('learn_files_jpeg', self._lib), paramflags)
        self.learn_files_jpeg.errcheck = self._errcheck_common
        
        # create_learner function
        prototype = CFUNCTYPE(c_uint, c_char_p, c_char_p, c_bool, c_bool)
        paramflags = (1, 'bg_file'), (1, 'repo_directory', ''), (1, 'th_opt_loocv', True), (1, 'debug', False)
        self.create_learner = prototype(('create_learner', self._lib), paramflags)
        self.create_learner.errcheck = self._errcheck_create_learner
        
        # destroy_learner function
        prototype = CFUNCTYPE(c_void_p, c_uint)
        paramflags = (1, 'learner'),
        self.destroy_learner = prototype(('destroy_learner', self._lib), paramflags)
        
        # learner_add_synset function
        prototype = CFUNCTYPE(c_int, c_uint, c_char_p, c_uint)
        paramflags = (1, 'learner'), (1, 'synset_id'), (1, 'max_samples', 0)
        self.learner_add_synset = prototype(('learner_add_synset', self._lib), paramflags)
        self.learner_add_synset.errcheck = self._errcheck_common
        
        # learner_add_file_jpeg function
        prototype = CFUNCTYPE(c_int, c_uint, c_char_p, FlatBoundingBox_p, c_uint)
        paramflags = (1, 'learner'), (1, 'imagefile'), (1, 'bboxes', None), (1, 'num_bboxes', 1)
        self.learner_add_file_jpeg = prototype(('learner_add_file_jpeg', self._lib), paramflags)
        self.learner_add_file_jpeg.errcheck = self._errcheck_common
        
        # learner_add_raw function
        prototype = CFUNCTYPE(c_int, c_uint, c_ubyte_p, c_uint, c_uint, c_bool, FlatBoundingBox_p, c_uint)
        paramflags = (1, 'learner'), (1, 'img_data'), (1, 'img_width'), (1, 'img_height'), (1, 'grayscale', False), \
                     (1, 'bboxes', None), (1, 'num_bboxes', 1)
        self.learner_add_raw = prototype(('learner_add_raw', self._lib), paramflags)
        self.learner_add_raw.errcheck = self._errcheck_common
        
        # learner_run function
        prototype = CFUNCTYPE(c_int, c_uint, c_uint, c_uint, progress_cb_t)
        paramflags = (1, 'learner'), (1, 'max_aspect_clusters', 2), (1, 'max_who_clusters', 3), (1, 'progress_cb', None)
        self.learner_run = prototype(('learner_run', self._lib), paramflags)
        self.learner_run.errcheck = self._errcheck_common
        
        # learner_optimize_th function
        prototype = CFUNCTYPE(c_int, c_uint, c_uint, c_uint, progress_cb_t)
        paramflags = (1, 'learner'), (1, 'max_positive', 0), (1, 'num_negative', 0), (1, 'progress_cb', None)
        self.learner_optimize_th = prototype(('learner_optimize_th', self._lib), paramflags)
        self.learner_optimize_th.errcheck = self._errcheck_common
        
        # learner_save function
        prototype = CFUNCTYPE(c_int, c_uint, c_char_p, c_bool)
        paramflags = (1, 'learner'), (1, 'modelfile'), (1, 'add', True)
        self.learner_save = prototype(('learner_save', self._lib), paramflags)
        self.learner_save.errcheck = self._errcheck_common
        
        # learner_reset function
        prototype = CFUNCTYPE(c_int, c_uint)
        paramflags = ((1, 'learner'), )
        self.learner_reset = prototype(('learner_reset', self._lib), paramflags)
        self.learner_reset.errcheck = self._errcheck_common
        
        # learn_bg function
        prototype = CFUNCTYPE(c_int, c_char_p, c_char_p, c_uint, c_uint, overall_progress_cb_t, c_bool)
        paramflags = (1, 'repo_directory'), (1, 'bg_file'), (1, 'num_images'), (1, 'max_offset', 19), (1, 'progress_cb', None), (1, 'accurate_autocorrelation', False)
        self.learn_bg = prototype(('learn_bg', self._lib), paramflags)
        self.learn_bg.errcheck = self._errcheck_common
        
        # list_synsets function
        prototype = CFUNCTYPE(c_int, c_char_p, SynsetSearchResult_p, c_uint_p)
        paramflags = (1, 'repo_directory'), (1, 'synset_buf'), (1, 'synset_buf_size')
        self.list_synsets = prototype(('list_synsets', self._lib), paramflags)
        self.list_synsets.errcheck = self._errcheck_common
        
        # search_synsets function
        prototype = CFUNCTYPE(c_int, c_char_p, c_char_p, SynsetSearchResult_p, c_uint_p)
        paramflags = (1, 'repo_directory'), (1, 'phrase'), (1, 'result_buf'), (1, 'result_buf_size')
        self.search_synsets = prototype(('search_synsets', self._lib), paramflags)
        self.search_synsets.errcheck = self._errcheck_common
        
        # extract_images_from_synset and extract_samples_from_synset function
        prototype = CFUNCTYPE(c_int, c_char_p, c_char_p, c_char_p, c_uint_p)
        paramflags = (1, 'repo_directory'), (1, 'synset_id'), (1, 'out_directory'), (1, 'num_images')
        self.extract_images_from_synset = prototype(('extract_images_from_synset', self._lib), paramflags)
        self.extract_images_from_synset.errcheck = self._errcheck_common
        self.extract_samples_from_synset = prototype(('extract_samples_from_synset', self._lib), paramflags)
        self.extract_samples_from_synset.errcheck = self._errcheck_common
        
        # extract_mixed_images function
        prototype = CFUNCTYPE(c_int, c_char_p, c_char_p, c_uint, c_uint)
        paramflags = (1, 'repo_directory'), (1, 'out_directory'), (1, 'num_images'), (1, 'per_synset', 1)
        self.extract_mixed_images = prototype(('extract_mixed_images', self._lib), paramflags)
        self.extract_mixed_images.errcheck = self._errcheck_common


    @staticmethod
    def _errcheck_create_detector(result, func, args):
        if (result == 0):
            raise MemoryError('[libartos] Not enough memory to create new detector')
        return args
    
    
    @staticmethod
    def _errcheck_create_learner(result, func, args):
        if (result == 0):
            raise LibARTOSException(LEARN_RES_INVALID_BG_FILE);
        return args


    @staticmethod
    def _errcheck_common(result, func, args):
        if (result < 0):
            raise LibARTOSException(result)
        return args



class LibARTOSException(Exception):

    errmsgs = {
        RES_INVALID_HANDLE                  : 'Invalid handle given',
        RES_DIRECTORY_NOT_FOUND             : 'Could not find the given directory',
        RES_FILE_NOT_FOUND                  : 'File not found',
        RES_FILE_ACCESS_DENIED              : 'Access to file denied',
        RES_ABORTED                         : 'Operation aborted by user',
        RES_INTERNAL_ERROR                  : 'Internal error',
        DETECT_RES_INVALID_IMG_DATA         : 'Invalid image',
        DETECT_RES_INVALID_MODEL_FILE       : 'Model file could not be read or parsed',
        DETECT_RES_INVALID_MODEL_LIST_FILE  : 'Model list file could not be read or parsed',
        DETECT_RES_NO_MODELS                : 'Models must be added to the detector before detecting',
        DETECT_RES_INVALID_IMAGE            : 'Image could not be processed',
        LEARN_RES_FAILED                    : 'Learning the model failed for some reason',
        LEARN_RES_INVALID_BG_FILE           : 'Given background statistics file is invalid',
        LEARN_RES_INVALID_IMG_DATA          : 'Invalid image',
        LEARN_RES_NO_SAMPLES                : 'No positive sample has been added yet',
        LEARN_RES_MODEL_NOT_LEARNED         : 'No model has been learned yet',
        IMGREPO_RES_INVALID_REPOSITORY      : 'Given path doesn\'t point to a valid image repository',
        IMGREPO_RES_SYNSET_NOT_FOUND        : 'Synset not found',
        IMGREPO_RES_EXTRACTION_FAILED       : 'Could not extract images from synset. Please check your image repository and make sure, ' \
                                              'that both the image and the annotation archive are there.'
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
