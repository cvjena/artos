"""Interface to the model learning part of LibARTOS.

Provides the ModelLearner class on the one hand, which can be used to learn WHO models based on image data
from ImageNet, from file or from PIL.Image.Image objects.
On the other hand, this module contains the ModelManager and Model classes, which can be used to list,
modify and visualize the learned models.
Methods for learning of background statistics and modifying the default feature extractor used or learning
can be found here as well.
"""


from . import artos_wrapper, imagenet, utils
from .artos_wrapper import libartos
from .detecting import BoundingBox

try:
    from PIL import Image, ImageDraw
except:
    import Image, ImageDraw
import math, re, os, ctypes



def learnBGStatistics(imageRepository, bgFile, numImages, maxOffset = 19, progressCallback = None):
    """Learns a negative mean and a stationary covariance matrix as autocorrelation function from ImageNet samples.
    
    Both are required for learning WHO models. This expensive computation has only to be done once and the default
    background statistics shipped with ARTOS should be sufficient for most purposes.
    
    imageRepository - Either the path to the image repository or an imagenet.ImageRepository instance.
    bgFile - Path to save the learned background statistics to.
    numImages - Number of ImageNet samples to extract features from for learning background statistics.
    maxOffset - Maximum available offset in x or y direction of the autocorrelation function to be learned.
                Determines the maximum size of the reconstructible covariance matrix, which will be `maxOffset + 1`.
    progressCallback - Optionally, a callback which is called between the steps of the learning process to populate the progress.  
                       The first parameter to the callback will be the number of steps performed in the entire process,
                       the second one will be the total number of steps. To date, the entire process is divided into just two steps:
                       Learning a negative mean and learning an autocorrelation function.  
                       The third and fourth parameters will be the number of processed images and the total number of images (equal
                       to `numImages`) of the current sub-procedure.
                       The callback may return False to abort the operation. To continue, it must return True.
    """
    
    if (libartos is None):
        raise RuntimeError('Can not find libartos')
    if isinstance(imageRepository, imagenet.ImageRepository):
        repoDirectory = imageRepository.repoDirectory
    else:
        repoDirectory = imageRepository
    cb = artos_wrapper.overall_progress_cb_t(progressCallback) if not progressCallback is None \
         else ctypes.cast(None, artos_wrapper.overall_progress_cb_t)
    libartos.learn_bg(utils.str2bytes(repoDirectory), utils.str2bytes(bgFile), \
                            numImages, maxOffset, cb)



class ModelLearner(object):
    """Handles learning of WHO (Whitened Histogram of Orientations) models."""
    
    
    THOPT_NONE = artos_wrapper.THOPT_NONE
    THOPT_OVERLAPPING = artos_wrapper.THOPT_OVERLAPPING
    THOPT_LOOCV = artos_wrapper.THOPT_LOOCV
    

    def __init__(self, bgFile, imageRepository = None, thOptLOOCV = True, debug = False):
        """Constructs and initializes a new model learner using given stationary background statistics and a given image repository.
        
        bgFile - Path to a file containing the stationary background statistics for whitening (negative mean and covariance).
        imageRepository - Optionally, a path to an image repository or an imagenet.ImageRepository instance.
                          Required if you want to use ImageNet functions with this learner.
        thOptLOOCV - If set to True, Leave-one-out-cross-validation will be performed for threshold optimization.
                     This will increase memory usage, since the WHO features of all samples have to be stored for this,
                     and will slow down threshold optimization as well as the model learning step if only one WHO cluster
                     is used. But it will guarantee that no model is tested against a sample it has been learned from
                     for threshold optimization.
        debug - If set to True, timing and debugging information will be printed to stdout.
        """
        
        object.__init__(self)
        if (libartos is None):
            raise RuntimeError('Can not find libartos')
        if isinstance(imageRepository, imagenet.ImageRepository):
            repoDirectory = imageRepository.repoDirectory
        elif imageRepository is None:
            repoDirectory = ""
        else:
            repoDirectory = imageRepository
        self.handle = libartos.create_learner(utils.str2bytes(bgFile), utils.str2bytes(repoDirectory), thOptLOOCV, debug)


    def __del__(self):
        try:
            libartos.destroy_learner(self.handle)
        except:
            pass


    def addPositiveSample(self, sample, boundingBoxes = ()):
        """Adds an image as positive sample to this learner.
        
        sample - The image to be added as sample, given either by as PIL.Image.Image instance or as path
                 to a JPEG file. In the latter case, the image will be read directly by the library.
        boundingBoxes - Sequence with BoundingBox instances or (left, top, right, bottom) tuples which
                        specify the bounding boxes around the objects on the given image. If the sequence
                        or one of the bounding boxes is empty, the entire image will be used.
        """
        
        if not (isinstance(sample, Image.Image) or utils.is_str(sample)):
            raise TypeError('{0}.addPositiveSample expects argument sample to be either PIL.Image.Image or string'.format(self.__class__.__name__))
        if isinstance(boundingBoxes, BoundingBox):
            boundingBoxes = (boundingBoxes,)
        
        # Convert bounding box sequence to FlatBoundingBox array
        if len(boundingBoxes) > 0:
            bboxes = (artos_wrapper.FlatBoundingBox * len(boundingBoxes))()
            for i, box in enumerate((BoundingBox(b) for b in boundingBoxes)):
                bboxes[i] = artos_wrapper.FlatBoundingBox(left = box.left, top = box.top, width = box.width, height = box.height);
            numBBoxes = len(bboxes)
        else:
            bboxes = None
            numBBoxes = 0
        
        # Add sample
        if isinstance(sample, Image.Image):
            # Convert image to plain RGB or grayscale
            if (sample.mode in ('1', 'L')):
                grayscale = True
                if (sample.mode != 'L'):
                    sample = sample.convert('L')
            else:
                grayscale = False
                if (sample.mode != 'RGB'):
                    sample = sample.convert('RGB')
            # Copy raw image data into a buffer
            numbytes = sample.size[0] * sample.size[1]
            if not grayscale:
                numbytes = numbytes * 3
            try:
                imgbytes = sample.tobytes()
            except:
                imgbytes = sample.tostring()
            imgdata = ctypes.create_string_buffer(numbytes)
            ctypes.memmove(imgdata, imgbytes, numbytes)
            # Add sample
            libartos.learner_add_raw(self.handle, ctypes.cast(imgdata, artos_wrapper.c_ubyte_p), \
                                     sample.size[0], sample.size[1], grayscale, bboxes, numBBoxes)
        else:
            # Treat sample as file name
            libartos.learner_add_file_jpeg(self.handle, utils.str2bytes(sample), bboxes, numBBoxes)


    def addPositiveSamplesFromSynset(self, synsetId, maxSamples = 0):
        """Adds positive samples from a synset of the associated image repository.
        
        The bounding box annotation data of the synset is used to extract the samples from the images.
        
        synsetId - The ID of the synset.
        maxSamples - Maximum number images to extract from the synset. Set this to 0, to extract
                     all images of the synset.
        """
    
        libartos.learner_add_synset(self.handle, utils.str2bytes(synsetId), maxSamples)


    def learn(self, maxAspectClusters = 2, maxWHOClusters = 3, progressCallback = None):
        """Performs the actual learning step.
        
        Optionally, clustering can be performed before learning, first by aspect ratio, then by WHO features. A separate model
        will be learned for each cluster. Thus, the maximum number of learned models will be maxAspectClusters * maxWHOClusters.
        
        Before calling this, some positive samples must have been added.
        
        The learned models will be provided with estimated thresholds. The estimate lacks an additive term involving a-priori probabilities
        and, thus, is not optimal. Hence, you'll probably want to call optimizeThreshold() afterwards.
        
        maxAspectClusters - Maximum number of clusters to form by the aspect ratio of the samples.
        maxWHOClusters - Maximum number of clusters to form by the WHO feature vectors of the samples of a single aspect ratio cluster.
        """
        
        cb = artos_wrapper.progress_cb_t(progressCallback) if not progressCallback is None \
             else ctypes.cast(None, artos_wrapper.progress_cb_t)
        libartos.learner_run(self.handle, maxAspectClusters, maxWHOClusters, cb)


    def optimizeThreshold(self, maxPositive = 0, numNegative = 0, progressCallback = None):
        """Tries to find the optimal thresholds for the models learned before by learn().
        
        The thresholds will be stored internally and will be used as a bias in the model file when save() is called.
        
        maxPositive - Maximum number of positive samples to test the models against. Set this to 0 to run the detector
                      against all samples.
        numNegative - Number of negative samples taken from different synsets if an image repository has been set.
                      Every detection on one of these images will be considered as a false positive.
        """
        
        cb = artos_wrapper.progress_cb_t(progressCallback) if not progressCallback is None \
             else ctypes.cast(None, artos_wrapper.progress_cb_t)
        return libartos.learner_optimize_th(self.handle, maxPositive, numNegative, cb)


    def save(self, modelfile, add = True):
        """Writes the model learned before to a mixture file.
        
        modelfile - Path to the file which the newly learned models will be written to
                    (the file will be created if it does not exist).
        add - If set to true, the new models will be added as additional mixture components if the model file does
              already exist, otherwise the model file will be overwritten with just the new mixture.
        """
        
        libartos.learner_save(self.handle, utils.str2bytes(modelfile), add)


    def reset(self):
        """Resets this learner to it's initial state by forgetting all learned models, thresholds and positive samples."""
        
        libartos.learner_reset(self.handle)


    @staticmethod
    def learnModelFromSynset(imageRepository, synsetId, bgFile, modelfile, add = True, \
                             maxAspectClusters = 2, maxWHOClusters = 3, \
                             thOptNumPositive = 0, thOptNumNegative = 0, thOptMode = THOPT_LOOCV, progressCallback = None, \
                             debug = False):
        """All-in-one short-cut method for learning a new WHO model from positive samples extracted from an ImageNet synset.
        
        imageRepository - Either the path to the image repository or an imagenet.ImageRepository instance.
        synsetId - The ID of the synset to extract positive samples from.
        bgFile - Path to a file containing the stationary background statistics for whitening (negative mean and covariance).
        modelfile - Path to the file which the newly learned models will be written to
                    (the file will be created if it does not exist).
        add - If set to true, the new models will be added as additional mixture components if the model file does
              already exist, otherwise the model file will be overwritten with just the new mixture.
        maxAspectClusters - Maximum number of clusters to form by the aspect ratio of the samples.
        maxWHOClusters - Maximum number of clusters to form by the WHO feature vectors of the samples of a single aspect ratio cluster.
        thOptNumPositive - Maximum number of positive samples to test the models against for finding the optimal threshold.
                           Set this to 0 to run the detector against all samples.
        thOptNumNegative - Number of negative samples taken from different synsets for finding the optimal threshold.
                           Every detection on one of these images will be considered as a false positive.
        thOptMode - Controls the mode of threshold optimization.
                    If set to THOPT_LOOCV, Leave-one-out-cross-validation will be performed for threshold optimization.
                    This will increase memory usage, since the WHO features of all samples have to be stored for this,
                    and will slow down threshold optimization as well as the model learning step if only one WHO cluster
                    is used. But it will guarantee that no model is tested against a sample it has been learned from
                    for threshold optimization, like it's done with THOPT_OVERLAPPING.
                    Setting this to THOPT_NONE will turn off threshold optimization. In that case, estimated thresholds will be used.
        progressCallback - Optionally, a callback which is called between the steps of the learning process to populate the progress.  
                           The first parameter to the callback will be the number of steps performed in the entire process,
                           the second one will be the total number of steps. To date, the entire process is divided into three steps:
                           image extraction, model creation and threshold optimization.  
                           The third and fourth parameters will be the number of performed steps and the total number of steps of
                           the current sub-procedure.
        debug - If set to True, timing and debugging information will be printed to stdout.
        """
        
        if (libartos is None):
            raise RuntimeError('Can not find libartos')
        if isinstance(imageRepository, imagenet.ImageRepository):
            repoDirectory = imageRepository.repoDirectory
        else:
            repoDirectory = imageRepository
        cb = artos_wrapper.overall_progress_cb_t(progressCallback) if not progressCallback is None \
             else ctypes.cast(None, artos_wrapper.overall_progress_cb_t)
        libartos.learn_imagenet(utils.str2bytes(repoDirectory), utils.str2bytes(synsetId), utils.str2bytes(bgFile), \
                                utils.str2bytes(modelfile), add, maxAspectClusters, maxWHOClusters, \
                                thOptNumPositive, thOptNumNegative, thOptMode, cb, debug)


    @staticmethod
    def learnModelFromFiles(imageFiles, boundingBoxes, bgFile, modelfile, add = True, \
                            maxAspectClusters = 2, maxWHOClusters = 3, \
                            thOptNumPositive = 0, thOptNumNegative = 0, thOptMode = THOPT_LOOCV, progressCallback = None, \
                            debug = False):
        """All-in-one short-cut method for learning a new WHO model from JPEG files.
        
        imageFiles - Sequence with paths of JPEG files to use as positive samples.
        boundingBoxes - Sequence with BoundingBox instances or (left, top, right, bottom) tuples which specify
                        the bounding boxes around the objects to learn corresponding to the given images.
                        Must have exactly as many components as imageFiles or must be an empty sequence. In the latter case,
                        the entire images will be considered as showing the object of interest. The same can be done for
                        a single image by specifying None as bounding box.
        bgFile - Path to a file containing the stationary background statistics for whitening (negative mean and covariance).
        modelfile - Path to the file which the newly learned models will be written to
                    (the file will be created if it does not exist).
        add - If set to true, the new models will be added as additional mixture components if the model file does
              already exist, otherwise the model file will be overwritten with just the new mixture.
        maxAspectClusters - Maximum number of clusters to form by the aspect ratio of the samples.
        maxWHOClusters - Maximum number of clusters to form by the WHO feature vectors of the samples of a single aspect ratio cluster.
        thOptNumPositive - Maximum number of positive samples to test the models against for finding the optimal threshold.
                           Set this to 0 to run the detector against all samples.
        thOptNumNegative - Number of negative samples taken from different synsets for finding the optimal threshold.
                           Every detection on one of these images will be considered as a false positive.
        thOptMode - Controls the mode of threshold optimization.
                    If set to THOPT_LOOCV, Leave-one-out-cross-validation will be performed for threshold optimization.
                    This will increase memory usage, since the WHO features of all samples have to be stored for this,
                    and will slow down threshold optimization as well as the model learning step if only one WHO cluster
                    is used. But it will guarantee that no model is tested against a sample it has been learned from
                    for threshold optimization, like it's done with THOPT_OVERLAPPING.
                    Setting this to THOPT_NONE will turn off threshold optimization. In that case, estimated thresholds will be used.
        progressCallback - Optionally, a callback which is called between the steps of the learning process to populate the progress.  
                           The first parameter to the callback will be the number of steps performed in the entire process,
                           the second one will be the total number of steps. To date, the entire process is divided into three steps:
                           image reading & decoding, model creation and threshold optimization.  
                           The third and fourth parameters will be the number of performed steps and the total number of steps of
                           the current sub-procedure.
        debug - If set to True, timing and debugging information will be printed to stdout.
        """
        
        # Check parameters
        if (libartos is None):
            raise RuntimeError('Can not find libartos')
        if (not boundingBoxes is None) and (len(boundingBoxes) > 0) and (len(boundingBoxes) != len(imageFiles)):
            raise RuntimeError('Number of given bounding boxes ({}) is different from number of given images ({})'.format(
                               len(boundingBoxes), len(imageFiles)))
       
        # Build file name array
        filenames = (ctypes.c_char_p * len(imageFiles))(*(utils.str2bytes(f) for f in imageFiles))
       
        # Convert bounding box sequence to FlatBoundingBox array
        if len(boundingBoxes) > 0:
            bboxes = (artos_wrapper.FlatBoundingBox * len(boundingBoxes))()
            for i, b in enumerate(boundingBoxes):
                if b is None:
                    bboxes[i] = artos_wrapper.FlatBoundingBox(0, 0, 0, 0)
                else:
                    box = BoundingBox(b)
                    bboxes[i] = artos_wrapper.FlatBoundingBox(left = box.left, top = box.top, width = box.width, height = box.height)
        else:
            bboxes = None
        
        # Learn model
        cb = artos_wrapper.overall_progress_cb_t(progressCallback) if not progressCallback is None \
             else ctypes.cast(None, artos_wrapper.overall_progress_cb_t)
        libartos.learn_files_jpeg(filenames, len(filenames), bboxes, utils.str2bytes(bgFile), utils.str2bytes(modelfile), \
                                  add, maxAspectClusters, maxWHOClusters, thOptNumPositive, thOptNumNegative, thOptMode, cb, True)



class Model(object):
    """Reads, modifies and visualizes model mixture files.
    
    The model will be read into a hierarchical structure of tuples, though, of course, we could use
    a 3-D numpy array for representing a model, but this is not such a performance critical task that
    we want to add numpy as a dependency just for reading DPM models.
    """
    
    def __init__(self, modelfile):
        """Reads models from a mixture file.
        
        modelfile - The file name of the model file.
        """
        
        object.__init__(self)
        self.filename = modelfile
        with open(modelfile, 'r') as file:
            self._readFile(file)
    
    
    def _readFile(self, file):
        """Reads a mixture file into the `models` list and sets the `type` and `parameters` attributes.
        
        Each component in the `models` list represents a component of the mixture and is a dictionary
        with the keys 'parts' and 'bias'. 'parts' is a list with the parts of that component
        which are dictionaries with the keys 'data', 'offset' (2-tuple) and 'params' (deformation
        coefficients, 4-tuple). 'data' is a list with the rows of the part. Finally, each row is a
        list with the cells in that row and each cell is a list with the feature weights.
        
        The `type` attribute is a string specifying the type of feature extractor used to create the
        model and `parameters` is a dictionary with the parameters of the feature extractor.
        """
        
        self.models = []
        line = file.readline().strip()
        try:
            # Old model format (v1)
            mixtures = int(line) # numMixtures
            self.formatVersion = 1
            self.type = "HOG"
            self.parameters = { 'cellSizeX' : 8, 'cellSizeY' : 8 }
        except ValueError:
            # New model format (v2)
            self.formatVersion = 2
            self.type = line
            self._parseParameters(file.readline().strip())
            line = ""
            while line == "":
                line = file.readline().strip()
            mixtures = int(line)
            
        for mix in range(mixtures):
            line = ""
            while line == "":
                line = file.readline().strip()
            mixDescr = line.split() # numParts bias
            mixture = { 'parts' : [], 'bias' : float(mixDescr[1]) }
            for p in range(int(mixDescr[0])):
                partDescr = file.readline().split() # rows cols features x y a b c d
                rows = int(partDescr[0])
                cols = int(partDescr[1])
                feats = int(partDescr[2])
                offset = (int(partDescr[3]), int(partDescr[4]))
                params = (float(partDescr[5]), float(partDescr[6]), float(partDescr[7]), float(partDescr[8]))
                part = [[[0 for k in range(feats)] for j in range(cols)] for i in range(rows)]
                for row in range(rows):
                    line = file.readline().split()
                    for col in range(cols):
                        for feat in range(feats):
                            part[row][col][feat] = float(line[col * feats + feat])
                mixture['parts'].append({ 'data' : part, 'offset' : offset, 'params' : params })
            self.models.append(mixture)
    
    
    def _parseParameters(self, paramLine):
        """Parses feature extractor parameters given as single string into the `parameters` dictionary."""
        
        self.parameters = {}
        tok = paramLine.split()
        
        # Merge string sequences
        isString = [False] * len(tok)
        stringStart = -1
        i = 0
        while i < len(tok):
            if stringStart >= 0:
                tok[stringStart] += tok[i]
            elif tok[i].startswith("{str{"):
                stringStart = i
                isString[stringStart] = True
            i += 1
            if (stringStart >= 0) and (tok[stringStart].endswith("}str}")):
                tok[stringStart] = tok[stringStart][5:-5]
                del tok[stringStart+1:i]
                i = stringStart + 1
                stringStart = -1
        
        # Parse parameters
        if len(tok) % 2 != 0:
            raise ValueError("Invalid feature extractor parameter line given.")
        for i in range(0, len(tok), 2):
            if isString[i+1]:
                self.parameters[tok[i]] = tok[i+1]
            else:
                try:
                    self.parameters[tok[i]] = int(tok[i+1])
                except ValueError:
                    try:
                        self.parameters[tok[i]] = float(tok[i+1])
                    except ValueError:
                        self.parameters[tok[i]] = tok[i+1]
    
    
    def removeComponent(self, compId):
        """Removes the model with the given index in the `models` attribute from this mixture."""
        
        if (compId >= 0) and (compId < len(self.models)):
            del self.models[compId]
    
    
    def save(self):
        """Writes the model data back to the file it was loaded from."""
        
        with open(self.filename, 'w') as f:
            if self.formatVersion > 1:
                f.write('{}\n'.format(self.type))
                f.write(' '.join('{} {}'.format(k, '{str{' + v + '}str}' if utils.is_str(v) else v)
                                 for k, v in self.parameters.items()) + '\n\n')
        
            f.write('{}\n'.format(len(self.models)))
            for model in self.models:
                f.write('{} {}\n'.format(len(model['parts']), model['bias']))
                for part in model['parts']:
                    f.write('{rows} {cols} {features} {offset[0]} {offset[1]} {params[0]} {params[1]} {params[2]} {params[3]}\n'.format(
                            rows = len(part['data']), cols = len(part['data'][0]), features = len(part['data'][0][0]),
                            offset = part['offset'], params = (part['params'] if part['params'] != (0, 0, 0, 0) else (0, 0, 0, 0))))
                    for row in part['data']:
                        for col in row:
                            for feat in col:
                                f.write('{} '.format(feat if int(feat) != feat else int(feat)))
                        f.write('\n')
                f.write('\n')
    
    
    def visualize(self, cs = 40, padding = 10):
        """Visualizes this model.
        
        Creates positive and negative HOG images for the root of each mixture component of this model
        and combines them in a single PIL image.
        `cs` specifies the size of each cell in the HOG images.
        `padding` specifies the padding between the single HOG images.
        Returns the created PIL image, which will be of mode 'RGBA'.
        Raises a TypeError if this model does not use HOG features.
        """
        
        if self.type != "HOG":
            raise TypeError("Visualization is only available for HOG models (given: " + self.type + ")")
        
        images = []
        for comp in self.models:
            images.append((self.__class__.hogImage(comp['parts'][0]['data'], cs, False), \
                           self.__class__.hogImage(comp['parts'][0]['data'], cs, True)))
        maxWidth = max(p.size[0] + n.size[0] for p, n in images)
        height = sum(p.size[1] for p, n in images)
        img = Image.new("RGBA", (maxWidth + padding, height + (len(images) - 1) * padding * 2), (255, 255, 255, 0))
        y = 0
        for p, n in images:
            x = (maxWidth - p.size[0] - n.size[0] - padding) // 2
            img.paste(p, (x, y))
            img.paste(n, (x + p.size[0] + padding, y))
            y += p.size[1] + padding * 2
        return img
    
    
    @staticmethod
    def hogImage(m, cs = 40, neg = False):
        """Creates a PIL image visualizing a given HOG model.
        
        Visualizes the given HOG feature vector `m`, which must be a list (rows) of lists (columns)
        of lists (features of a single cell). The cell feature vectors must have exactly at least
        27 components with the components at indices 18 to 26 being the 'unsigned' or
        'contrast insensitive' HOG features, which will be visualized.
        
        `cs` specifies the size of each cell in the resulting image.
        
        By default, only the positive features will be visualized. If `neg` is set to true,
        the negative features will be used instead.
        
        Returns the created PIL image, which will be of mode 'L' (Intensity).
        """
        
        img = Image.new('L', (len(m[0]) * cs, len(m) * cs), 0)
        draw = ImageDraw.Draw(img)
        if not neg:
            maxIntensity = max(x for i in m for j in i for x in j[18:27])
        else:
            maxIntensity = -1 * min(x for i in m for j in i for x in j[18:27])
        for rowIndex, row in enumerate(m):
            for colIndex, col in enumerate(row):
                center = (colIndex * cs + cs // 2, rowIndex * cs + cs // 2)
                features = col[18:27]
                features = [max(x, 0) for x in features] if not neg else [-1 * min(x, 0) for x in features]
                maxFeat = max(features)
                if maxFeat > 0:
                    for featIndex, feat in enumerate(features):
                        angle = math.pi / 2 - (float(featIndex) / 9) * math.pi
                        intensity = round(((feat / maxIntensity) ** 0.4) * 255)
                        length = round((feat / maxFeat) * (cs / 2))
                        x1 = round(center[0] + math.cos(angle) * length)
                        y1 = round(center[1] - math.sin(angle) * length)
                        x2 = round(center[0] + math.cos(angle + math.pi) * length)
                        y2 = round(center[1] - math.sin(angle + math.pi) * length)
                        draw.line(((x1,y1), (x2,y2)), fill = intensity)
        return img



class ModelManager(object):
    """Manages a model list file.
    
    The model list file contains information about one model per row in the following format:
    
    classname modelfile threshold [synset-id]
    
    The single components of a record like that are separated by spaces. `classname` and `modelfile`
    may contain spaces if they are surrounded by quotes. `synset-id` is optional.
    Lines beginning with a hash-sign (#) as well as empty lines will be ignored, with one exception:
    If the text directly after the hash-sign is a valid model description, it will be read, but flagged
    as disabled.
    """
    
    def __init__(self, listfile):
        """Reads a model list file.
        
        listfile - The path of the list file. Does not need to exist.
        """
        
        object.__init__(self)
        self.filename = os.path.abspath(listfile)
        try:
            with open(listfile) as file:
                self._readFile(file)
        except:
            self.models = []
    
    
    def _readFile(self, file):
        """Tries to read a opened list file into the `models` list.
        
        `models` will be a list of dictionaries with the keys 'classname', 'modelfile', 'threshold',
        'synsetId' (may be None) and 'disabled'.
        
        Raises a RuntimeError or ValueError if the file could not be parsed.
        """
        
        self.models = []
        lineRE = re.compile(r'("[^"]+"|\S+)\s+("[^"]+"|\S+)\s+(-?[0-9]+(?:\.[0-9]+)?)(?:\s+([a-z0-9]+))?', re.I)
        for line in file:
            if (line.strip() != ''):
                if (line[0] != '#'):
                    info = lineRE.match(line.strip())
                    disabled = False
                    if not info:
                        raise RuntimeError('Invalid line in model list file:\n{}'.format(line))
                else:
                    # Try to parse part after # as model description
                    info = lineRE.match(line[1:].strip())
                    if not info:
                        continue
                    disabled = True
                self.models.append({
                    'classname' : info.group(1).strip('"'),
                    'modelfile' : info.group(2).strip('"'),
                    'threshold' : float(info.group(3)),
                    'synsetId'  : info.group(4),
                    'disabled'  : disabled
                })
    
    
    def save(self):
        """Writes the model list back to the file it was read from."""
        
        with open(self.filename, 'w') as f:
            for model in self.models:
                if model['disabled']:
                    f.write('#')
                f.write('{classname} {modelfile} {threshold}'.format(
                    classname = '"{}"'.format(model['classname']) if re.search(r'\s', model['classname']) else model['classname'],
                    modelfile = '"{}"'.format(model['modelfile']) if re.search(r'\s', model['modelfile']) else model['modelfile'],
                    threshold = model['threshold']
                ))
                if (not model['synsetId'] is None) and (not model['synsetId'] == ''):
                    f.write(' {}'.format(model['synsetId']))
                f.write('\n')
    
    
    def addModel(self, modelfile, classname, threshold = 0.0, synsetId = None, disabled = False):
        """Adds a new model to the list file.
        
        Note that an explicit call to save() is necessary to actually write the updated list file.
        
        modelfile - The file name of the model file. May be relative to the location of the list file.
                    This function will make absolute paths relative automatically if the model file is
                    located in the same directory as the model list file or some sub-directory.
        classname - The name of the class the model was learned for.
        threshold - Detection score threshold for the model.
        synsetId  - Optionally, the ID of the ImageNet synset the model is associated with.
        disabled  - If set to true, the record will be written to the model list file, but will be 'commented out'.
        """
        
        abspath = modelfile if os.path.isabs(modelfile) else os.path.abspath(modelfile)
        basedir = os.path.dirname(self.filename)
        if basedir[-1] not in ('/', '\\'):
            basedir += os.path.sep
        if abspath.startswith(basedir):
            modelfile = abspath[len(basedir):]
        self.models.append({
            'classname' : classname.strip(),
            'modelfile' : modelfile,
            'threshold' : float(threshold),
            'synsetId'  : synsetId if synsetId != '' else None,
            'disabled'  : bool(disabled)
        })
    
    
    def deleteModel(self, modelIndex):
        """Deletes a model, not just from the list file, but also from disk.
        
        This implies an automatic call to save().
        
        modelIndex - The index of the model in the `models` list.
        """
        
        if isinstance(modelIndex, int) and (modelIndex >= 0) and (modelIndex < len(self.models)):
            modelfile = self.models[modelIndex]['modelfile']
            if not os.path.isabs(modelfile):
                modelfile = os.path.join(os.path.dirname(self.filename), modelfile)
            if os.path.isfile(modelfile):
                os.remove(modelfile)
            del self.models[modelIndex]
            self.save()
    
    
    def getModelPath(self, modelIndex):
        """Returns the full absolute path to a specific model file at the index given by `modelIndex` in the `models` list."""
        
        if (modelIndex < 0) or (modelIndex >= len(self.models)):
            return None
        modelfile = self.models[modelIndex]['modelfile']
        return modelfile if os.path.isabs(modelfile) else os.path.join(os.path.dirname(self.filename), modelfile)
    
    
    def readModel(self, model):
        """Reads a model file and returns it as Model instance.
        
        model - Either the name of the model file (may be relative to the location of the model list file)
                or the index of the model in the `models` list.
        Returns: A Model instance or None if the model file could not be found.
        """
        
        if isinstance(model, int):
            if (model < 0) or (model >= len(self.models)):
                return None
            model = self.models[model]['modelfile']
        if not os.path.isabs(model):
            model = os.path.join(os.path.dirname(self.filename), model)
        return Model(model) if os.path.isfile(model) else None



class FeatureExtractor(object):
    """Represents a feature extractor used by libartos."""
    
    def __init__(self, type = None):
        """Initializes a feature extractor.
        
        type - The type specifier of the feature extractor. If set to `None`, the new object will represent the
               default feature extractor.
        """
        
        if not type:
            info = artos_wrapper.FeatureExtractorInfo()
            libartos.feature_extractor_get_info(info)
            self._type = utils.bytes2str(info.type)
            self._name = utils.bytes2str(info.name)
        else:
            featureExtractors = self.__class__.listFeatureExtractors()
            if type not in featureExtractors:
                raise artos_wrapper.LibARTOSException(artos_wrapper.SETTINGS_RES_UNKNOWN_FEATURE_EXTRACTOR)
            self._type = type
            self._name = featureExtractors[type]
        self._params = {}
    
    
    @property
    def type(self):
        """The type specifier of this feature extractor."""
        return self._type
    
    
    @property
    def name(self):
        """The human-readable name of this feature extractor."""
        return self._name
    
    
    @property
    def isDefault(self):
        """True if this feature extractor is the current default feature extractor."""
        return self.type == self.__class__().type
    
    
    def setAsDefault(self):
        """Install this feature extractor as the default feature extractor used by libartos."""
        
        libartos.change_feature_extractor(utils.str2bytes(self.type))
        for k, v in self._params.items():
            self.setParam(k, v)
    
    
    def getParams(self):
        """Retrieves all parameters supported by this feature extractor along with their values.
        
        Returns a dictionary whose keys are parameter names and whose values are the values of the
        respective parameter.
        If this feature extractor is the current default feature extractor, its current values will
        be specified, otherwise the default values are used.
        """
        
        if len(self._params) == 0:
        
            getParams = (lambda buf, bufSize: libartos.feature_extractor_list_params(buf, bufSize)) \
                        if self.isDefault \
                        else (lambda buf, bufSize: libartos.list_feature_extractor_params(utils.str2bytes(self.type), buf, bufSize))
            numParams = ctypes.c_uint()
            getParams(None, numParams)
            paramBuf = (artos_wrapper.FeatureExtractorParameter * numParams.value)()
            getParams(paramBuf, numParams)
            
            for i in range(numParams.value):
                val = None
                try:
                    if paramBuf[i].type == artos_wrapper.PARAM_TYPE_INT:
                        val = paramBuf[i].val.intVal
                    elif paramBuf[i].type == artos_wrapper.PARAM_TYPE_SCALAR:
                        val = paramBuf[i].val.scalarVal
                    elif paramBuf[i].type == artos_wrapper.PARAM_TYPE_STRING:
                        val = utils.bytes2str(paramBuf[i].val.stringVal)
                except:
                    pass
                self._params[utils.bytes2str(paramBuf[i].name)] = val
        
        return self._params
    
    
    def setParam(self, name, value):
        """Changes to value of a parameter of this feature extractor.
        
        name - The name of the parameter to be set.
        value - The value of the parameter to be set. The type if the parameter will be inferred
                from the type of the value which may be int, float or str.
        """
        
        if value is None:
            return
        
        if self.isDefault:
            if isinstance(value, int):
                libartos.feature_extractor_set_int_param(utils.str2bytes(name), value)
            elif isinstance(value, float):
                libartos.feature_extractor_set_scalar_param(utils.str2bytes(name), value)
            elif utils.is_str(value):
                libartos.feature_extractor_set_string_param(utils.str2bytes(name), utils.str2bytes(value))
        
        self.getParams()
        self._params[name] = value
    
    
    @staticmethod
    def listFeatureExtractors():
        """Lists all available feature extraction methods.
        
        Returns: A dictionary with the type specifiers of the feature extractors as keys and their
                 human-readable names as values.
        """
        
        numFE = ctypes.c_uint()
        libartos.list_feature_extractors(None, numFE)
        info = (artos_wrapper.FeatureExtractorInfo * numFE.value)()
        libartos.list_feature_extractors(info, numFE)
        return dict((utils.bytes2str(info[i].type), utils.bytes2str(info[i].name)) for i in range(numFE.value))

