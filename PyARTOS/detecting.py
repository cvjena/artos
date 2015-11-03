"""Interface to the detection part of LibARTOS.

The class Detector can be used to detect objects matching a given
set of models in images.
"""


from . import artos_wrapper, utils
from .artos_wrapper import libartos
try:
    from PIL import Image, ImageDraw
except:
    import Image, ImageDraw
import ctypes
import sys


class BoundingBox(object):
    """A rectangle, used to specify bounding boxes of objects in images.

    The bounding box is defined by the coordinates of it's four edges:
    left, top, right, bottom, whereat left and top are inside of the box,
    but right and bottom are outside of it, so that the width of the box
    can be calculated as (right - left).
    Besides those four attributes, you can also read and write width, height, x and y.
    Setting width or height changes the value of right or bottom, respectively.
    Setting x or y moves the box to a new origin while keeping width and height.
    """

    def __init__(self, *coords, **kwargs):
        """Construct a new bounding box by giving 4 coordinates or 2 coordinates combined with width and height.
        
        The following calls are equivalent:
        BoundingBox(left, top, right, bottom)
        BoundingBox((left, top, right, bottom))
        BoundingBox(BoundingBox(left, top, right, bottom))
        BoundingBox(left, top, width = (right - left), height = (bottom - top))
        """
        
        object.__init__(self)
        if (len(coords) == 1):
            if ((isinstance(coords[0], tuple) or isinstance(coords[0], list)) and (len(coords[0]) == 4)):
                self._coords = list(coords[0])
            elif isinstance(coords[0], BoundingBox):
                self._coords = coords[0].coords
            else:
                raise TypeError('First argument to BoundingBox constructor must be a sequence of 4 integral coordinates.')
        elif (len(coords) == 4):
            self._coords = list(coords)
        elif (len(coords) == 2) and ('width' in kwargs) and ('height' in kwargs):
            self._coords = [coords[0], coords[1], coords[0] + kwargs['width'], coords[1] + kwargs['height']]
        else:
            raise TypeError('BoundingBox constructor must be called with 4 integral arguments.')
        if ((self._coords[2] < self._coords[0]) or (self._coords[3] < self._coords[1])):
            raise TypeError('The coordinates of the right and bottom borders of a bounding box must not be less than those of the top and left borders.')


    def __getattr__(self, name):
        if (name == 'coords'):
            return self._coords
        elif (name == 'left'):
            return self._coords[0]
        elif (name == 'top'):
            return self._coords[1]
        elif (name == 'right'):
            return self._coords[2]
        elif (name == 'bottom'):
            return self._coords[3]
        elif (name == 'width'):
            return (self._coords[2] - self._coords[0])
        elif (name == 'height'):
            return (self._coords[3] - self._coords[1])
        else:
            raise AttributeError('Attribute {} is not defined.'.format(name))


    def __setattr__(self, name, value):
        if ((name in ('coords', '_coords')) and isinstance(value, tuple)):
            value = list(value)
        if ((name in ('coords', '_coords')) and ((not isinstance(value, list)) or (len(value) != 4) or (value[2] < value[0]) or (value[3] < value[1]))):
            if ((not isinstance(value, list)) or (len(value) != 4)):
                raise TypeError('Attribute {} must be a sequence of length 4.'.format(name))
            else:
                raise TypeError('The coordinates of the right and bottom borders of a bounding box must not be less than those of the top and left borders.')
        elif (name == 'left'):
            self._coords[0] = value
        elif (name == 'top'):
            self._coords[1] = value
        elif (name == 'right'):
            self._coords[2] = value
        elif (name == 'bottom'):
            self._coords[3] = value
        elif (name == 'x'):
            self._coords[2] += value - self._coords[0]
            self._coords[0] = value
        elif (name == 'y'):
            self._coords[3] += value - self._coords[1]
            self._coords[1] = value
        elif (name == 'width'):
            self._coords[2] = self._coords[0] + value
        elif (name == 'height'):
            self._coords[3] = self._coords[1] + value
        else:
            object.__setattr__(self, name, value)


    def scale(self, ratio = 1.0):
        """Creates a scaled copy of this bounding box.
        
        The parameter specifies the ratio of the size of the new bounding box to
        the size of the original one. For instance, 0.5 results in a bounding box half the
        size of this one. The position of the box will be changed too.
        """
        
        return BoundingBox([int(round(x * ratio)) for x in self.coords])


    def drawToImage(self, img, color = (255,0,0), width = 2):
        """Draws the bounding box onto an image.
        
        The first parameter to this function is the PIL.Image object
        and the second one is the colour of the bounding box to be drawn
        as RGB tuple, while the third parameter specifies the width of the border.
        
        The image will be converted to RGB color space, if it isn't already.
        
        This function returns the image object, which the bounding box has been drawn to,
        which must not be identical (in terms of instance identity) to the img argument.
        Because if a color space conversion is performed, a new Image object will be created.
        """
        
        if (img.mode != 'RGB'):
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Draw inner white frame
        draw.rectangle((self.left, self.top, self.right, self.bottom), outline = (255, 255, 255))
        # Draw coloured border
        for i in range(1, width + 1):
            draw.rectangle((self.left - i, self.top - i, self.right + i, self.bottom + i), outline = color)
        # Draw outer white frame
        draw.rectangle((self.left - width - 1, self.top - width - 1, self.right + width + 1, self.bottom + width + 1), outline = (255, 255, 255))
        
        return img



class Detection(BoundingBox):
    """Stores information about an object detected on an image.

    Instances of this class are bounding boxes, specifying a rectangle on the image
    which contains the object.
    Further attributes are:
    classname - The name of the class of the detected object
    synsetId - The ID of the ImageNet synset associated with the class of the detected object
    score - The detection score
    """

    def __init__(self, classname, synsetId, score, *coords):
        """Constructs a new detection result.
        
        Syntax:
        Detection(classname, synsetId, score, left, top, right, bottom)
        Detection(classname, synsetId, score, (left, top, right, bottom))
        Detection(classname, synsetId, score, boundingBox)
        """
        
        BoundingBox.__init__(self, *coords)
        self.classname = classname
        self.synsetId = synsetId
        self.score = score


    @staticmethod
    def fromFlatDetection(flatDetection):
        """Class method, that creates a Detection instance from a FlatDetection instance of the LibARTOS wrapper."""
        
        if not isinstance(flatDetection, artos_wrapper.FlatDetection):
            raise TypeError('1st argument to Detection.fromFlatDetection must be an instance of FlatDetection')
        # flatDetection.classname is of type c_char_p. In Python 2, it is converted to str automatically, in Python 3 it will be bytes.
        # In the latter case, we have to do the conversion to str here by ourselves:
        classname = utils.bytes2str(flatDetection.classname)
        # The same holds for flatDetection.synset_id
        synsetId = utils.bytes2str(flatDetection.synset_id)
        return Detection(classname, synsetId, flatDetection.score, flatDetection.left, flatDetection.top, flatDetection.right, flatDetection.bottom)



class Detector(object):

    def __init__(self, overlap = 0.5, padding = 12, interval = 10, debug=False):
        """Constructs and initializes a new detector with specific settings.
        
        overlap - Minimum overlap in non maxima suppression.
        padding - Amount of zero padding in HOG cells. Must be greater or equal to half the greatest filter dimension.
        interval - Number of levels per octave in the HOG pyramid.
        """
        
        object.__init__(self)
        if (libartos is None):
            raise RuntimeError('Can not find libartos')
        self.handle = libartos.create_detector(overlap, padding, interval, debug)


    def __del__(self):
        try:
            libartos.destroy_detector(self.handle)
        except:
            pass


    def addModel(self, classname, modelfile, threshold, synsetId = ''):
        """Adds a model to the detection stack.
        
        classname - The name of the class ('bicycle' for example). It is used to name the objects detected in an image.
        modelfile - The filename of the model to load.
        threshold - The detection threshold for this model.
        synsetId  - The ID of the ImageNet synset associated with the class. Will be present in the detection results.
        
        If the model can not be added, a LibARTOSException is thrown.
        """
        
        synsetId = utils.str2bytes(synsetId) if ((synsetId != '') and (not synsetId is None)) else None
        libartos.add_model(self.handle, utils.str2bytes(classname), utils.str2bytes(modelfile), threshold, synsetId)


    def addModels(self, listfile):
        """Adds multiple models to the detection stack at once using information given in a file enumerating the models.
        
        The list file contains one model per line.
        Each model is described by
        1. the name of the class,
        2. the filename of the model file (must not contain white spaces),
        3. the detection threshold as double,
        4. optionally, the ID of the ImageNet synset associated with the class.
        Those 3 or 4 components are separated by spaces, while the class and file name can be surrounded by quotes
        to enable spaces within them. Lines starting with a hash-sign ('#') as well as empty lines will be ignored.
        
        modellistfn - The filename of the model list.
        Returns: The number of successfully added models.
        
        If the model list file can not be read, a LibARTOSException is thrown.
        """
        
        return libartos.add_models(self.handle, utils.str2bytes(listfile))


    def detect(self, img, limit = 3):
        """Detects objects in a given image which match one of the models added before using addModel() or addModels().
        
        img - Either a PIL.Image.Image object or a path to a JPEG file. In the latter case, the image will be read
              directly by the library.
        limit - Maximum number of detections returned (affects memory allocated for library call)
        Returns: A list of objects detected in img, each described by an instance of the Detection class.
        
        If an error occurs, a LibARTOSException is thrown.
        """
        
        if not (isinstance(img, Image.Image) or utils.is_str(img)):
            raise TypeError('{0}.detect expects argument img to be either PIL.Image.Image or string'.format(self.__class__.__name__))
        if (limit < 1):
            limit = 1
        
        # Allocate buffer memory, where the library will store the detection results
        buf_size = ctypes.c_uint(limit)
        buf = (artos_wrapper.FlatDetection * buf_size.value)()
        
        # Run detector
        if isinstance(img, Image.Image):
            # Convert image to plain RGB or grayscale
            if (img.mode in ('1', 'L')):
                grayscale = True
                if (img.mode != 'L'):
                    img = img.convert('L')
            else:
                grayscale = False
                if (img.mode != 'RGB'):
                    img = img.convert('RGB')
            # Copy raw image data into a buffer
            numbytes = img.size[0] * img.size[1]
            if not grayscale:
                numbytes = numbytes * 3
            try:
                imgbytes = img.tobytes()
            except:
                imgbytes = img.tostring()
            imgdata = ctypes.create_string_buffer(numbytes)
            ctypes.memmove(imgdata, imgbytes, numbytes)
            # Run detector
            libartos.detect_raw(self.handle, ctypes.cast(imgdata, artos_wrapper.c_ubyte_p), img.size[0], img.size[1], grayscale, buf, buf_size)
        else:
            # Treat img as filename
            libartos.detect_file_jpeg(self.handle, utils.str2bytes(img), buf, buf_size)
       
        # Convert detection results (buf_size is set to the actual number of detection results by the library)
        return [Detection.fromFlatDetection(buf[i]) for i in range(buf_size.value)]
