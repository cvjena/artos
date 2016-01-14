"""Provides some common helper functions."""

import math, re, os.path, sys, time, ctypes
try:
    from PIL import Image, ImageTk
except:
    import Image, ImageTk


basedir = os.path.dirname(os.path.abspath(__file__))


def is_str(x):
    """Checks if a given value is a string (8-bit or unicode).
    
    In Python 2, a string can be either of type 'str' or 'unicode', but the 'unicode' type is
    gone in Python 3. This helper function handles both cases.
    """

    if (isinstance(x, str)):
        return True
    else:
        try:
            return True if (isinstance(x, unicode)) else False
        except NameError:
            return False


def str2bytes(s):
    """Converts an 8-bit or unicode string to a byte stream.
    
    For foreign library functions which take (8-bit) char arguments, Python's unicode strings have to be converted
    to byte streams. Since such arguments are usually filenames, we use the default file system encoding as target
    encoding of the byte stream.
    """
    
    fse = sys.getfilesystemencoding()
    try:
        return s.encode(fse, 'ignore') # Fine in Python 3, but may fail in Python 2 if s contains non-ascii characters
    except:
        # Tell the decoder which encoding is used in s and create a unicode string, that will be then converted to bytes
        return s.decode(fse, 'ignore').encode(fse, 'ignore') # decode() method is Python 2 only


def bytes2str(b):
    """Converts bytes to a native python string.
    
    Calls to foreign library functions, which return a null-terminated C string in some way, Python 2 will automatically
    convert it to str, while Python 3 turns it into bytes. This function handles both cases and returns a string.
    """

    return b if is_str(b) else b.decode(sys.getfilesystemencoding(), 'replace')


def classnameFromFilename(fn):
    """Turns a filename (of a model) into a nicely readable class name.
    
    This is done by stripping the file extension, turning hyphens and underscores into spaces
    and capitalizing the first character of each word.
    """

    return re.sub('[_-]+', ' ', os.path.splitext(os.path.basename(fn))[0]).title()


def splitFilenames(filenames):
    """Splits a string containing multiple file paths.
    
    The single file paths in the input string are separated by white-spaces.
    If a path contains white-spaces, it is surrounded by { and } in the input string.
    Such a list of paths is returned by the file dialogs of Tk for instance.
    This function returns a list with the single file names (without any encapsulating { or }).
    """

    # Return input unchanged if it isn't a string
    if not is_str(filenames):
        return filenames;
    
    # Split string into tokens separated by spaces
    filenames = filenames.strip().split(' ')
    # Loop over filenames and write them to fn, while entering "concatenation mode" when
    # hitting { und leaving concatenation mode on }.
    fn = []
    concat = False
    for f in filenames:
        if (concat):
            if (len(f) > 0) and (f[-1] == '}'):
                f = f[:-1]
                concat = False
            fn[-1] += ' ' + f
        elif (len(f) > 0):
            if (f[0] == '{'):
                if (f[-1] == '}'):
                    f = f[1:-1]
                else:
                    f = f[1:]
                    concat = True
            fn.append(f)
    return fn


def imgResizeCropped(img, size, filter = Image.ANTIALIAS):
    """Resizes a PIL Image by cropping instead of stretching.
    
    The given image will be resized and cropped if it's aspect ratio doesn't match
    the one of the new size.
    
    img - The image to be resized.
    size - 2-tuple with the new size of the image as (width, height).
    filter - The resampling kernel to use (NEAREST, BILINEAR, BICUBIC or ANTIALIAS).
    
    Returns: The resized image.
    """
    
    imgAspect = float(img.size[0]) / float(img.size[1])
    newAspect = float(size[0]) / float(size[1])
    if newAspect <= imgAspect:
        # Resize to height
        result = img.resize((int(imgAspect * size[1]), size[1]), filter)
    else:
        # Resize to width
        result = img.resize((size[0], int(size[0] / imgAspect)), filter)
    # Crop
    left = (result.size[0] - size[0]) // 2
    top = (result.size[1] - size[1]) // 2
    result = result.crop((left, top, left + size[0], top + size[1]))
    return result


def img2buffer(img):
    """Creates a raw ctypes string buffer containing the raw pixel data of a given image.
    
    img - The image to convert to raw pixel data.
    
    Returns: A tuple consisting of the string buffer and a boolean variable that indicates if
             the image is a grayscale image (true) or an RGB image (false).
    """
    
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
    return imgdata, grayscale


def figure2img(fig, w, h):
    """Converts a matplotlib figure to PIL.Image.Image instance.
    
    fig - The matplotlib figure.
    w - The width of the resulting image.
    h - The height of the resulting image.
    """
    
    dpi = fig.get_dpi()
    fig.set_size_inches(float(w) / dpi, float(h) / dpi)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_argb = fig.canvas.tostring_argb()
    # canvas.tostring_argb returns pixmap in ARGB mode. Roll the alpha channel to have it in RGBA mode:
    if isinstance(buf_argb, str):
        # Python 2
        buf_rgba = ''.join(buf_argb[i+j] for i in range(0, 4*w*h, 4) for j in (1,2,3,0))
    else:
        # Python 3
        buf_rgba = bytes(buf_argb[i+j] for i in range(0, 4*w*h, 4) for j in (1,2,3,0))
    return Image.frombytes('RGBA', (w, h), buf_rgba)



class Timer(object):
    """A timer used to measure the execution time of (multiple runs of) a code snippet."""


    def __init__(self, maxRuns = 0):
        """Constructs a new timer.
        
        If maxRuns is greater than 0, it limits the number of captured spaces of time.
        When the limit is reached, the oldest measure will be removed.
        """
        
        object.__init__(self)
        self._maxRuns = maxRuns
        self.reset()


    def start(self, ignoreIfRunning = True):
        if ((not self.running) or (not ignoreIfRunning)):
            if self.running:
                self.stop()
            self._start = time.time()
            self.running = True


    def stop(self):
        t = time.time()
        if self.running:
            self.measurements.append(t - self._start)
            if (self._maxRuns > 0) and (len(self.measurements) > self._maxRuns):
                del self.measurements[0]
            self.running = False


    def reset(self):
        self.measurements = []
        self.running = False


    @property
    def maxRuns(self):
        """Maximum number of captured spaces of time.
        
        When the limit is reached, the oldest measure will be removed.
        A value equal to 0 means unlimited measures.
        """
        return self._maxRuns


    @maxRuns.setter
    def maxRuns(self, value):
        if (value < 0):
            value = 0
        self._maxRuns = value
        if (self._maxRuns > 0) and (self._maxRuns > len(self.measurements)):
            del self.measurements[0:(len(self.measurements) - self._maxRuns)]


    @property
    def runs(self):
        """Number of captured spaces of time"""
        return len(self.measurements)


    @property
    def total(self):
        """Total time in seconds"""
        return math.fsum(self.measurements)


    @property
    def avg(self):
        """Average time per run in seconds"""
        return (self.total / self.runs) if (self.runs > 0) else 0


    @property
    def rate(self):
        """Average number of runs per second (i. e. the inverse of avg)"""
        return (self.runs / self.total) if (self.total > 0) else 0
