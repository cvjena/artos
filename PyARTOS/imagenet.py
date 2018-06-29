"""ImageNet interface"""


from . import artos_wrapper, utils
from .artos_wrapper import libartos
try:
    from PIL import Image
except:
    import Image
from glob import glob
import ctypes, os.path, re, tarfile


class ImageRepository(object):
    """Provides access to the synsets and images of an image repository."""
    
    def __init__(self, repoDirectory):
        """Initializes a new ImageRepository instance.
        
        repoDirectory - Path to the root directory of the image repository.
        """
        
        object.__init__(self)
        self._repoDirectory = repoDirectory
        self._numSynsets = None
    
    
    @property
    def repoDirectory(self):
        """Path to the root directory of the image repository."""
        
        return self._repoDirectory
    
    
    @property
    def numSynsets(self):
        """Number of synsets listed in the synset list file."""
        
        if (libartos is None):
            raise RuntimeError('Can not find libartos')
        if self._numSynsets is None:
            num = ctypes.c_uint(0)
            libartos.list_synsets(utils.str2bytes(self._repoDirectory), None, num)
            self._numSynsets = num.value
        return self._numSynsets
    
    
    def listSynsets(self):
        """Lists all synsets in this repository.
        
        Returns a list of all synsets in the image repository, represented by tuples
        with 2 components: The ID and the description of the synset.
        """
        
        if self.numSynsets > 0:
            buf_size = ctypes.c_uint(self.numSynsets)
            buf = (artos_wrapper.SynsetSearchResult * buf_size.value)()
            libartos.list_synsets(utils.str2bytes(self._repoDirectory), buf, buf_size)
            return [(utils.bytes2str(buf[i].synsetId), utils.bytes2str(buf[i].description)) for i in range(buf_size.value)]
        else:
            return []
    
    
    def searchSynsets(self, phrase, limit = 10):
        """Searches for synsets in this repository by a given search phrase.
        
        Returns a list of synsets matching phrase, represented by tuples with 3 components:
        The ID and the description of the synset and a score rating the precision of this search result.
        The list is sorted descending by score.
        limit defines the maximum number of returned synsets,
        """
        
        if (libartos is None):
            raise RuntimeError('Can not find libartos')
        buf_size = ctypes.c_uint(limit)
        buf = (artos_wrapper.SynsetSearchResult * buf_size.value)()
        libartos.search_synsets(utils.str2bytes(self._repoDirectory), utils.str2bytes(phrase), buf, buf_size)
        return [(utils.bytes2str(buf[i].synsetId), utils.bytes2str(buf[i].description), buf[i].score) for i in range(buf_size.value)]


    def getImagesFromSynset(self, synsetId, num):
        """Returns the first images of a given synset.
        
        synsetId - The ID of the synset.
        num - The maximum number of images to extract.
        Returns: The extracted images as a list of PIL.Image.Image instances.
        
        If the synset can not be found, an exception is thrown.
        """
        
        images = []
        repoType = self.__class__.type()
        
        if repoType == 'ImageNet':
            
            with tarfile.open(name = os.path.join(self._repoDirectory, 'Images', synsetId + '.tar'), mode = 'r') as tar:
                extRE = re.compile(r'\.jpe?g$', re.I)
                while (len(images) < num):
                    info = tar.next()
                    if info is None:
                        break
                    if info.isfile() and extRE.search(info.name):
                        images.append(Image.open(tar.extractfile(info)))
                        images[-1].load() # force reading of the whole image from the archive
            
        elif repoType == 'ImageDirectories':
            
            files = [os.path.join(dir, fn) \
                     for dir, subdirs, files in os.walk(os.path.join(self._repoDirectory, synsetId)) \
                     for fn in files \
                     if fn.lower().endswith('.jpg') or fn.lower().endswith('.jpeg')]
            for fn in files[:num]:
                images.append(Image.open(fn))
        
        return images
    
    
    @staticmethod
    def hasRepositoryStructure(dir):
        """Checks if a given directory looks like an image repository.
        
        Returns a tuple with 2 values, the first one being a boolean value specifying if the given directory
        could be an image repository or not and if not, the second one is an error message telling what
        exactly is wrong with it.
        """
        
        if (libartos is None):
            return False, 'Can not find libartos'
        else:
            errMsg = ctypes.c_char_p()
            res = libartos.check_repository_directory(utils.str2bytes(dir), ctypes.pointer(errMsg))
            return res, utils.bytes2str(errMsg.value)
    
    
    @staticmethod
    def type():
        """Returns the type identifier of the image repository driver built into ARTOS.
        
        Sometimes one may want to detect which type of image repository is being used by ARTOS,
        i.e. which driver has been compiled in. This function returns the type identifier of the
        image repository driver, which in the case of ImageNet repositories is "ImageNet" and
        "ImageDirectories" in the case of plain directory repositories.
        """
        
        if (libartos is None):
            raise RuntimeError('Can not find libartos')
        
        return utils.bytes2str(libartos.get_image_repository_type())