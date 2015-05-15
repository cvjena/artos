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
        
        If the synset archive can not be found, an exception is thrown.
        """
        
        images = []
        with tarfile.open(name = os.path.join(self._repoDirectory, 'Images', synsetId + '.tar'), mode = 'r') as tar:
            extRE = re.compile(r'\.jpe?g$', re.I)
            while (len(images) < num):
                info = tar.next()
                if info is None:
                    break
                if info.isfile() and extRE.search(info.name):
                    images.append(Image.open(tar.extractfile(info)))
                    images[-1].load() # force reading of the whole image from the archive
        return images
    
    
    @staticmethod
    def hasRepositoryStructure(dir):
        """Checks if a given directory looks like an image repository.
        
        This function checks if the given directory contains the synset list file (synset_wordlist.txt)
        and the directories 'Images' and 'Annotation'. Additionally, the 'Images' directory must contain
        at least one tar file.
        
        Returns a tuple with 2 values, the first one being a boolean value specifying if the given directory
        could be an image repository or not and if not, the second one is an error message telling what
        exactly is wrong with it.
        """
        
        if not os.path.isdir(dir):
            return False, 'The specified directory could not be found.'
        elif not os.path.isfile(os.path.join(dir, 'synset_wordlist.txt')):
            return False, 'Could not find synset_wordlist.txt'
        elif not os.path.isdir(os.path.join(dir, 'Images')):
            return False, 'Could not find "Images" subdirectory.'
        elif not os.path.isdir(os.path.join(dir, 'Annotation')):
            return False, 'Could not find "Annotation" subdirectory.'
        elif len(glob(os.path.join(dir, 'Images', '*.tar'))) == 0:
            return False, 'Could not find any synset image archive.'
        else:
            return True, ''