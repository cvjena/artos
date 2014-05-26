try:
    # Python 3
    from urllib.request import urlopen
except:
    # Python 2
    from urllib2 import urlopen

import re



wordlist_url = 'http://image-net.org/api/text/imagenet.bbox.obtain_synset_wordlist'


def fetchSynsetWordlist(targetFile = 'synset_wordlist.txt', fetchURL = wordlist_url, verbose = False):
    """Downloads and converts the list of ImageNet synsets which bounding box annotations are available for.
    
    targetFile - Name of the output file which the synset list will be written to, one synset per line
                 in the format '<synset-id> <description>'. For example: 'n02119789 kit fox, Vulpes macrotis'
    fetchURL - The URL to fetch the synset list from.
    verbose - If set to true, status messages will be printed out.
    
    Return: The number of synsets written to the resulting list file.
    
    Throws: IOError if output file could not be opened for writing, RuntimeError in one of the following cases:
            - HTTP request failed
            - Invalid format (conversion failed)
    """
    
    htmlRE  = re.compile('<a [^>]*href="[^"]+wnid=(n[0-9]+)"[^>]*>([^<]+)</a>', re.I)
    plainRE = re.compile(r'^\s*(n[0-9]+)\s+(.+?)\s*$', re.I)
    numSynsets = 0
    
    if verbose:
        print('Downloading {}...'.format(fetchURL))
    req = urlopen(fetchURL)
    if (req.getcode() == 200):
        try:
            outFile = open(targetFile, 'w')
            try:
                if verbose:
                    print('Parsing wordlist...')
                for line in req:
                    line = line.decode()
                    if line.strip() != '':
                        match = plainRE.match(line)
                        if not match:
                            match = htmlRE.search(line)
                        if not match:
                            raise RuntimeError('Format of downloaded synset wordlist not recognized.')
                        outFile.write('{sid} {descr}\n'.format(sid = match.group(1), descr = match.group(2)))
                        numSynsets += 1
            finally:
                outFile.close()
        finally:
            req.close()
    else:
        req.close()
        raise RuntimeError('HTTP request failed with status code {}.'.format(req.getcode()))
    if verbose:
        print('Fetched IDs and descriptions of {} synsets.'.format(numSynsets))
    return numSynsets



if (__name__ == '__main__'):
    import sys
    targetFile = sys.argv[1] if len(sys.argv) > 1 else 'synset_wordlist.txt'
    fetchSynsetWordlist(targetFile = targetFile, verbose = True)