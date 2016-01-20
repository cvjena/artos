#include "TarExtractor.h"
#include <cstdlib>
#include <map>
#include "sysutils.h"
using namespace ARTOS;
using namespace std;

typedef map<string, TarFileInfo> FileInfoMap;
typedef map<string, FileInfoMap> ArchiveFileInfoMap;
typedef map<unsigned int, ifstream::pos_type> FileOffsetMap;
typedef map<string, FileOffsetMap> ArchiveFileOffsetMap;

/**
* Map of caches (also maps) for file and directory information in Tar archives
* used by TarExtractor::findFile() and TarExtractor::findFileInArchive().
*/
static ArchiveFileInfoMap fileInfoCache;

/**
* Map of caches (also maps) for file information in Tar archives (ignoring file extensions)
* used by TarExtractor::findFile() and TarExtractor::findFileInArchive().
*/
static ArchiveFileInfoMap fileInfoCache_noExt;

/**
* Map of caches (also maps), which store the stream offset for file indices in Tar archives
* for faster use of the TarExtractor::seekFile() method without having to walk through the
* entire archive sequentially.
*/
static ArchiveFileOffsetMap fileOffsetCache;

/**
* Converts an octal number string from a Tar file header to an uint64_t.
* This special purpose function is necessary, since the octal strings in the Tar file headers
* may be not null-terminated, so we can't just use `strtoull()`.
*/
static uint64_t tar_octal_to_uint64(const char * tar_octal, size_t len)
{
    char num_str[len + 1];
    strncpy(num_str, tar_octal, len);
    num_str[len] = 0;
    return strtoull(num_str, NULL, 8);
}



void TarExtractor::open(const string & tarfilename)
{
    if (this->isOpen())
        this->close();
    this->m_tarfile.open(tarfilename.c_str(), ifstream::in | ifstream::binary);
    this->m_tarPath = tarfilename;
    this->m_fileIndex = 0;
    this->m_eof = false;
}

void TarExtractor::close()
{
    this->m_tarfile.close();
}

bool TarExtractor::isOpen() const
{
    return this->m_tarfile.is_open();
}

bool TarExtractor::good() const
{
    return (this->m_tarfile.good() && !this->m_eof);
}

void TarExtractor::listFiles(std::vector<TarFileInfo> & fileinfo, const TarFileType filterType)
{
    this->rewind();
    TarFileInfo info;
    do
    {
        info = readHeader();
        if (info.type != tft_unknown && (filterType == tft_unknown || filterType == info.type))
            fileinfo.push_back(info);
    }
    while (this->nextFile());
}

TarFileInfo TarExtractor::findFile(string filename, const unsigned int flags)
{
    if (flags & TarExtractor::IGNORE_FILE_EXT)
        filename = strip_file_extension(filename);
    
    // Search for cache
    ArchiveFileInfoMap * cache = (flags & TarExtractor::IGNORE_FILE_EXT) ? &fileInfoCache_noExt : &fileInfoCache;
    ArchiveFileInfoMap::const_iterator infoCache = cache->find(this->m_tarPath);
    if (infoCache == cache->end())
    {
        // Create cache if it does not exist
        this->cacheFileInfo(flags & TarExtractor::IGNORE_FILE_EXT);
        infoCache = cache->find(this->m_tarPath);
    }
    
    // Search in cache
    FileInfoMap::const_iterator fileInfo = infoCache->second.find(filename);
    if (fileInfo != infoCache->second.end())
        return fileInfo->second;
    else
    {
        TarFileInfo info;
        info.filename = "";
        info.type = tft_unknown;
        return info;
    }
}

TarFileInfo TarExtractor::readHeader()
{
    ifstream::pos_type startPos = this->m_tarfile.tellg(); // save current position to restore it later
    TarFileHeader header;
    TarFileInfo info;
    this->m_tarfile.read(reinterpret_cast<char*>(&header), headerSize);
    if (this->m_tarfile.good())
    {
        info.filename = header.filename;
        if (header.isUStar() && header.filename_prefix[0] != 0)
        {
            string prefix(header.filename_prefix);
            if (prefix[prefix.length() - 1] != '/' && prefix[prefix.length() - 1] != '\\')
                prefix += "/";
            info.filename = prefix + info.filename;
        }
        info.filesize = tar_octal_to_uint64(header.filesize, sizeof(header.filesize));
        info.mtime = tar_octal_to_uint64(header.mtime, sizeof(header.mtime));
        info.type = (header.type_flag >= '0' && header.type_flag <= '7') ? static_cast<TarFileType>(header.type_flag - '0') : tft_unknown;
        if (info.type == tft_file && (info.filename[info.filename.length() - 1] == '/' || info.filename[info.filename.length() - 1] == '\\'))
            info.type = tft_directory;
        info.index = this->tellIndex();
        info.offset = startPos + streamoff(512);
    }
    else
    {
        info.type = tft_unknown;
        this->m_eof = true;
    }
    this->m_tarfile.seekg(startPos); // restore file position
    return info;
}

TarFileInfo TarExtractor::readHeader(const unsigned int fileIndex)
{
    this->seekFile(fileIndex);
    return this->readHeader();
}

void TarExtractor::extract(const std::string & outFilename)
{
    ofstream outFile(outFilename.c_str(), ofstream::out | ofstream::binary | ofstream::trunc);
    if (outFile.is_open() && outFile.good())
    {
        TarFileInfo info = this->readHeader();
        if (info.type == tft_file)
        {
            ifstream::pos_type startPos = this->m_tarfile.tellg(); // save current position to restore it later
            this->m_tarfile.seekg(headerSize, ifstream::cur); // skip header
            
            // Read data from tar and write it to outFile in chunks of headerSize
            uint64_t bytesLeft = info.filesize;
            uint64_t chunkSize;
            char * buf = new char[headerSize];
            while (bytesLeft > 0 && this->m_tarfile.good() && outFile.good())
            {
                chunkSize = (bytesLeft >= headerSize) ? headerSize : bytesLeft;
                this->m_tarfile.read(buf, chunkSize);
                outFile.write(buf, chunkSize);
                bytesLeft -= chunkSize;
            }
            delete[] buf;
            
            this->m_tarfile.seekg(startPos); // restore file position
        }
    }
}

void TarExtractor::extract(const unsigned int fileIndex, const std::string & outFilename)
{
    this->seekFile(fileIndex);
    this->extract(outFilename);
}

char * TarExtractor::extract(uint64_t & bufsize)
{
    TarFileInfo info = this->readHeader();
    if (info.type == tft_file)
    {
        ifstream::pos_type startPos = this->m_tarfile.tellg(); // save current position to restore it later
        this->m_tarfile.seekg(headerSize, ifstream::cur); // skip header
        
        // Allocate buffer
        bufsize = info.filesize;
        char * buf = reinterpret_cast<char*>(malloc(bufsize));
        if (buf != NULL)
            this->m_tarfile.read(buf, bufsize); // extract data
        
        this->m_tarfile.seekg(startPos); // restore file position
        return buf;
    }
    else
        return NULL;
}
    
char * TarExtractor::extract(const unsigned int fileIndex, uint64_t & bufsize)
{
    this->seekFile(fileIndex);
    return this->extract(bufsize);
}

bool TarExtractor::nextFile()
{
    TarFileInfo info = this->readHeader();
    if (!this->good())
        return false;
    uint64_t fsize_overhang = info.filesize % headerSize;
    uint64_t padded_filesize = (fsize_overhang == 0) ? info.filesize : info.filesize + 512 - fsize_overhang;
    this->m_tarfile.seekg(512 + padded_filesize, ifstream::cur);
    this->m_fileIndex++;
    fileOffsetCache[this->m_tarPath][this->m_fileIndex] = this->m_tarfile.tellg();
    return this->good();
}

bool TarExtractor::seekFile(const unsigned int fileIndex)
{
    // Search in cache
    FileOffsetMap & cache = fileOffsetCache[this->m_tarPath];
    FileOffsetMap::iterator cacheEntry = cache.find(fileIndex);
    if (cacheEntry != cache.end())
    {
        this->m_tarfile.clear();
        this->m_tarfile.seekg(cacheEntry->second);
        this->m_fileIndex = fileIndex;
        this->m_eof = false;
        return true;
    }
    else
    {
        // Search sequentially
        this->rewind();
        for (unsigned int i = 0; i < fileIndex; i++)
            if (!this->nextFile())
                return false;
        return true;
    }
}

void TarExtractor::rewind()
{
    this->m_tarfile.clear();
    this->m_tarfile.seekg(0);
    this->m_fileIndex = 0;
    this->m_eof = false;
}

void TarExtractor::cacheFileInfo(const bool ignoreExt)
{
    FileInfoMap & cache = (ignoreExt) ? fileInfoCache_noExt[this->m_tarPath] : fileInfoCache[this->m_tarPath];
    vector<TarFileInfo> fileInfo;
    unsigned int startPos = this->tellIndex();
    this->listFiles(fileInfo);
    this->seekFile(startPos);
    string filename;
    for (vector<TarFileInfo>::const_iterator it = fileInfo.begin(); it != fileInfo.end(); it++)
        if (it->type == tft_file || (!ignoreExt && it->type == tft_directory))
        {
            filename = (ignoreExt) ? strip_file_extension(it->filename) : it->filename;
            cache.insert(FileInfoMap::value_type(filename, *it));
        }
}


TarFileInfo TarExtractor::findFileInArchive(const string & tarfilename, string filename, const unsigned int flags)
{
    TarFileInfo info;
    info.filename = "";
    info.type = tft_unknown;
    
    if (flags & TarExtractor::IGNORE_FILE_EXT)
        filename = strip_file_extension(filename);
    
    // Search in cache
    ArchiveFileInfoMap * cache = (flags & TarExtractor::IGNORE_FILE_EXT) ? &fileInfoCache_noExt : &fileInfoCache;
    ArchiveFileInfoMap::const_iterator infoCache = cache->find(tarfilename);
    if (infoCache != cache->end())
    {
        FileInfoMap::const_iterator fileInfo = infoCache->second.find(filename);
        if (fileInfo != infoCache->second.end())
            info = fileInfo->second;
    }
    else
    {
        // Initially cache file information in that archive
        TarExtractor tar(tarfilename);
        if (tar.isOpen())
        {
            info = tar.findFile(filename, flags);
            tar.close();
        }
    }
    
    return info;
}
