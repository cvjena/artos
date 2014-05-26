/**
* @file
* Simple utilities for extracting files from (uncompressed) tar archives.
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/

#ifndef ARTOS_TAREXTRACTOR_H
#define ARTOS_TAREXTRACTOR_H

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <stdint.h>

namespace ARTOS
{


/**
* ASCII file header in tar archives.
*/
typedef struct __attribute__((__packed__)) {

    char filename[100];   /**< null-terminated path name. */
    char filemode[8];     /**< File mode, encoded as octal number string. */
    char owner_id[8];     /**< Numeric ID of the owner of the file, encoded as octal number string. */
    char group_id[8];     /**< Numeric ID of the user group of the file, encoded as octal number string. */
    char filesize[12];    /**< File size in bytes, encoded as octal number string. */
    char mtime[12];       /**< Last modification time as UNIX timestamp, encoded as octal number string. */
    char checksum[8];
    
    /**
    * Flag indicating the type of the record. Possible values:
    * '0'/NUL: normal file
    * '1': hard link
    * '2': symbolic link
    * '3': character special
    * '4': block special
    * '5': directory
    * '6': FIFO
    * '7': contiguous file
    */
    char type_flag;
    char linked_file[100]; /**< Name of the linked file. */
    
    // All following fields are only available in UStar format.
    char ustar_indicator[6]; /**< Contains "ustar" if this format is used. */
    char ustar_version[2];
    char owner_name[32];     /**< User name of the owner of the file as null-terminated string. */
    char group_name[32];     /**< Name of the group of the file as null-terminated string. */
    char device_major[8];
    char device_minor[8];
    
    /**
    * Prefix of filename if it is too long to fit in the first field.
    * Split at any '/' character, that may have to be inserted between filename_prefix and filename.
    */
    char filename_prefix[155];
    
    char reserved[12];


    bool isUStar()
    {
        return (memcmp("ustar", ustar_indicator, 5) == 0);
    }

} TarFileHeader;


typedef enum {
    tft_file, tft_hard_link, tft_symbolic_link, tft_character_special, tft_block_special, tft_directory, tft_fifo, tft_contiguous, tft_unknown
} TarFileType;


/**
* Information about a file in a tar archive.
*/
typedef struct {
    std::string filename; /**< Name of the file */
    uint64_t filesize;    /**< Size of the file in bytes */
    uint64_t mtime;       /**< Last modification time as UNIX timestamp */
    TarFileType type;     /**< The type of the file record */
    unsigned int index;   /**< Index of the file in the Tar archive. */
    std::ifstream::pos_type offset; /**< Position of the file (without the header) in the underlying file stream */
} TarFileInfo;


/**
* Lists and extracts files from an uncompressed tar archive.
* Extraction can be done to file or directly into memory.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class TarExtractor
{

public:

    /**
    * Size of a tar file header (512 bytes).
    */
    static const int headerSize = sizeof(TarFileHeader);
    
    /**
    * Flag that tells findFile() to ignore the file extensions.
    */
    static const unsigned int IGNORE_FILE_EXT = 1;


    /**
    * Creates a new TarExtractor which is not yet associated with any tar archive.
    */
    TarExtractor() : m_tarfile(), m_tarPath(""), m_fileIndex(0), m_eof(true) { };
    
    /**
    * Creates a new TarExtractor and opens a tar archive directly.
    *
    * Use isOpen() to check if the archive could be opened.
    *
    * @param[in] tarfilename Filename of the archive to open.
    */
    TarExtractor(const std::string & tarfilename)
    : m_tarfile(tarfilename.c_str(), std::ifstream::in | std::ifstream::binary), m_tarPath(tarfilename), m_fileIndex(0), m_eof(false) { };
    
    /**
    * Opens a tar archive. If another archive is already associated with this TarExtractor, it is closed.
    *
    * Use isOpen() to check if the archive could be opened.
    *
    * @param[in] tarfilename Filename of the archive to open.
    */
    void open(const std::string & tarfilename);
    
    /**
    * Closes the associated tar archive.
    */
    void close();
    
    /**
    * Checks if the tar archive could be opened and has not been closed yet.
    * 
    * @return True if the archive could be opened, otherwise false.
    * 
    * @note This does not mean that the associated file is a valid tar archive. It is just readable.
    */
    bool isOpen() const;
    
    /**
    * Propagates the state of the underlying file stream.
    *
    * @return True if the last operation has not failed, otherwise false.
    */
    bool good() const;

    /**
    * @return Returns the path to the currently opened Tar archive. The resulting string may be empty, if no archive is opened.
    */
    std::string getTarPath() const { return this->m_tarPath; };
    
    /**
    * @return Returns the index of the file at the current position in the tar archive.
    */
    unsigned int tellIndex() const { return this->m_fileIndex; };
    
    /**
    * Lists all files (and/or directories, links etc.) in the tar archive and stores information
    * about them in a vector of TarFileInfo structures.
    *
    * @param[out] fileinfo Vector which receives information about the records in the archive.
    *
    * @param[in] filterType If different from `tft_unknown`, only records of that type will be returned.
    */
    void listFiles(std::vector<TarFileInfo> & fileinfo, const TarFileType filterType = tft_unknown);
    
    /**
    * Searches for information about a file or directory with a given name in the tar archive.
    *
    * This operation is faster for multiple search requests than iterating over all files in the archive
    * using nextFile() and readHeader(), since it does this only once and uses an std::map as cache.
    *
    * The file position indicator won't be moved by this operation.
    *
    * @param[in] filename The name of the file to search for.
    *
    * @param[in] flags Optionally, a bit-wise combination of one of the following flags:
    *                    - IGNORE_FILE_EXT - If this is set, the file extension will be irrelevant for searching.
    *                                        In this case, only files, but not directories, may be returned.
    *
    * @return TarFileInfo structure with information about the requested file or directory.
    *         If no file matching `filename` could be found, the `type` member of the structure will be `tft_unknown`.
    */
    TarFileInfo findFile(std::string filename, const unsigned int flags = 0);
    
    /**
    * Reads the file header at the current position in the tar archive.
    * The file position indicator is not moved.
    *
    * @return Returns information about the file at the current position as TarFileInfo structure.
    */
    TarFileInfo readHeader();
    
    /**
    * Reads the file header of a specific file in the tar archive.
    * The file position indicator is moved to the specified file.
    *
    * @param[in] fileIndex The index of the file whose header is to be read.
    *
    * @return Returns information about the file as TarFileInfo structure.
    */
    TarFileInfo readHeader(const unsigned int fileIndex);
    
    /**
    * Extracts the file at the current position in the tar archive to disk.
    * The file position indicator is not moved.
    *
    * @param[in] outFilename Output filename.
    */
    void extract(const std::string & outFilename);
    
    /**
    * Extracts a specific file in the tar archive to disk.
    * The file position indicator is moved to the specified file.
    *
    * @param[in] fileIndex The index of the file to extract.
    *
    * @param[in] outFilename Output filename.
    */
    void extract(const unsigned int fileIndex, const std::string & outFilename);
    
    /**
    * Extracts the file at the current position in the tar archive directly into memory.
    * The file position indicator is not moved.
    *
    * @param[out] bufsize Will be set to the size (in bytes) of the allocated memory.
    *
    * @return Returns a pointer to the memory location where the file data has been extracted to.
    *
    * @note The memory for the file data is allocated by this function via `malloc()`.
    *       You have to free it yourself via `free()`.
    */
    char * extract(uint64_t & bufsize);
    
    /**
    * Extracts a specific file in the tar archive directly into memory.
    * The file position indicator is not moved.
    *
    * @param[in] fileIndex The index of the file to extract.
    *
    * @param[out] bufsize Will be set to the size (in bytes) of the allocated memory.
    *
    * @return Returns a pointer to the memory location where the file data has been extracted to.
    *
    * @note The memory for the file data is allocated by this function via `malloc()`.
    *       You have to free it yourself via `free()`.
    */
    char * extract(const unsigned int fileIndex, uint64_t & bufsize);
    
    /**
    * Moves the file position indicator to the beginning of the next file header.
    *
    * @return If the end of the tar file is reached or another input error occurs,
    * false is returned, otherwise true.
    */
    bool nextFile();
    
    /**
    * Moves the file position indicator to the beginning of a specific file header.
    *
    * Since tar archives don't provide random access, this operation must rewind the file first
    * and then move from file header to file header sequentially.
    *
    * @param[in] fileIndex The index of the file to place the file position indicator before.
    *
    * @return False if the given index is out of range or an input error occurs, otherwise true.
    */
    bool seekFile(const unsigned int fileIndex);
    
    /**
    * Moves the file position indicator to the beginning of the tar archive and clears all
    * error flags of the underlying file stream.
    */
    void rewind();


    /**
    * Searches for information about a file or directory with a given name in a specific tar archive.
    *
    * This operation is faster for multiple search requests than iterating over all files in the archive
    * using nextFile() and readHeader(), since it does this only once and uses an std::map as cache, so that
    * the archive doesn't even need to be opened on further search requests.
    *
    * @param[in] tarfilename The path of the Tar archive.
    *
    * @param[in] filename The name of the file to search for.
    *
    * @param[in] flags Optionally, a bit-wise combination of one of the following flags:
    *                    - IGNORE_FILE_EXT - If this is set, the file extension will be irrelevant for searching.
    *                                        In this case, only files, but not directories, may be returned.
    *
    * @return TarFileInfo structure with information about the requested file or directory.
    *         If no file matching `filename` could be found, the `type` member of the structure will be `tft_unknown`.
    */
    static TarFileInfo findFileInArchive(const std::string & tarfilename, std::string filename, const unsigned int flags = 0);


protected:
    std::ifstream m_tarfile; /**< Tar file stream */
    std::string m_tarPath; /**< Path to the opened Tar archive (may be empty) */
    unsigned int m_fileIndex; /**< Index of the current file in the tar archive */
    bool m_eof; /**< Indicates if the end of the tar file has been reached, so that there is no more data to be read */
    
    /**
    * Caches information about each file and directory in the archive for use by findFile().
    * The file position indicator is not moved.
    *
    * @param[in] ignoreExt If set to true, the cache for filenames without file extension
    *                      will be built, otherwise the cache for files and directories including
    *                      the file extension.
    */
    void cacheFileInfo(const bool ignoreExt = false);

};

}

#endif
