#ifndef ARTOS_IMAGEREPOSITORY_H
#define ARTOS_IMAGEREPOSITORY_H

#include <string>
#include <vector>

#include "Synset.h"
#include "SynsetIterators.h"

namespace ARTOS
{

/**
* Provides access to synsets and images in an image repository (composed of image and annotation
* files in simple directories in this case).
* This class also serves as a factory for Synset instances.
*
* The images of each synset have to be stored in a plain (i.e. not nested) folder along with the
* respective annotation files for the images.
* Those synset folders have to be located on disk in one single directory, which will be referred
* to as "repository directory" or "repoDirectory":
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ImageRepository
{

public:

    /**
    * @param[in] repoDirectory The path to the repository directory.
    */
    ImageRepository(const std::string & repoDirectory);
    
    /**
    * Copy constructor.
    */
    ImageRepository(const ImageRepository & other);
    
    /**
    * @return Returns the path to the repository directory.
    */
    std::string getRepoDirectory() const;
    
    /**
    * Returns the number of synsets in the synset list file.
    *
    * @note This value is cached after the first call to getNumSynsets() or listSynsets().
    */
    size_t getNumSynsets() const;
    
    /**
    * Lists all synsets in this repository.
    * 
    * @param[out] ids Pointer to a string vector that will receive the IDs (i.e. directory names) of the synsets.
    *                 Existing vector elements will be removed. May be NULL if not used.
    * 
    * @param[out] descriptions Pointer to a string vector that will receive the descriptions of the synsets.
    *                          In this implementation, the description of a synset will be equal to it's ID.
    *                          Existing vector elements will be removed. May be NULL if not used.
    */
    void listSynsets(std::vector<std::string> * ids, std::vector<std::string> * descriptions) const;
    
    /**
    * Searches for synsets whose description is similar to the words in a given search phrase.
    *
    * @param[in] phrase A space-separated list of words to search for.
    *
    * @param[out] results Vector that will be filled with the search results, having the best match
    *                     at the front of the vector and the match with the lowest score at the back.
    *
    * @param[in] limit Maximum number of search results.
    *
    * @param[out] scores Pointer to a float vector that optionally receives the scores of the
    *                    search results (greater is better). May be NULL if not used.
    */
    void searchSynsets(const std::string & phrase, std::vector<Synset> & results, 
                       const size_t limit = 10, std::vector<float> * scores = NULL) const;
    
    /**
    * Initializes an iterator over the synsets in this repository.
    *
    * @return Returns the new SynsetIterator instance.
    */
    SynsetIterator getSynsetIterator() const;
    
    /**
    * Initializes a Synset instance for a given synset ID.
    *
    * @return Returns the new Synset instance. If there is no synset with that ID, the `id` field of the
    *         returned object will be empty.
    */
    Synset getSynset(const std::string & synsetId) const;
    
    /**
    * Initializes a MixedImageIterator to iterate over images from diverse synsets in this repository.
    * 
    * @param[in] perSynset Maximum number of images taken from the same synset.
    * 
    * @return Returns the new MixedImageIterator instance.
    */
    MixedImageIterator getMixedIterator(const unsigned int perSynset = 1) const;


    /**
    * Checks if a given directory is structured like an image repository.
    *
    * In this implementation, this function just checks if the given path actually
    * is a directory.
    *
    * @param[in] directory Path to the directory to be checked.
    *
    * @param[out] errMsg If the given directory is not a valid image repository, this pointer will be set to
    * a pointer to a NUL-terminated string containing an adequate error message.
    * If everything is fine, the pointer will be set to an empty string.
    *
    * @return Returns false if the given directory is not an image repository directory.
    *         True is returned, if it *may* be one.
    */
    static bool hasRepositoryStructure(const std::string & directory, const char ** errMsg = NULL);
    
    
    /**
    * Sometimes one may want to detect which type of image repository is being used by ARTOS, i.e. which driver
    * has been compiled in. This function returns the type identifier of the image repository driver, which
    * in the case of plain directory repositories is "ImageDirectories".
    *
    * @return Returns a pointer to a NUL-terminated string specifying the type of this image repository driver.
    */
    static const char * type();


protected:

    std::string m_dir; /**< Path to the repository directory. */
    mutable size_t m_numSynsets; /**< Number of synsets cached after the first call to listSynsets(). */

};

}

#endif