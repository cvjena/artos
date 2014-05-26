#ifndef ARTOS_IMAGEREPOSITORY_H
#define ARTOS_IMAGEREPOSITORY_H

#include <string>
#include <vector>

#include "Synset.h"
#include "SynsetIterators.h"

namespace ARTOS
{

/**
* Provides access to synsets and images in an image repository (ImageNet in this case).
* This class also serves as a factory for Synset instances.
* 
* All images and bounding box annotations as well as a synset list file have to be stored on
* disk in a directory, which will be referred to as "repository directory" or "repoDirectory".
*
* The images of a synset are expected to be packed together in an uncompressed (!) Tar archive
* which is located at `<repoDirectory>/Images/<synsetId>.tar`.
* 
* The bounding box annotations (one XML file per image) of each synset are also expected to be
* packed together in an uncompressed (!) Tar archive which is located at
* `<repoDirectory>/Annotation/<synsetId>.tar`.
*
* Last, but not least, the synset list file should be located at `<repoDirectory>/synset_wordlist.txt`
* and contains one record per line consisting of the synsets id, followed by a space and a list of
* words or phrases describing the synset (e. g. "n02119789 kit fox, Vulpes macrotis").
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class ImageRepository
{

public:

    /**
    * @param[in] repoDirectory The path to the repository directory.
    */
    ImageRepository(const std::string & repoDirectory) : m_dir(repoDirectory), m_numSynsets(0) { };
    
    /**
    * Copy constructor.
    */
    ImageRepository(const ImageRepository & other) : m_dir(other.m_dir), m_numSynsets(other.m_numSynsets) { };
    
    /**
    * @return Returns the path to the repository directory.
    */
    std::string getRepoDirectory() const { return this->m_dir; };
    
    /**
    * Returns the number of synsets in the synset list file.
    *
    * @note This value is cached after the first call to getNumSynsets() or listSynsets().
    */
    size_t getNumSynsets() const
    {
        if (this->m_numSynsets == 0)
            this->listSynsets(NULL, NULL);
        return this->m_numSynsets;
    };
    
    /**
    * Lists all synsets in this repository as specified in the synset list file.
    * 
    * @param[out] ids Pointer to a string vector that will receive the IDs of the synsets.
    *                 Existing vector elements will be removed. May be NULL if not used.
    * 
    * @param[out] descriptions Pointer to a string vector that will receive the descriptions of the synsets,
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
    SynsetIterator getSynsetIterator() const { return SynsetIterator(this->m_dir); };
    
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
    MixedImageIterator getMixedIterator(const unsigned int perSynset = 1) const
    {
        return MixedImageIterator(this->m_dir, perSynset);
    };


    /**
    * Checks if a given directory is structured like an image repository. To be valid in terms of this function,
    * the directory must contain the file 'synset_wordlist.txt' and the directories 'Images' and 'Annotation'.
    *
    * @note This function does not check if the synset list file is valid or the image directory contains
    *       any synset archives at all. It is just a simple indicator, if a directory *could* contain an image
    *       repository.
    *
    * @param[in] directory Path to the directory to be checked.
    *
    * @return Returns false if the given directory is not an image repository directory.
    *         True is returned, if it *may* be one.
    */
    static bool hasRepositoryStructure(const std::string & directory);


protected:

    std::string m_dir; /**< Path to the repository directory. */
    mutable size_t m_numSynsets; /**< Number of synsets cached after the first call to listSynsets(). */

};

}

#endif