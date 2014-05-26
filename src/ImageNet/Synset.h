#ifndef ARTOS_SYNSET_H
#define ARTOS_SYNSET_H

#include <string>
#include "SynsetIterators.h"

namespace ARTOS
{

/**
* Represents a synset.
* 
* @note Should be constructed using ImageRepository::getSynset().
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
struct Synset
{

    std::string id; /**< ID of the Synset (e. g. 'n02119789') */
    std::string description; /**< Words and phrases describing the synset (e. g. 'kit fox, Vulpes macrotis') */
    std::string repoDirectory; /**< Directory where the image repository, this synset belongs to, resides */
    
    Synset() : id(""), description(""), repoDirectory("") { };
    
    /**
    * @note Using ImageRepository::getSynset() is better than calling the constructor directly,
    *       since it will determine the description of the synset.
    * 
    * @param[in] aRepoDirectory Path to the the repository directory.
    *
    * @param[in] aId The ID of the synset.
    *
    * @param[in] aDescription Words and phrases describing the synset.
    */
    Synset(const std::string & aRepoDirectory, const std::string & aId, const std::string aDescription)
    : id(aId), description(aDescription), repoDirectory(aRepoDirectory) { };
    
    /**
    * Copy constructor.
    */
    Synset(const Synset & other) : id(other.id), description(other.description), repoDirectory(other.repoDirectory) { };
    
    /**
    * Initializes a SynsetImageIterator to iterate over the images in this synset.
    *
    * @param[in] bboxRequired If set to true, only images with bounding boxes available will
    *                         be taken into account (annotations archive will be read and iterated),
    *                         otherwise all images in the synset (image archive will be read and iterated).
    *
    * @return Returns the new SynsetImageIterator instance.
    */
    SynsetImageIterator getImageIterator(const bool & bboxRequired = false) const
    {
        return SynsetImageIterator(repoDirectory, id, bboxRequired);
    };

};

}

#endif