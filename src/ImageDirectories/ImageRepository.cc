#include "ImageRepository.h"
#include <algorithm>
#include "libartos_def.h"
#include "strutils.h"
#include "sysutils.h"
using namespace ARTOS;
using namespace std;


struct SynsetSearchResult
{

    Synset synset; /**< The synset itself. */
    float score; /**< Score of this search result. */

    SynsetSearchResult(Synset aSynset, float aScore): synset(aSynset), score(aScore) { };
    
    /**
    * Used for sorting search results descending by score.
    */
    bool operator<(const SynsetSearchResult & other) const
    {
        return (other.score < this->score);
    };

};


/**
* Compares the descriptions of two synsets and returns a similarity score.
*
* @param[in] words1 Description of the first synset split up into separate words.
*
* @param[in] words2 Description of the second synset split up into separate words.
*
* @return Similarity score (higher is more similar, 0 means no similarity).
*/
static float cmpSynsetDescriptions(const vector<string> & words1, const vector<string> & words2)
{
    // Count words appearing in both strings and use them as score
    float score = 0;
    for (vector<string>::const_iterator w1 = words1.begin(); w1 != words1.end(); w1++)
        for (vector<string>::const_iterator w2 = words2.begin(); w2 != words2.end(); w2++)
            if (*w1 == *w2)
            {
                score += 1;
                break;
            }
    return score;
}


ImageRepository::ImageRepository(const string & repoDirectory)
: m_dir(repoDirectory), m_numSynsets(0)
{ }


ImageRepository::ImageRepository(const ImageRepository & other)
: m_dir(other.m_dir), m_numSynsets(other.m_numSynsets)
{ }


string ImageRepository::getRepoDirectory() const
{
    return this->m_dir;
}


size_t ImageRepository::getNumSynsets() const
{
    if (this->m_numSynsets == 0)
        this->listSynsets(NULL, NULL);
    return this->m_numSynsets;
}


void ImageRepository::listSynsets(vector<string> * ids, vector<string> * descriptions) const
{
    vector<string> directories;
    scandir(this->m_dir, directories, ftDirectory);
    this->m_numSynsets = directories.size();
    if (ids != NULL)
        *ids = directories;
    if (descriptions != NULL)
        *descriptions = directories;
}


void ImageRepository::searchSynsets(const string & phrase, vector<Synset> & results,
                                    const size_t limit, vector<float> * scores) const
{
    // Split search phrase up into single words
    vector<string> phraseWords, descrWords;
    splitString(strtolower(phrase), " .,;_-", phraseWords);
    
    // Iterate over synsets, compare them with search phrase and add matches to temporary results vector
    Synset synset;
    float score;
    vector<SynsetSearchResult> _results;
    for (SynsetIterator synsetIt = this->getSynsetIterator(); synsetIt.ready(); ++synsetIt)
    {
        synset = *synsetIt;
        descrWords.clear();
        splitString(strtolower(synset.description), " .,;_-", descrWords);
        score = cmpSynsetDescriptions(phraseWords, descrWords);
        if (score > 0)
            _results.push_back(SynsetSearchResult(synset, score));
    }
    sort(_results.begin(), _results.end()); // sort descending by score
    
    // Copy sorted results to output argument
    size_t numResults = min(_results.size(), limit);
    results.resize(numResults);
    if (scores != NULL)
        scores->resize(numResults);
    for (size_t i = 0; i < numResults; i++)
    {
        results[i] = _results[i].synset;
        if (scores != NULL)
            (*scores)[i] = _results[i].score;
    }
}


SynsetIterator ImageRepository::getSynsetIterator() const
{
    return SynsetIterator(this->m_dir);
}


Synset ImageRepository::getSynset(const string & synsetId) const
{
    return (is_dir(join_path(2, this->m_dir.c_str(), synsetId.c_str()))) ? Synset(this->m_dir, synsetId, synsetId) : Synset();
}


MixedImageIterator ImageRepository::getMixedIterator(const unsigned int perSynset) const
{
    return MixedImageIterator(this->m_dir, perSynset);
}


bool ImageRepository::hasRepositoryStructure(const string & directory, const char ** errMsg)
{
    if (errMsg)
        *errMsg = "";
    
    if (!is_dir(directory))
    {
        if (errMsg)
            *errMsg = "The specified directory could not be found.";
        return false;
    }
    
    return true;
}


const char * ImageRepository::type()
{
    return "ImageDirectories";
}
