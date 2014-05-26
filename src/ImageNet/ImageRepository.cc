#include "ImageRepository.h"
#include <fstream>
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


void ImageRepository::listSynsets(vector<string> * ids, vector<string> * descriptions) const
{
    if (ids != NULL)
        ids->clear();
    if (descriptions != NULL)
        descriptions->clear();
        
    ifstream listFile(join_path(2, this->m_dir.c_str(), "synset_wordlist.txt").c_str());
    if (listFile.is_open())
    {
        this->m_numSynsets = 0;
        string line;
        size_t pos; // position of ID and description delimiter
        while (listFile.good())
        {
            getline(listFile, line);
            line = trim(line);
            if (!line.empty())
            {
                this->m_numSynsets++;
                if (ids != NULL || descriptions != NULL)
                {
                    pos = line.find(' ');
                    if (ids != NULL)
                        ids->push_back(trim(line.substr(0, pos)));
                    if (descriptions != NULL && pos != string::npos)
                        descriptions->push_back(trim(line.substr(pos + 1)));
                }
            }
        }
    }
}


void ImageRepository::searchSynsets(const string & phrase, vector<Synset> & results,
                                    const size_t limit, vector<float> * scores) const
{
    // Split search phrase up into single words
    vector<string> phraseWords, descrWords;
    splitString(strtolower(phrase), " .,;", phraseWords);
    
    // Iterate over synsets, compare them with search phrase and add matches to temporary results vector
    Synset synset;
    float score;
    vector<SynsetSearchResult> _results;
    for (SynsetIterator synsetIt = this->getSynsetIterator(); synsetIt.ready(); ++synsetIt)
    {
        synset = *synsetIt;
        descrWords.clear();
        splitString(strtolower(synset.description), " .,;", descrWords);
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


Synset ImageRepository::getSynset(const string & synsetId) const
{
    Synset synset;
    for (SynsetIterator synsetIt = this->getSynsetIterator(); synsetIt.ready(); ++synsetIt)
    {
        synset = *synsetIt;
        if (synset.id == synsetId)
            return synset;
    }
    return Synset();
}


bool ImageRepository::hasRepositoryStructure(const string & directory)
{
    return (is_file(join_path(2, directory.c_str(), "synset_wordlist.txt").c_str())
            && is_dir(join_path(2, directory.c_str(), IMAGENET_IMAGE_DIR).c_str())
            && is_dir(join_path(2, directory.c_str(), IMAGENET_ANNOTATION_DIR).c_str()));
}
