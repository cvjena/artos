#include "SynsetIterators.h"
#include <cstdlib>
#include <cstdio>
#include "Synset.h"
#include "ImageRepository.h"
#include "libartos_def.h"
#include "strutils.h"
#include "sysutils.h"
using namespace ARTOS;
using namespace std;

SynsetIterator::SynsetIterator(const std::string & aRepoDirectory)
: m_repoDir(aRepoDirectory), m_pos(0)
{
    ImageRepository(this->m_repoDir).listSynsets(&this->m_ids, NULL);
}

SynsetIterator::SynsetIterator(const SynsetIterator & other)
: m_repoDir(other.m_repoDir), m_pos(0), m_ids(other.m_ids)
{ }

SynsetIterator & SynsetIterator::operator++()
{
    if (this->ready())
        this->m_pos++;
    return *this;
}

Synset SynsetIterator::operator*() const
{
    if (this->ready())
        return Synset(this->m_repoDir, this->m_ids[this->m_pos], this->m_ids[this->m_pos]);
    else
        return Synset();
}



void ImageIterator::listImagesInSynset(vector<string> & filenames, const string & synsetDirectory, bool bboxMode)
{
    vector<string> files, subdirs, newSubdirs;
    subdirs.push_back(".");
    string curSubdir, curDir;
    
    filenames.clear();
    while (!subdirs.empty())
    {
        curSubdir = subdirs.back();
        curDir = join_path(2, synsetDirectory.c_str(), curSubdir.c_str());
        subdirs.pop_back();
        
        if (bboxMode)
        {
            scandir(curDir, files, ftFile, "xml");
            string fn, path;
            for (vector<string>::const_iterator xmlFile = files.begin(); xmlFile != files.end(); xmlFile++)
            {
                fn = strip_file_extension(*xmlFile);
                path = join_path(2, curDir.c_str(), fn.c_str());
                if (is_file(path + ".jpg") || is_file(path + ".jpeg") || is_file(path + ".JPG") || is_file(path + ".JPEG"))
                    filenames.push_back(join_path(2, curSubdir.c_str(), fn.c_str()));
            }
        }
        else
        {
            scandir(curDir, files, ftFile, "jpg");
            scandir(curDir, files, ftFile, "jpeg");
            for (vector<string>::const_iterator fn = files.begin(); fn != files.end(); fn++)
                filenames.push_back(join_path(2, curSubdir.c_str(), strip_file_extension(*fn).c_str()));
        }
        files.clear();
        
        scandir(curDir, newSubdirs, ftDirectory);
        for (vector<string>::const_iterator sd = newSubdirs.begin(); sd != newSubdirs.end(); sd++)
            subdirs.push_back(join_path(2, curSubdir.c_str(), sd->c_str()));
        newSubdirs.clear();
    }
}



SynsetImageIterator::SynsetImageIterator(const string & aRepoDirectory, const string & aSynsetId, const bool & bboxRequired)
: ImageIterator(aRepoDirectory), m_synsetId(aSynsetId), m_bboxMode(bboxRequired)
{
    this->listImagesInSynset(this->m_filenames, join_path(2, this->m_repoDir.c_str(), this->m_synsetId.c_str()), this->m_bboxMode);
}

SynsetImageIterator::SynsetImageIterator(const SynsetImageIterator & other)
: ImageIterator(other.m_repoDir), m_synsetId(other.m_synsetId), m_bboxMode(other.m_bboxMode), m_filenames(other.m_filenames)
{ }

SynsetImageIterator & SynsetImageIterator::operator++()
{
    if (this->ready())
        this->m_pos++;
    return *this;
}

SynsetImage SynsetImageIterator::operator*()
{
    return (this->ready()) ? SynsetImage(this->m_repoDir, this->m_synsetId, this->m_filenames[this->m_pos]) : SynsetImage();
}



MixedImageIterator::MixedImageIterator(const std::string & aRepoDirectory, const unsigned int & aPerSynset)
: ImageIterator(aRepoDirectory), m_currentSynset(0), m_posCurrent(0), m_perSynset(aPerSynset), m_run(0), m_foundAny(false)
{
    this->init();
}

MixedImageIterator::MixedImageIterator(const MixedImageIterator & other)
: ImageIterator(other.m_repoDir), m_currentSynset(0), m_posCurrent(0), m_perSynset(other.m_perSynset), m_run(0), m_foundAny(false)
{
    this->init();
}

void MixedImageIterator::init()
{
    ImageRepository(this->m_repoDir).listSynsets(&this->m_synsets, NULL);
    if (this->m_perSynset == 0)
        this->m_perSynset = 1;
    // List files in first synset
    if (this->m_synsets.size() > 0)
    {
        this->listImagesInSynset(this->m_filenames, join_path(2, this->m_repoDir.c_str(), this->m_synsets[this->m_currentSynset].c_str()));
        if (!this->m_filenames.empty())
            this->m_foundAny = true;
        else
            this->nextSynset();
    }
}

void MixedImageIterator::nextSynset()
{
    this->m_currentSynset++;
    if (this->m_currentSynset >= this->m_synsets.size())
    {
        this->m_currentSynset = 0;
        this->m_run++;
        if (!this->m_foundAny)
        {
            // There are no synsets. Stop here for not entering an endless loop.
            return;
        }
    }
    
    this->listImagesInSynset(this->m_filenames, join_path(2, this->m_repoDir.c_str(), this->m_synsets[this->m_currentSynset].c_str()));
    this->m_posCurrent = 0;
    if (!this->m_filenames.empty())
        this->m_foundAny = true;
    else
        this->nextSynset();
}

MixedImageIterator & MixedImageIterator::operator++()
{
    if (this->ready())
    {
        this->m_posCurrent++;
        // Move on to next synset if number of images per synset reached
        if (this->m_posCurrent >= this->m_perSynset || this->m_posCurrent >= this->m_filenames.size())
            this->nextSynset();
    }
    return *this;
}

SynsetImage MixedImageIterator::operator*()
{
    if (this->ready())
    {
        return SynsetImage(
            this->m_repoDir,
            this->m_synsets[this->m_currentSynset],
            this->m_filenames[(this->m_run * this->m_perSynset + this->m_posCurrent) % this->m_filenames.size()]
        );
    }
    else
        return SynsetImage();
}

string MixedImageIterator::extract(const string & outDirectory)
{
    if (this->ready())
    {
        SynsetImage simg = **this;
        FFLD::JPEGImage img = simg.getImage();
        if (!img.empty())
        {
            string resultFileName = join_path(2, outDirectory.c_str(), (simg.getFilename() + ".jpg").c_str());
            img.save(resultFileName);
            return resultFileName;
        }
    }
    return "";
}

void MixedImageIterator::rewind()
{
    this->m_currentSynset = 0;
    this->m_posCurrent = 0;
    this->m_run = 0;
    this->init();
}
