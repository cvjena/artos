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
: m_repoDir(aRepoDirectory), m_lastLine(""), m_pos(0)
{
    this->m_listFile.open(join_path(2, this->m_repoDir.c_str(), "synset_wordlist.txt").c_str());
    ++(*this); // read first line
}

SynsetIterator::SynsetIterator(const SynsetIterator & other)
: m_repoDir(other.m_repoDir), m_lastLine(""), m_pos(0)
{
    this->m_listFile.open(join_path(2, this->m_repoDir.c_str(), "synset_wordlist.txt").c_str());
    ++(*this); // read first line
}

SynsetIterator & SynsetIterator::operator++()
{
    if (this->ready())
    {
        // Skip empty lines
        do
        {
            getline(this->m_listFile, this->m_lastLine);
            this->m_lastLine = trim(this->m_lastLine);
        }
        while (this->m_lastLine.empty() && this->m_listFile.good());
        this->m_pos++;
    }
    else
        this->m_lastLine = "";
    return *this;
}

Synset SynsetIterator::operator*() const
{
    if (!this->m_lastLine.empty())
    {
        size_t pos = this->m_lastLine.find(' ');
        return Synset(this->m_repoDir, trim(this->m_lastLine.substr(0, pos)), trim(this->m_lastLine.substr(pos + 1)));
    }
    else
        return Synset();
}



SynsetImageIterator::SynsetImageIterator(const string & aRepoDirectory, const string & aSynsetId, const bool & bboxRequired)
: ImageIterator(aRepoDirectory), m_synsetId(aSynsetId), m_bboxMode(bboxRequired), m_lastFileName(""), m_lastFileIndex(0), m_lastFileOffset(0)
{
    string tarFilename = this->m_synsetId + ".tar";
    string subdir = (this->m_bboxMode) ? IMAGENET_ANNOTATION_DIR : IMAGENET_IMAGE_DIR;
    this->m_tar.open(join_path(3, this->m_repoDir.c_str(), subdir.c_str(), tarFilename.c_str()));
    ++(*this); // read first record
}

SynsetImageIterator::SynsetImageIterator(const SynsetImageIterator & other)
: ImageIterator(other.m_repoDir), m_synsetId(other.m_synsetId), m_bboxMode(other.m_bboxMode), m_lastFileName(""), m_lastFileIndex(0), m_lastFileOffset(0)
{
    string tarFilename = this->m_synsetId + ".tar";
    string subdir = (this->m_bboxMode) ? IMAGENET_ANNOTATION_DIR : IMAGENET_IMAGE_DIR;
    this->m_tar.open(join_path(3, this->m_repoDir.c_str(), subdir.c_str(), tarFilename.c_str()));
    ++(*this); // read first record
}

SynsetImageIterator & SynsetImageIterator::operator++()
{
    if (this->m_tar.isOpen() && this->m_tar.good())
    {
        TarFileInfo info;
        do
            info = this->m_tar.readHeader();
        while (this->m_tar.nextFile() && info.type != tft_file && this->m_tar.good());
        if (info.type == tft_file)
        {
            this->m_lastFileName = extract_basename(info.filename);
            this->m_lastFileIndex = info.index;
            this->m_lastFileOffset = info.offset;
        }
        else
            this->m_lastFileName = ""; // invalidate iterator
        this->m_pos++;
    }
    return *this;
}

SynsetImage SynsetImageIterator::operator*()
{
    if (!this->m_lastFileName.empty())
    {
        SynsetImage simg(this->m_repoDir, this->m_synsetId, this->m_lastFileName);
#ifndef NO_CACHE_POSITIVES
        if (!this->m_bboxMode)
        {
            // Try to read image from the stored offset in the tar archive
            simg.readImageFromFileOffset(this->m_tar.getTarPath(), this->m_lastFileOffset);
        }
        // Removed the else-block. It was intended to be faster that way, but for some reason it was slower instead.
        /*else
        {
            // Try to read bounding box data directly
            uint64_t bufsize;
            char * xmlData = this->m_tar.extract(this->m_lastFileIndex, bufsize);
            if (xmlData != NULL)
            {
                simg.loadBoundingBoxes(xmlData, bufsize);
                free(xmlData);
            }
        }*/
#endif
        return simg;
    }
    else
        return SynsetImage();
}

void SynsetImageIterator::rewind()
{
    this->m_tar.rewind();
    this->m_pos = 0;
    ++(*this);
}



MixedImageIterator::MixedImageIterator(const std::string & aRepoDirectory, const unsigned int & aPerSynset)
: ImageIterator(aRepoDirectory), m_currentSynset(0), m_lastSynset(""), m_lastFileName(""), m_lastFileOffset(0),
  m_lastFileIndex(0), m_perSynset(aPerSynset), m_posCurrent(0), m_run(0), m_foundAny(false)
{
    this->init();
}

MixedImageIterator::MixedImageIterator(const MixedImageIterator & other)
: ImageIterator(other.m_repoDir), m_currentSynset(0), m_lastSynset(""), m_lastFileName(""), m_lastFileOffset(0),
  m_lastFileIndex(0), m_perSynset(other.m_perSynset), m_posCurrent(0), m_run(0), m_foundAny(false)
{
    this->init();
}

void MixedImageIterator::init()
{
    ImageRepository(this->m_repoDir).listSynsets(&this->m_synsets, NULL);
    if (this->m_perSynset == 0)
        this->m_perSynset = 1;
    // Open Tar archive of first synset
    if (this->m_synsets.size() > 0)
    {
        string tarFilename = this->m_synsets[this->m_currentSynset] + ".tar";
        this->m_tar.open(join_path(3, this->m_repoDir.c_str(), IMAGENET_IMAGE_DIR, tarFilename.c_str()));
        if (this->m_tar.isOpen())
            this->m_foundAny = true;
        else
            this->nextSynset();
        ++(*this); // read first record
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
            // There are no archives. Stop here for not entering an endless loop.
            return;
        }
    }
    string tarFilename = this->m_synsets[this->m_currentSynset] + ".tar";
    this->m_tar.open(join_path(3, this->m_repoDir.c_str(), IMAGENET_IMAGE_DIR, tarFilename.c_str()));
    if (this->m_tar.isOpen())
    {
        this->m_foundAny = true;
        if (!this->m_tar.seekFile(this->m_run * this->m_perSynset))
            this->m_tar.rewind();
        this->m_posCurrent = 0;
    }
    else
        this->nextSynset();
}

MixedImageIterator & MixedImageIterator::operator++()
{
    if (this->ready())
    {
        // Move on to next synset of number of images per synset reached
        if (this->m_posCurrent >= this->m_perSynset)
            this->nextSynset();
        
        // Get image from current synset
        TarFileInfo info;
        do
            info = this->m_tar.readHeader();
        while (this->m_tar.nextFile() && info.type != tft_file && this->m_tar.good());
        if (info.type == tft_file)
        {
            this->m_lastSynset = this->m_synsets[this->m_currentSynset];
            this->m_lastFileName = extract_basename(info.filename);
            this->m_lastFileOffset = info.offset;
            this->m_lastFileIndex = this->m_tar.tellIndex();
            this->m_posCurrent++;
            this->m_pos++;
        }
        else
        {
            // End of this synset reached - try next
            this->nextSynset();
            ++(*this);
        }
    }
    return *this;
}

SynsetImage MixedImageIterator::operator*()
{
    if (!this->m_lastFileName.empty())
    {
        SynsetImage simg(this->m_repoDir, this->m_lastSynset, this->m_lastFileName);
#ifndef NO_CACHE_POSITIVES
        // Try to read image from the stored offset in the tar archive
        string tarFilename = this->m_lastSynset + ".tar";
        simg.readImageFromFileOffset(join_path(3, this->m_repoDir.c_str(), IMAGENET_IMAGE_DIR, tarFilename.c_str()), this->m_lastFileOffset);
#endif
        return simg;
    }
    else
        return SynsetImage();
}

string MixedImageIterator::extract(const string & outDirectory)
{
    if (!this->m_lastFileName.empty())
    {
        string resultFileName = join_path(2, outDirectory.c_str(), this->m_lastFileName.c_str());
        if (this->m_posCurrent > 0)
        {
            // Tar file still loaded
            unsigned int curPos = this->m_tar.tellIndex();
            this->m_tar.extract(this->m_lastFileIndex, resultFileName);
            this->m_tar.seekFile(curPos);
        }
        else
        {
            string tarFilename = this->m_lastSynset + ".tar";
            TarExtractor tar(join_path(3, this->m_repoDir.c_str(), IMAGENET_IMAGE_DIR, tarFilename.c_str()));
            tar.extract(this->m_lastFileIndex, resultFileName);
        }
        return resultFileName;
    }
    else
        return "";
}

void MixedImageIterator::rewind()
{
    this->m_currentSynset = 0;
    this->m_pos = 0;
    this->m_posCurrent = 0;
    this->m_run = 0;
    this->init();
}
