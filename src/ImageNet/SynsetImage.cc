#include "SynsetImage.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <limits>
#include <utility>
#include "libartos_def.h"
#include "TarExtractor.h"
#include "sysutils.h"
#include "Scene.h"
using namespace ARTOS;
using namespace std;


SynsetImage::SynsetImage(const string & repoDirectory, const string & synsetId,
                         const string & filename, const JPEGImage * img)
: m_repoDir(repoDirectory), m_synsetId(synsetId), m_filename(strip_file_extension(filename)), m_imgLoaded(false), m_bboxesLoaded(false)
{
#ifndef NO_CACHE_POSITIVES
    if (img != NULL && !img->empty())
        this->m_img = JPEGImage(img->width(), img->height(), img->depth(), img->bits());
#endif
}


SynsetImage::SynsetImage(SynsetImage && other)
: bboxes(move(other.bboxes)), m_repoDir(move(other.m_repoDir)), m_synsetId(move(other.m_synsetId)), m_filename(move(other.m_filename)),
  m_img(move(other.m_img)), m_imgLoaded(other.m_imgLoaded), m_bboxesLoaded(other.m_bboxesLoaded)
{
    other.m_imgLoaded = other.m_bboxesLoaded = false;
}


SynsetImage & SynsetImage::operator=(SynsetImage && other)
{
    this->bboxes = move(other.bboxes);
    this->m_img = move(other.m_img);
    this->m_repoDir = move(other.m_repoDir);
    this->m_synsetId = move(other.m_synsetId);
    this->m_filename = move(other.m_filename);
    this->m_imgLoaded = other.m_imgLoaded;
    this->m_bboxesLoaded = other.m_bboxesLoaded;
    other.m_imgLoaded = other.m_bboxesLoaded = false;
    return *this;
}


void SynsetImage::loadImage(JPEGImage * target) const
{
    // Search for the image in the Tar archive and determine it's offset
    string tarFilename = this->m_synsetId + ".tar";
    string tarPath = join_path(3, this->m_repoDir.c_str(), IMAGENET_IMAGE_DIR, tarFilename.c_str());
    TarFileInfo info = TarExtractor::findFileInArchive(tarPath, this->m_filename, TarExtractor::IGNORE_FILE_EXT);
    if (info.type == tft_file)
    {
        // Load image from file using libjpeg
        this->readImageFromFileOffset(tarPath, info.offset, target);
    }
}


bool SynsetImage::readImageFromFileOffset(const string & filename, streamoff offset, JPEGImage * target) const
{
    bool success = false;
    if (target == NULL)
        target = &(this->m_img);
    FILE * fh = fopen(filename.c_str(), "rb");
    if (fh != NULL)
    {
        long int_max = numeric_limits<long>::max();
        while (offset > int_max)
        {
            fseek(fh, int_max, SEEK_CUR);
            offset -= int_max;
        }
        if (fseek(fh, offset, SEEK_CUR) == 0)
        {
            *target = JPEGImage(fh);
            success = true;
        }
        fclose(fh);
    }
    return success;
}


bool SynsetImage::loadBoundingBoxes()
{
    if (!this->m_bboxesLoaded)
    {
        // Search for the annotation file in the annotations tar archive and extract it
        string tarFilename = this->m_synsetId + ".tar";
        string tarPath = join_path(3, this->m_repoDir.c_str(), IMAGENET_ANNOTATION_DIR, tarFilename.c_str());
        TarExtractor tar(tarPath);
        TarFileInfo info;
        char * xmlData = NULL;
        uint64_t bufsize;
        do
        {
            info = tar.readHeader();
            if (info.type == tft_file && strip_file_extension(extract_basename(info.filename)) == this->m_filename)
                xmlData = tar.extract(bufsize);
        }
        while (xmlData == NULL && tar.nextFile());
        tar.close();
        
        this->loadBoundingBoxes(xmlData, bufsize);
        if (xmlData != NULL)
            free(xmlData);
    }
    return !this->bboxes.empty();
}


bool SynsetImage::loadBoundingBoxes(const char * xmlBuffer, const uint64_t bufsize)
{
    this->bboxes.clear();
    
    // Parse XML data into a Scene
    if (xmlBuffer != NULL)
    {
        Scene scene(xmlBuffer, static_cast<int>(bufsize));
        if (!scene.empty())
        {
            JPEGImage img = this->getImage();
            double scale = (img.empty()) ? 1.0 : (static_cast<double>(scene.width()) / img.width());
            for (vector<Object>::const_iterator objIt = scene.objects().begin(); objIt != scene.objects().end(); objIt++)
            {
                Rectangle bbox = objIt->bndbox();
                bbox.setX(round(bbox.x() * scale));
                bbox.setY(round(bbox.y() * scale));
                bbox.setWidth(round(bbox.width() * scale));
                bbox.setHeight(round(bbox.height() * scale));
                if (bbox.x() > 0 && bbox.y() > 0 && bbox.x() < img.width() && bbox.y() < img.height() && bbox.width() > 0 && bbox.height() > 0)
                    this->bboxes.push_back(bbox);
            }
        }
    }
    
    this->m_bboxesLoaded = true;
    return !this->bboxes.empty();
}


void SynsetImage::getSamplesFromBoundingBoxes(vector<JPEGImage> & samples)
{
    if (this->loadBoundingBoxes())
    {
        JPEGImage img = this->getImage(), sample;
        if (!img.empty())
            for (vector<Rectangle>::const_iterator bbox = this->bboxes.begin(); bbox != this->bboxes.end(); bbox++)
            {
                sample = img.crop(bbox->x(), bbox->y(), bbox->width(), bbox->height());
                if (!sample.empty())
                    samples.push_back(sample);
            }
    }
}
