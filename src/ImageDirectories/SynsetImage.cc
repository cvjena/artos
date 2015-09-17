#include "SynsetImage.h"
#include <cmath>
#include "libartos_def.h"
#include "sysutils.h"
#include "ffld/Scene.h"
using namespace ARTOS;
using namespace FFLD;
using namespace std;


SynsetImage::SynsetImage(const string & repoDirectory, const string & synsetId,
                         const string & filename, const JPEGImage * img)
: m_repoDir(repoDirectory), m_synsetId(synsetId), m_filename(strip_file_extension(filename)), m_imgLoaded(false), m_bboxesLoaded(false)
{
    if (img != NULL && !img->empty())
        this->m_img = JPEGImage(img->width(), img->height(), img->depth(), img->bits());
}


string SynsetImage::getPath() const
{
    string path, basename = join_path(3, this->m_repoDir.c_str(), this->m_synsetId.c_str(), this->m_filename.c_str());
    if (is_file(path = basename + ".jpg") || is_file(path = basename + ".jpeg") || is_file(path = basename + ".JPG") || is_file(path = basename + ".JPEG"))
        return path;
    return "";
}


void SynsetImage::loadImage()
{
    string path = this->getPath();
    if (!path.empty())
        this->m_img = JPEGImage(path);
}


bool SynsetImage::loadBoundingBoxes()
{
    if (!this->m_bboxesLoaded)
    {
        string path;
        if (is_file(path = join_path(3, this->m_repoDir.c_str(), this->m_synsetId.c_str(), (this->m_filename + ".xml").c_str()))
            || is_file(path = join_path(3, this->m_repoDir.c_str(), this->m_synsetId.c_str(), (this->m_filename + ".XML").c_str())))
        {
            Scene scene(path);
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
    }
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
