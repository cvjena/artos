#include "FeaturePyramid.h"
#include <cmath>
#include <algorithm>
#include <utility>
#include <Eigen/Core>
#include "blf.h"
using namespace ARTOS;
using namespace std;


FeaturePyramid::FeaturePyramid(int interval, const vector<FeatureMatrix> & levels, const vector<double> * scales)
: m_interval(0), m_scales(), m_featureExtractor(FeatureExtractor::defaultFeatureExtractor())
{
    if (interval < 1)
        return;
    
    m_interval = interval;
    m_levels = levels;
    if (scales)
        m_scales = *scales;
}


FeaturePyramid::FeaturePyramid(int interval, vector<FeatureMatrix> && levels, const vector<double> * scales)
: m_interval(0), m_scales(), m_featureExtractor(FeatureExtractor::defaultFeatureExtractor())
{
    if (interval < 1)
        return;
    
    m_interval = interval;
    m_levels = move(levels);
    if (scales)
        m_scales = *scales;
}


FeaturePyramid::FeaturePyramid(const JPEGImage & image, const shared_ptr<FeatureExtractor> & featureExtractor, int interval, unsigned int minSize)
: m_interval(0)
{
    this->m_featureExtractor = (featureExtractor) ? featureExtractor : FeatureExtractor::defaultFeatureExtractor();
    
    if (image.empty() || (interval < 1))
        return;
    
    // Compute the number of scales such that the smallest size of the last level is minSize
    const Size minPixelSize = this->m_featureExtractor->cellsToPixels(Size(minSize));
    const int maxScale = interval * ceil(log(min(
        image.width() / static_cast<double>(minPixelSize.width),
        image.height() / static_cast<double>(minPixelSize.height)
    )) / log(2.0));
    
    // Begin with scales smaller than the size of the original image if the feature extractor requires this
    const Size maxImgSize = this->m_featureExtractor->maxImageSize();
    const int minScale = interval * max(0, static_cast<int>(max(
        (maxImgSize.width > 0) ? ceil(log(2 * image.width() / static_cast<double>(maxImgSize.width)) / log(2.0) * interval) : 0,
        (maxImgSize.height > 0) ? ceil(log(2 * image.height() / static_cast<double>(maxImgSize.height)) / log(2.0) * interval) : 0
    )));
    
    // Cannot compute the pyramid on images too small
    if (maxScale - minScale < interval)
        return;
    
    // Compute scales of each level in the pyramid
    m_interval = interval;
    m_scales.resize(maxScale - minScale + 1);
    
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < interval; ++i)
    {
        double scale = pow(2.0, static_cast<double>(-i) / interval);
        
        // First octave at twice the image resolution
        if (i >= minScale)
            this->m_scales[i - minScale] = scale * 2;
        
        // Second octave at the original resolution
        if (i + interval >= minScale && i + interval <= maxScale)
            this->m_scales[i + interval - minScale] = scale;
        
        // Remaining octaves
        for (int j = 2; i + j * interval <= maxScale; ++j)
        {
            scale *= 0.5;
            if (i + j * interval >= minScale)
                this->m_scales[i + j * interval - minScale] = scale;
        }
    }
    
    if (this->m_featureExtractor->patchworkProcessing())
        this->buildLevelsPatchworked(image);
    else
        this->buildLevels(image);
}


void FeaturePyramid::buildLevels(const JPEGImage & image)
{
    if (image.empty() || this->m_scales.empty())
        return;
    
    this->m_levels.resize(this->m_scales.size());
    
    int i;
    bool threadSafe = this->m_featureExtractor->supportsMultiThread();
    #pragma omp parallel for private(i) if(threadSafe)
    for (i = 0; i < this->m_scales.size(); ++i)
    {
        double scale = this->m_scales[i];
        
        if (scale == 1.0)
            this->m_featureExtractor->extract(image, m_levels[i]);
        else if (scale > 1.0 && this->m_featureExtractor->cellSize().min() > 1 && this->m_featureExtractor->supportsVariableCellSize())
        {
            // First octave at twice the image resolution
            try
            {
                this->m_featureExtractor->extract(
                    image.resize(image.width() * scale / 2 + 0.5, image.height() * scale / 2 + 0.5),
                    m_levels[i],
                    this->m_featureExtractor->cellSize() / 2
                );
            }
            catch (NotSupportedException & e)
            {
                // This should not happen if the feature extractor behaves consistently.
                this->m_featureExtractor->extract(image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5), m_levels[i]);
            }
        }
        else
            this->m_featureExtractor->extract(image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5), m_levels[i]);
    }
}


void FeaturePyramid::buildLevelsPatchworked(const JPEGImage & image)
{
    if (image.empty() || this->m_scales.empty())
        return;
    
    // Plan patchwork
    int i;
    const Size maxSize = Size(image.width() * this->m_scales[0] + 0.5, image.height() * this->m_scales[0] + 0.5);
    const Size padding = max(this->m_featureExtractor->patchworkPadding(), Size(0));
    const Size borderSize = this->m_featureExtractor->borderSize();
    
    vector<PatchworkRectangle> rectangles;
    vector<Size> paddingPerLevel(this->m_scales.size(), Size(0,0));
    rectangles.reserve(this->m_scales.size());
    Size scaledSize;
    for (i = 0; i < this->m_scales.size(); i++)
    {
        scaledSize = Size(image.width() * this->m_scales[i] + 0.5, image.height() * this->m_scales[i] + 0.5);
        if (scaledSize.width + padding.width < maxSize.width && scaledSize.height + padding.height < maxSize.height)
        {
            // Add padding at the right/bottom
            paddingPerLevel[i] = padding;
            
            // Align with cell size
            Size overhang = scaledSize + paddingPerLevel[i];
            overhang = Size(
                overhang.width % this->m_featureExtractor->cellSize().width,
                overhang.height % this->m_featureExtractor->cellSize().height
            );
            if (overhang.width > 0)
                paddingPerLevel[i].width += this->m_featureExtractor->cellSize().width - overhang.width;
            if (overhang.height > 0)
                paddingPerLevel[i].height += this->m_featureExtractor->cellSize().height - overhang.height;
            paddingPerLevel[i] = min(paddingPerLevel[i], maxSize - scaledSize);
            
            scaledSize += paddingPerLevel[i];
        }
        rectangles.push_back(PatchworkRectangle(scaledSize.width, scaledSize.height));
    }
    
    int numPlanes = BLF(rectangles, maxSize.width, maxSize.height);
    if (numPlanes <= 0)
        throw runtime_error("Could not construct feature pyramid: Bottom-left fill algorithm failed.");
    
    // Fill patchwork planes
    vector<JPEGImage> planes;
    planes.reserve(numPlanes);
    for (i = 0; i < numPlanes; i++)
    {
        planes.push_back(JPEGImage(maxSize.width, maxSize.height, image.depth()));
        planes.back().toMatrix().setZero();
    }
    
    #pragma omp parallel for private(i)
    for (i = 0; i < rectangles.size(); i++)
    {
        PatchworkRectangle & rect = rectangles[i];
        assert(rect.plane() >= 0 && rect.plane() < planes.size());
        assert(rect.x() % this->m_featureExtractor->cellSize().width == 0 && rect.y() % this->m_featureExtractor->cellSize().height == 0);
        double scale = this->m_scales[i];
        JPEGImage scaled = (scale != 1.0) ? image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5) : image;
        planes[rect.plane()].toMatrix().data().block(rect.y(), rect.x() * scaled.depth(), scaled.height(), scaled.width() * scaled.depth()) = scaled.toMatrix().data();
    }
    
    // Run feature extractor over planes
    bool threadSafe = this->m_featureExtractor->supportsMultiThread();
    vector<FeatureMatrix> features(numPlanes);
    #pragma omp parallel for private(i) if(threadSafe)
    for (i = 0; i < numPlanes; i++)
        this->m_featureExtractor->extract(planes[i], features[i]);
    planes.clear();
    
    // Extract levels from planes
    this->m_levels.resize(this->m_scales.size());
    for (i = 0; i < rectangles.size(); i++)
    {
        PatchworkRectangle & rect = rectangles[i];
        FeatureMatrix & level = this->m_levels[i];
        Size levelSize = this->m_featureExtractor->pixelsToCells(Size(rect.width(), rect.height()) - paddingPerLevel[i]);
        Size levelCoords = this->m_featureExtractor->pixelCoordsToCells(Size(rect.x(), rect.y()));
        
        level.resize(levelSize.height, levelSize.width, features[0].channels());
        level.data() = features[rect.plane()].data().block(
            levelCoords.height,
            levelCoords.width * level.channels(),
            level.rows(),
            level.cols() * level.channels()
        );
    }
}
