#include "HOGFeatureExtractor.h"
#include <limits>
#include <cstdint>
#include <cmath>
#include <cassert>
using namespace ARTOS;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


HOGFeatureExtractor::HOGFeatureExtractor() : HOGFeatureExtractor(Size(8)) {};


HOGFeatureExtractor::HOGFeatureExtractor(const Size & cellSize)
: m_cellSize((cellSize.width > 0 && cellSize.height > 0 && cellSize.width % 2 == 0 && cellSize.height % 2 == 0) ? cellSize : Size(8))
{
    this->m_intParams["cellSizeX"] = this->m_cellSize.width;
    this->m_intParams["cellSizeY"] = this->m_cellSize.height;
};


void HOGFeatureExtractor::extract(const JPEGImage & img, FeatureMatrix & feat, const Size & cellSize) const
{
    HOGFeatureExtractor::HOG(img, feat, Size(1, 1), (cellSize.width > 0 && cellSize.height > 0) ? cellSize : this->cellSize());
    if (feat.rows() > 2 && feat.cols() > 2)
        feat.crop(1, 1, feat.rows() - 2, feat.cols() - 2); // cut off padding
}


void HOGFeatureExtractor::flip(const FeatureMatrix & feat, FeatureMatrix & flipped) const
{
    // Symmetric features
    const int symmetry[32] = {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
        18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
        28, 27, 30, 29, // Texture
        31 // Truncation
    };

    // Symmetric filter
    flipped = FeatureMatrix(feat.rows(), feat.cols(), FeatureCell::Zero(32));
    for (int y = 0; y < feat.rows(); ++y)
        for (int x = 0; x < feat.cols(); ++x)
            for (int i = 0; i < 32; ++i)
                flipped(y, x, i) = feat(y, feat.cols() - 1 - x, symmetry[i]);
}


void HOGFeatureExtractor::setParam(const std::string & paramName, int32_t val)
{
    if (paramName == "cellSizeX")
    {
        if (val < 1 || val % 2 != 0)
            throw invalid_argument("cellSizeX must be a positive multiple of 2.");
        this->m_cellSize.width = val;
    }
    else if (paramName == "cellSizeY")
    {
        if (val < 1 || val % 2 != 0)
            throw invalid_argument("cellSizeY must be a positive multiple of 2.");
        this->m_cellSize.height = val;
    }
    FeatureExtractor::setParam(paramName, val);
}


Size HOGFeatureExtractor::computeOptimalModelSize(const vector<Size> & sizes, const Size & maxSize) const
{
    // Some shortcuts
    Size ms = max(Size(0), maxSize);
    const Size maxImgSize = this->maxImageSize();
    const Size maxCellSize = this->pixelsToCells(maxImgSize);
    if (maxImgSize.width > 0 && (ms.width == 0 || ms.width > maxCellSize.width))
        ms.width = maxCellSize.width;
    if (maxImgSize.height > 0 && (ms.height == 0 || ms.height > maxCellSize.height))
        ms.height = maxCellSize.height;
    int csx = this->cellSize().width, csy = this->cellSize().height,
        msx = ms.width, msy = ms.height;
    
    // Compute common aspect ratio
    vector<int> areas;
    float aspect = FeatureExtractor::commonAspectRatio(sizes, &areas);
    
    // Pick 20 percentile area
    size_t areaInd = static_cast<size_t>(floor(areas.size() * 0.2));
    partial_sort(areas.begin(), areas.begin() + areaInd + 1, areas.end());
    int area = areas[areaInd];
    
    // The following is a hack from the original WHO code which sets lower and
    // upper bounds for the area of the samples if the given parameters correspond
    // to the ones of the original WHO implementation.
    // It is not strictly necessary and it's only justification here is, that it
    // led to good results in practice and needs less memory than using the maximum
    // possible model size.
    if (csx == 8 && csy == 8 && msx == 20 && msy == 20)
        area = max(min(area, 7000), 5000);
    
    // Ensure that feature areas are neither too big nor too small
    if (msx > 0 || msy > 0)
    {
        float scale = max(
            (msx > 0) ? area / (aspect * msx * msx * csx * csx) : 0,
            (msy > 0) ? (area * aspect) / (msy * msy * csy * csy) : 0
        );
        if (scale > 1)          // larger dimension exceeds maxSize
            area /= scale;      // -> scale it to match maxSize
    }
    
    // Calculate model size in cells
    float width = sqrt(static_cast<float>(area) / aspect);
    float height = width * aspect;
    Size size;
    size.width = max(static_cast<int>(round(width / csx)), 1);
    size.height = max(static_cast<int>(round(height / csy)), 1);
    return size;
}


// Bilinear interpolation among the 4 neighboring cells
static inline void interpolate(int x, int y, int bin0, int bin1, FeatureScalar magnitude0, FeatureScalar magnitude1,
                               const Size & cellSize, FeatureMatrix & matrix)
{
    // Find the bin into which (x, y) falls
    const Size csh = cellSize / 2;
    const int i = (y - csh.height) / cellSize.height;
    const int j = (x - csh.width) / cellSize.width;
    const int k = (y - csh.height) % cellSize.height;
    const int l = (x - csh.width) % cellSize.width;

    // Bilinear interpolation
    const int a = k * 2 + 1;
    const int b = cellSize.height * 2 - a;
    const int c = l * 2 + 1;
    const int d = cellSize.width * 2 - c;

    matrix(i    , j    , bin0) += magnitude0 * (b * d);
    matrix(i    , j    , bin1) += magnitude1 * (b * d);
    matrix(i    , j + 1, bin0) += magnitude0 * (b * c);
    matrix(i    , j + 1, bin1) += magnitude1 * (b * c);
    matrix(i + 1, j    , bin0) += magnitude0 * (a * d);
    matrix(i + 1, j    , bin1) += magnitude1 * (a * d);
    matrix(i + 1, j + 1, bin0) += magnitude0 * (a * c);
    matrix(i + 1, j + 1, bin1) += magnitude1 * (a * c);
}


void HOGFeatureExtractor::HOG(const JPEGImage & image, FeatureMatrix & feat, const Size & padding, const Size & cellSize)
{
    // Table of all the possible tangents (1MB)
    static FeatureScalar ATAN2_TABLE[512][512] = {{0}};
    
    // Fill the atan2 table
#pragma omp critical
    if (ATAN2_TABLE[0][0] == 0) {
        for (int dy = -255; dy <= 255; ++dy) {
            for (int dx = -255; dx <= 255; ++dx) {
                // Angle in the range [-pi, pi]
                double angle = atan2(static_cast<double>(dy), static_cast<double>(dx));
                
                // Convert it to the range [9.0, 27.0]
                angle = angle * (9.0 / M_PI) + 18.0;
                
                // Convert it to the range [0, 18)
                if (angle >= 18.0)
                    angle -= 18.0;
                
                ATAN2_TABLE[dy + 255][dx + 255] = max(angle, 0.0);
            }
        }
    }
    
    while (ATAN2_TABLE[510][510] == 0);
    
    // Some shortcuts
    const int width = image.width();
    const int height = image.height();
    const int depth = image.depth();
    const Size padCells = cellSize * padding;
    
    // Make sure the image is big enough
    assert(cellSize.width % 2 == 0);
    assert(cellSize.height % 2 == 0);
    assert(width >= cellSize.width / 2);
    assert(height >= cellSize.height / 2);
    assert(depth >= 1);
    assert(padding.width >= 1);
    assert(padding.height >= 1);
    
    // Resize the feature matrix
    feat = FeatureMatrix((height + cellSize.height / 2) / cellSize.height + padding.height * 2,
                         (width + cellSize.width / 2) / cellSize.width + padding.width * 2,
                         FeatureCell::Zero(32));
    
    for (int y = 0; y < height; ++y)
    {
        const int yp = min(y + 1, height - 1);
        const int ym = max(y - 1, 0);
        
        const uint8_t * linep = reinterpret_cast<const uint8_t *>(image.scanLine(yp));
        const uint8_t * line = reinterpret_cast<const uint8_t *>(image.scanLine(y));
        const uint8_t * linem = reinterpret_cast<const uint8_t *>(image.scanLine(ym));
        
        for (int x = 0; x < width; ++x)
        {
            const int xp = min(x + 1, width - 1);
            const int xm = max(x - 1, 0);
            
            // Use the channel with the largest gradient magnitude
            FeatureScalar magnitude = 0;
            FeatureScalar theta = 0;
            
            for (int i = 0; i < depth; ++i)
            {
                const int dx = static_cast<int>(line[xp * depth + i]) -
                               static_cast<int>(line[xm * depth + i]);
                const int dy = static_cast<int>(linep[x * depth + i]) -
                               static_cast<int>(linem[x * depth + i]);
                
                if (dx * dx + dy * dy > magnitude)
                {
                    magnitude = dx * dx + dy * dy;
                    theta = ATAN2_TABLE[dy + 255][dx + 255];
                }
            }
            
            magnitude = sqrt(magnitude);
            
            // Bilinear interpolation
            const int theta0 = theta;
            const int theta1 = (theta0 < 17) ? (theta0 + 1) : 0;
            const FeatureScalar magnitude1 = magnitude * (theta - theta0);
            interpolate(x + padCells.width, y + padCells.height,
                        theta0, theta1, magnitude - magnitude1, magnitude1,
                        cellSize, feat);
        }
    }
    
    // Compute the "gradient energy" of each cell, i.e. ||C(i,j)||^2
    feat.channel(31).setZero();
    for (int i = 0; i < 9; ++i)
        feat.channel(31) += (feat.channel(i) + feat.channel(i + 9)).cwiseAbs2();
    
    // Compute the four normalization factors then normalize and clamp everything
    const FeatureScalar EPS = numeric_limits<FeatureScalar>::epsilon();
    for (int y = padding.height; y < feat.rows() - padding.height; ++y)
        for (int x = padding.width; x < feat.cols() - padding.width; ++x)
        {
            // Normalization factors
            const FeatureScalar n0 = 1 / sqrt(feat(y - 1, x - 1, 31) +
                                              feat(y - 1, x    , 31) +
                                              feat(y    , x - 1, 31) +
                                              feat(y    , x    , 31) + EPS);
            const FeatureScalar n1 = 1 / sqrt(feat(y - 1, x    , 31) +
                                              feat(y - 1, x + 1, 31) +
                                              feat(y    ,     x, 31) +
                                              feat(y    , x + 1, 31) + EPS);
            const FeatureScalar n2 = 1 / sqrt(feat(y    , x - 1, 31) +
                                              feat(y    , x    , 31) +
                                              feat(y + 1, x - 1, 31) +
                                              feat(y + 1, x    , 31) + EPS);
            const FeatureScalar n3 = 1 / sqrt(feat(y    , x    , 31) +
                                              feat(y    , x + 1, 31) +
                                              feat(y + 1, x    , 31) +
                                              feat(y + 1, x + 1, 31) + EPS);
            
            // Contrast-insensitive features
            for (int i = 0; i < 9; ++i)
            {
                const FeatureScalar sum = feat(y, x, i) + feat(y, x, i + 9);
                const FeatureScalar h0 = min(sum * n0, FeatureScalar(0.2));
                const FeatureScalar h1 = min(sum * n1, FeatureScalar(0.2));
                const FeatureScalar h2 = min(sum * n2, FeatureScalar(0.2));
                const FeatureScalar h3 = min(sum * n3, FeatureScalar(0.2));
                feat(y, x, i + 18) = (h0 + h1 + h2 + h3) * FeatureScalar(0.5);
            }
            
            // Contrast-sensitive features
            FeatureScalar t0 = 0, t1 = 0, t2 = 0, t3 = 0;
            for (int i = 0; i < 18; ++i)
            {
                const FeatureScalar sum = feat(y, x, i);
                const FeatureScalar h0 = min(sum * n0, FeatureScalar(0.2));
                const FeatureScalar h1 = min(sum * n1, FeatureScalar(0.2));
                const FeatureScalar h2 = min(sum * n2, FeatureScalar(0.2));
                const FeatureScalar h3 = min(sum * n3, FeatureScalar(0.2));
                feat(y, x, i) = (h0 + h1 + h2 + h3) * FeatureScalar(0.5);
                t0 += h0;
                t1 += h1;
                t2 += h2;
                t3 += h3;
            }
            
            // Texture features
            feat(y, x, 27) = t0 * FeatureScalar(0.2357);
            feat(y, x, 28) = t1 * FeatureScalar(0.2357);
            feat(y, x, 29) = t2 * FeatureScalar(0.2357);
            feat(y, x, 30) = t3 * FeatureScalar(0.2357);
        }
    
    // Truncation features
    for (int y = 0; y < feat.rows(); ++y)
        for (int x = 0; x < feat.cols(); ++x)
        {
            if (y < padding.height || y >= feat.rows() - padding.height || x < padding.width || x >= feat.cols() - padding.width)
            {
                feat(y, x).setZero();
                feat(y, x, 31) = 1;
            }
            else
                feat(y, x, 31) = 0;
        }
}