#include "FeatureExtractor.h"
#include <cmath>
#include <algorithm>
using namespace ARTOS;
using namespace std;
using FFLD::JPEGImage;

FeaturePyramid::FeaturePyramid(int interval, const vector<FeatureExtractor::FeatureMatrix> & levels) : m_interval(0)
{
    if (interval < 1)
        return;
    
    m_interval = interval;
    m_levels = levels;
}

FeaturePyramid::FeaturePyramid(const JPEGImage & image, int interval) : m_interval(0)
{
    if (image.empty() || (interval < 1))
        return;
    
    // Compute the number of scales such that the smallest size of the last level is 5
    const int maxScale = ceil(log(min(image.width(), image.height()) / 40.0) / log(2.0)) * interval;
    
    // Cannot compute the pyramid on images too small
    if (maxScale < interval)
        return;
    
    m_interval = interval;
    m_levels.resize(maxScale + 1);
    
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < interval; ++i) {
        double scale = pow(2.0, static_cast<double>(-i) / interval);
        
        JPEGImage scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
        
        // First octave at twice the image resolution
        if (FeatureExtractor::cellSize > 1)
            FeatureExtractor::extract(scaled, m_levels[i], FeatureExtractor::cellSize / 2);
        else
            FeatureExtractor::extract(image.resize(image.width() * scale * 2 + 0.5, image.height() * scale * 2 + 0.5), m_levels[i]);
        
        // Second octave at the original resolution
        if (i + interval <= maxScale)
            FeatureExtractor::extract(scaled, m_levels[i + interval]);
        
        // Remaining octaves
        for (int j = 2; i + j * interval <= maxScale; ++j) {
            scale *= 0.5;
            scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
            FeatureExtractor::extract(scaled, m_levels[i + j * interval]);
        }
    }
}