#include "FeatureExtractor.h"
#include <cmath>
#include <algorithm>
#include <utility>
#include <cstring>
#include <Eigen/Core>
#include "strutils.h"
using namespace ARTOS;
using namespace std;

////////////////////////////////////////////////////////////////////
//// Static stuff for feature extractor enumartion and creation ////
////////////////////////////////////////////////////////////////////

#include "HOGFeatureExtractor.h"
typedef HOGFeatureExtractor DefaultFeatureExtractor;

shared_ptr<FeatureExtractor> FeatureExtractor::dfltFeatureExtractor = nullptr;

static shared_ptr<FeatureExtractor> createHOGFeatureExtractor() { return make_shared<HOGFeatureExtractor>(); };

map<string, shared_ptr<FeatureExtractor> (*)()> FeatureExtractor::featureExtractorFactories {
    { "HOG", createHOGFeatureExtractor } // the name specified here must correspond to the return value of type()
};



/////////////////////////////////////////////////////////////
//// Default implementations of FeatureExtractor methods ////
/////////////////////////////////////////////////////////////


int32_t FeatureExtractor::getIntParam(const string & paramName) const
{
    auto it = this->m_intParams.find(paramName);
    if (it == this->m_intParams.end())
        throw UnknownParameterException(string(this->type()) + " feature extractor has no integer parameter called " + paramName + ".");
    return it->second;
}


FeatureScalar FeatureExtractor::getScalarParam(const string & paramName) const
{
    auto it = this->m_scalarParams.find(paramName);
    if (it == this->m_scalarParams.end())
        throw UnknownParameterException(string(this->type()) + " feature extractor has no scalar parameter called " + paramName + ".");
    return it->second;
}


string FeatureExtractor::getStringParam(const string & paramName) const
{
    auto it = this->m_stringParams.find(paramName);
    if (it == this->m_stringParams.end())
        throw UnknownParameterException(string(this->type()) + " feature extractor has no string parameter called " + paramName + ".");
    return it->second;
}


void FeatureExtractor::setParam(const string & paramName, int32_t val)
{
    auto it = this->m_intParams.find(paramName);
    if (it == this->m_intParams.end())
        throw UnknownParameterException(string(this->type()) + " feature extractor has no integer parameter called " + paramName + ".");
    it->second = val;
}


void FeatureExtractor::setParam(const string & paramName, FeatureScalar val)
{
    auto it = this->m_scalarParams.find(paramName);
    if (it == this->m_scalarParams.end())
        throw UnknownParameterException(string(this->type()) + " feature extractor has no scalar parameter called " + paramName + ".");
    it->second = val;
}


void FeatureExtractor::setParam(const string & paramName, const string & val)
{
    auto it = this->m_stringParams.find(paramName);
    if (it == this->m_stringParams.end())
        throw UnknownParameterException(string(this->type()) + " feature extractor has no string parameter called " + paramName + ".");
    it->second = val;
}


void FeatureExtractor::listParameters(vector<ParameterInfo> & params)
{
    params.clear();
    params.reserve(this->m_intParams.size() + this->m_scalarParams.size() + this->m_stringParams.size());
    for (const auto & param : this->m_intParams)
        params.push_back(ParameterInfo(param.first, param.second));
    for (const auto & param : this->m_scalarParams)
        params.push_back(ParameterInfo(param.first, param.second));
    for (const auto & param : this->m_stringParams)
        params.push_back(ParameterInfo(param.first, param.second));
}


bool FeatureExtractor::operator==(const FeatureExtractor & other)
{
    return (strcmp(this->type(), other.type()) == 0
            && this->m_intParams == other.m_intParams
            && this->m_scalarParams == other.m_scalarParams
            && this->m_stringParams == other.m_stringParams);
}


Size FeatureExtractor::computeOptimalModelSize(const vector<Size> & sizes, const Size & maxSize) const
{
    // Some shortcuts
    Size ms = min(maxSize, this->pixelsToCells(this->maxImageSize()));
    int csx = this->cellSize().width, csy = this->cellSize().height,
        msx = ms.width, msy = ms.height;
    
    // Compute common aspect ratio
    vector<int> areas;
    float aspect = FeatureExtractor::commonAspectRatio(sizes, &areas);
    
    // Pick 20 percentile area
    size_t areaInd = static_cast<size_t>(floor(areas.size() * 0.2));
    partial_sort(areas.begin(), areas.begin() + areaInd + 1, areas.end());
    int area = areas[areaInd];
    
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



////////////////////////
//// Static methods ////
////////////////////////


float FeatureExtractor::commonAspectRatio(const vector<Size> & sizes, vector<int> * areas)
{
    int i, j;
    
    // Fill histogram and area vector
    if (areas != NULL)
    {
        areas->clear();
        areas->reserve(sizes.size());
    }
    Eigen::Array<float, 1, 201> hist; // histogram of logarithmic aspect ratios with bins from -2 to +2 in steps of 0.02
    hist.setConstant(0.0f);
    int aspectIndex;
    for (vector<Size>::const_iterator size = sizes.begin(); size != sizes.end(); size++)
    {
        aspectIndex = round(log(static_cast<float>(size->height) / static_cast<float>(size->width)) * 50 + 100);
        if (aspectIndex >= 0 && aspectIndex < hist.size())
            hist(aspectIndex) += 1;
        if (areas != NULL)
            areas->push_back(size->width * size->height);
    }
    
    // Filter histogram with large gaussian smoothing filter and select maximum as aspect ratio
    Eigen::Array<float, 1, 201> filter;
    for (i = 0; i < filter.size(); i++)
        filter(i) = exp(static_cast<float>((100 - i) * (100 - i)) / -400.0f);
    float curValue, maxValue = 0;
    int maxIndex = 0;
    for (i = 0; i < hist.size(); i++)
    {
        curValue = 0;
        for (j = max(i - 100, 0); j < min(i + 100, 200); j++)
            curValue += hist(j) * filter(j - i + 100);
        if (curValue > maxValue)
        {
            maxIndex = i;
            maxValue = curValue;
        }
    }
    return exp(maxIndex * 0.02f - 2);
}


shared_ptr<FeatureExtractor> FeatureExtractor::create(const string & type)
{
    const auto it = FeatureExtractor::featureExtractorFactories.find(type);
    if (it == FeatureExtractor::featureExtractorFactories.end())
        throw UnknownFeatureExtractorException(type);
    return it->second();
}


shared_ptr<FeatureExtractor> FeatureExtractor::defaultFeatureExtractor()
{
    if (!FeatureExtractor::dfltFeatureExtractor)
        FeatureExtractor::dfltFeatureExtractor = make_shared<DefaultFeatureExtractor>();
    return FeatureExtractor::dfltFeatureExtractor;
}


void FeatureExtractor::setDefaultFeatureExtractor(const shared_ptr<FeatureExtractor> & newDefault)
{
    if (newDefault)
        FeatureExtractor::dfltFeatureExtractor = newDefault;
}


void FeatureExtractor::setDefaultFeatureExtractor(const string & type)
{
    FeatureExtractor::setDefaultFeatureExtractor(FeatureExtractor::create(type));
}


int FeatureExtractor::numFeatureExtractors()
{
    return FeatureExtractor::featureExtractorFactories.size();
}


void FeatureExtractor::listFeatureExtractors(vector<string> & featureExtractors)
{
    featureExtractors.clear();
    for (const auto & fe : FeatureExtractor::featureExtractorFactories)
        featureExtractors.push_back(fe.first);
}


void FeatureExtractor::listFeatureExtractors(vector< shared_ptr<FeatureExtractor> > & featureExtractors)
{
    featureExtractors.clear();
    for (const auto & fe : FeatureExtractor::featureExtractorFactories)
        featureExtractors.push_back(fe.second());
}


ostream & ARTOS::operator<<(ostream & os, const FeatureExtractor & featureExtractor)
{
    for (const auto & param : featureExtractor.m_intParams)
        os << param.first << ' ' << param.second << ' ';
    for (const auto & param : featureExtractor.m_scalarParams)
        os << param.first << ' ' << param.second << ' ';
    for (const auto & param : featureExtractor.m_stringParams)
        os << param.first << " {str{" << param.second << "}str} ";
    os << endl;
    return os;
}


istream & ARTOS::operator>>(istream & is, FeatureExtractor & featureExtractor)
{
    // Fetch line from the stream and split it into tokens
    string line;
    getline(is, line);
    vector<string> tokens;
    splitString(trim(line), " ", tokens);
    
    // Merge string sequences
    int str_start = -1;
    for (int i = 0; i < tokens.size(); i++)
    {
        if (str_start >= 0)
            tokens[str_start] += tokens[i];
        else if (tokens[i].substr(0, 5) == "{str{")
        {
            str_start = i;
            tokens[i].erase(0, 5);
        }
        if (str_start >= 0 && tokens[i].substr(tokens[i].size() - 5) == "}str}")
        {
            tokens[str_start].erase(tokens[str_start].size() - 5);
            tokens.erase(tokens.begin() + str_start + 1, tokens.begin() + i + 1);
            i = str_start;
            str_start = -1;
        }
    }
    
    // Check for an even number of tokens
    if (tokens.size() % 2 != 0)
        throw DeserializationException("The given stream could not be deserialized into a feature extractor.");
    
    // Deserialize parameters
    for (size_t i = 0; i < tokens.size(); i += 2)
    {
        if (featureExtractor.m_intParams.find(tokens[i]) != featureExtractor.m_intParams.end())
        {
            char * trailing;
            long val = strtol(tokens[i+1].c_str(), &trailing, 0);
            if (trailing != NULL && *trailing != '\0')
                throw DeserializationException("Invalid value for parameter " + tokens[i] + " of " + featureExtractor.type()
                        + " feature extractor: " + tokens[i+1] + " (expected int)");
            featureExtractor.setParam(tokens[i], static_cast<int32_t>(val));
        }
        else if (featureExtractor.m_scalarParams.find(tokens[i]) != featureExtractor.m_scalarParams.end())
        {
            char * trailing;
            double val = strtod(tokens[i+1].c_str(), &trailing);
            if (trailing != NULL && *trailing != '\0')
                throw DeserializationException("Invalid value for parameter " + tokens[i] + " of " + featureExtractor.type()
                        + " feature extractor: " + tokens[i+1] + " (expected float)");
            featureExtractor.setParam(tokens[i], static_cast<FeatureScalar>(val));
        }
        else if (featureExtractor.m_stringParams.find(tokens[i]) != featureExtractor.m_stringParams.end())
        {
            featureExtractor.setParam(tokens[i], tokens[i+1]);
        }
        else
            throw UnknownParameterException(string(featureExtractor.type()) + " feature extractor has no parameter called "
                    + tokens[i] + " (found on input stream during deserialization).");
    }
    
    return is;
}
