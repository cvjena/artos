#include "StationaryBackground.h"
#include <fstream>
#include <algorithm>
#include <vector>
#include <complex>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <fftw3.h>
#include "portable_endian.h"
#include "FeaturePyramid.h"
#include "JPEGImage.h"
using namespace ARTOS;
using namespace std;

template<class Derived>
static typename Derived::PlainObject fftshift(const Eigen::MatrixBase<Derived> &);

bool StationaryBackground::readFromFile(const string & filename)
{
    ifstream file(filename.c_str(), ifstream::in | ifstream::binary);
    if (!file.is_open())
        return false;
    
    // Read parameters (file format version, cell size, number of features, number of offsets) from file
    uint32_t formatVersion, csx, csy, nf, no;
    file.read(reinterpret_cast<char*>(&formatVersion), sizeof(uint32_t));
    formatVersion = le32toh(formatVersion);
    if ((formatVersion & 0xFFFFFF00) == ARTOS_BG_MAGIC) // Check if first 24 bits of first integer correspond to magic number.
        formatVersion = formatVersion & 0xFF;           // If so, the last 8 bits specify the file format version.
    else                                                // Otherwise, this is the first format version, which did not have
    {                                                   // a dedicated version field at all.
        csx = csy = formatVersion;
        formatVersion = 0;
    }
    if (formatVersion > 1)
        return false;
    
    if (formatVersion >= 1)
    {
        file.read(reinterpret_cast<char*>(&csx), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&csy), sizeof(uint32_t));
        csx = le32toh(csx);
        csy = le32toh(csy);
    }
    file.read(reinterpret_cast<char*>(&nf), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&no), sizeof(uint32_t));
    nf = le32toh(nf);
    no = le32toh(no);
    if (csx == 0 || csy == 0 || nf == 0 || no == 0)
        return false;
    
    // Initialize matrices and arrays
    this->cellSize = Size(csx, csy);
    this->mean.resize(nf);
    this->cov.resize(no);
    this->offsets.resize(no, Eigen::NoChange);
    
    float buf;
    char * buf_p = reinterpret_cast<char*>(&buf);
    uint32_t i, j, k;
    // Read mean
    for (i = 0; i < nf; i++)
    {
        file.read(buf_p, sizeof(float));
        if (!file.good())
        {
            this->clear();
            return false;
        }
        this->mean(i) = le32toh(buf);
    }
    
    // Read covariance
    CovMatrix tmpCovMatrix(nf, nf);
    for (i = 0; i < no; i++)
    {
        for (j = 0; j < nf; j++)
            for (k = 0; k < nf; k++)
            {
                file.read(buf_p, sizeof(float));
                if (!file.good())
                {
                    this->clear();
                    return false;
                }
                tmpCovMatrix(j,k) = le32toh(buf);
            }
        this->cov(i) = tmpCovMatrix;
    }
    
    // Read offsets
    int32_t ibuf;
    buf_p = reinterpret_cast<char*>(&ibuf);
    for (i = 0; i < no; i++)
    {
        file.read(buf_p, sizeof(int32_t));
        if (!file.good())
        {
            this->clear();
            return false;
        }
        this->offsets(i,0) = le32toh(ibuf);
        file.read(buf_p, sizeof(int32_t));
        if (!file.good())
        {
            this->clear();
            return false;
        }
        this->offsets(i,1) = le32toh(ibuf);
    }

    return true;
}

bool StationaryBackground::writeToFile(const string & filename)
{
    if (this->empty())
        return false;
    ofstream file(filename.c_str(), ofstream::out | ofstream::binary);
    if (!file.is_open())
        return false;
    
    // Write parameters (file format version, cell size, number of features, number of offsets)
    uint32_t formatVersion, csx, csy, nf, no;
    formatVersion = htole32(ARTOS_BG_MAGIC | 1);
    csx = htole32(this->cellSize.width);
    csy = htole32(this->cellSize.height);
    nf = htole32(this->getNumFeatures());
    no = htole32(this->getNumOffsets());
    file.write(reinterpret_cast<char*>(&formatVersion), sizeof(uint32_t));
    file.write(reinterpret_cast<char*>(&csx), sizeof(uint32_t));
    file.write(reinterpret_cast<char*>(&csy), sizeof(uint32_t));
    file.write(reinterpret_cast<char*>(&nf), sizeof(uint32_t));
    file.write(reinterpret_cast<char*>(&no), sizeof(uint32_t));
    
    float buf;
    char * buf_p = reinterpret_cast<char*>(&buf);
    uint32_t i, j, k;
    // Write mean
    for (i = 0; i < this->getNumFeatures(); i++)
    {
        buf = htole32(static_cast<float>(this->mean(i)));
        file.write(buf_p, sizeof(float));
    }
    
    // Write covariance
    for (i = 0; i < this->getNumOffsets(); i++)
        for (j = 0; j < this->getNumFeatures(); j++)
            for (k = 0; k < this->getNumFeatures(); k++)
            {
                buf = htole32(static_cast<float>(this->cov(i)(j,k)));
                file.write(buf_p, sizeof(float));
            }
    
    // Write offsets
    int32_t ibuf;
    buf_p = reinterpret_cast<char*>(&ibuf);
    for (i = 0; i < this->getNumOffsets(); i++)
    {
        ibuf = htole32(this->offsets(i,0));
        file.write(buf_p, sizeof(int32_t));
        ibuf = htole32(this->offsets(i,1));
        file.write(buf_p, sizeof(int32_t));
    }
    
    return file.good();
}

void StationaryBackground::clear()
{
    this->mean.resize(0);
    this->cov.resize(0);
    this->offsets.resize(0, 2);
    this->cellSize = Size();
}

StationaryBackground::CovMatrixMatrix StationaryBackground::computeCovariance(const int rows, const int cols)
{
    if (rows <= 0 || cols <= 0)
        return CovMatrixMatrix();
    // Check if target matrix is not larger than the maximum offset
    if (max(rows, cols) > this->getMaxOffset() + 1)
        return CovMatrixMatrix();
    
    int n = rows * cols;
    CovMatrixMatrix result(n, n);
    int i1, i2, y1, x1, y2, x2, dy, dx, o;
    for (y1 = 0; y1 < rows; y1++)
        for (x1 = 0; x1 < cols; x1++)
        {
            i1 = y1 * cols + x1;
            for (o = 0; o < this->offsets.rows(); o++)
            {
                dx = this->offsets(o, 0);
                dy = this->offsets(o, 1);
                x2 = x1 + dx;
                y2 = y1 + dy;
                if (x2 >= 0 && x2 < cols && y2 >= 0 && y2 < rows)
                {
                    i2 = y2 * cols + x2;
                    result(i1, i2) = this->cov(o);
                }
                x2 = x1 - dx;
                y2 = y1 - dy;
                if (x2 >= 0 && x2 < cols && y2 >= 0 && y2 < rows)
                {
                    i2 = y2 * cols + x2;
                    result(i1, i2) = this->cov(o).transpose();
                }
            }
        }
    
    return result;
}

ScalarMatrix StationaryBackground::computeFlattenedCovariance(const int rows, const int cols, unsigned int features)
{
    // Check if number of features is at least as large as the number of features in this background model
    unsigned int ourFeatures = this->getNumFeatures();
    if (features == 0)
        features = ourFeatures;
    else if (features < ourFeatures)
        return ScalarMatrix();

    CovMatrixMatrix cov = this->computeCovariance(rows, cols);
    if (cov.size() == 0)
        return ScalarMatrix();
    
    unsigned int n = cov.rows() * features;
    ScalarMatrix flat(n, n);
    unsigned int i, j, k, l; // (i * features + j) gives the row, (k * features + l) the column of the flat matrix
    unsigned int p, q; // used for iterative access to the rows and columns of flat matrix to not evaluate the above expression every time
    for (i = 0, p = 0; i < cov.rows(); i++)
        for (j = 0; j < features; j++, p++)
            for (k = 0, q = 0; k < cov.cols(); k++)
                for (l = 0; l < features; l++, q++)
                    flat(p, q) = (j < ourFeatures && l < ourFeatures) ? cov(i, k)(j, l) : 0.0f;
    
    // Make sure the returned matrix is close to symmetric
    assert((flat - flat.transpose()).cwiseAbs().maxCoeff() < 1e-5);
    return (flat + flat.transpose()) / 2;
}

void StationaryBackground::learnMean(ImageIterator & imgIt, const unsigned int numImages, ProgressCallback progressCB, void * cbData)
{
    // Initialize member variables
    this->cellSize = this->m_featureExtractor->cellSize();
    
    // Iterate over the images and compute the mean feature vector
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(this->m_featureExtractor->numRelevantFeatures());
    int i;
    vector<FeatureMatrix>::const_iterator levelIt;
    unsigned long long numSamples = 0;
    for (imgIt.rewind(); imgIt.ready() && (numImages == 0 || (unsigned int) imgIt < numImages); ++imgIt)
    {
        if (progressCB != NULL && numImages > 0 && !progressCB((unsigned int) imgIt, numImages, cbData))
            break;
        JPEGImage img = (*imgIt).getImage();
        if (!img.empty())
        {
            FeaturePyramid pyra(img, this->m_featureExtractor);
            // Loop over various scales
            for (levelIt = pyra.levels().begin(); levelIt != pyra.levels().end(); levelIt++)
            {
                for (i = 0; i < levelIt->numCells(); i++)
                    mean += Eigen::VectorXd(levelIt->cell(i).head(mean.size()).cast<double>());
                numSamples += levelIt->numCells();
            }
        }
    }
    if (progressCB != NULL && numImages > 0)
        progressCB(numImages, numImages, cbData);
    mean /= static_cast<double>(numSamples);
    
    // Store computed mean
    this->mean = mean.cast<FeatureScalar>();
}

void StationaryBackground::learnCovariance(ImageIterator & imgIt, const unsigned int numImages, const unsigned int maxOffset,
                                           ProgressCallback progressCB, void * cbData)
{
    int numFeat = this->m_featureExtractor->numRelevantFeatures();
    if (this->mean.size() < numFeat)
        return;
    
    // Local declarations and variables
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DoubleCovMatrix;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf;
    typedef Eigen::Matrix<complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXcf;
    int i, j, l, o, cx, cy, p1, p2;
    unsigned int minLevelSize = maxOffset * 2;
    vector<FeatureMatrix>::iterator levelIt;
    
    // Load wisdom for FFTW
    FILE * wisdom_file = fopen("wisdom.fftw", "r");
    if (wisdom_file)
    {
        fftwf_import_wisdom_from_file(wisdom_file);
        fclose(wisdom_file);
    }
    
    // Initialize member variables
    this->cellSize = this->m_featureExtractor->cellSize();
    this->makeOffsetArray(maxOffset);
    
    // Iterate over the images and compute the autocorrelation function
    Eigen::Array< DoubleCovMatrix, Eigen::Dynamic, 1 > cov(this->offsets.rows());
    cov.setConstant(DoubleCovMatrix::Zero(numFeat, numFeat));
    Eigen::Matrix<unsigned long long, Eigen::Dynamic, 1> numSamples(cov.size());
    numSamples.setZero();
    for (imgIt.rewind(); imgIt.ready() && (numImages == 0 || (unsigned int) imgIt < numImages); ++imgIt)
    {
        if (progressCB != NULL && numImages > 0 && !progressCB((unsigned int) imgIt, numImages, cbData))
            break;
        JPEGImage img = (*imgIt).getImage();
        if (!img.empty())
        {
            FeaturePyramid pyra(img, this->m_featureExtractor);
            // Subtract mean from all features
            #pragma omp parallel for private(l, i)
            for (l = 0; l < pyra.levels().size(); l++)
                for (i = 0; i < pyra.levels()[l].numCells(); i++)
                    pyra.levels()[l].cell(i).head(numFeat) -= this->mean;
            // Loop over various scales and compute covariances
            for (levelIt = pyra.levels().begin(); levelIt != pyra.levels().end(); levelIt++)
            {
                // Initialize matrices for transformations and correlations
                // (Some of them are just dummies for the planner, because we will use thread-local variables later on.)
                MatrixXcf freq(levelIt->rows() * numFeat, levelIt->cols() / 2 + 1);
                MatrixXcf powerSpectrum(levelIt->rows(), freq.cols());
                MatrixXf correlations(levelIt->rows(), levelIt->cols());
                // Plan fourier transforms
                fftwf_plan ft_forwards, ft_inverse;
                {
                    int size[2] = {static_cast<int>(levelIt->rows()), static_cast<int>(levelIt->cols())};
                    FeatureMatrix tmp(*levelIt); // backup data of the level
                    ft_forwards = fftwf_plan_many_dft_r2c(
                        2, size, numFeat,
                        levelIt->raw(), NULL, levelIt->channels(), 1,
                        reinterpret_cast<fftwf_complex*>(freq.data()), NULL, 1, size[0] * (size[1] / 2 + 1), FFTW_ESTIMATE
                    );
                    ft_inverse = fftwf_plan_dft_c2r_2d(
                        size[0], size[1],
                        reinterpret_cast<fftwf_complex*>(powerSpectrum.data()), correlations.data(), FFTW_MEASURE
                    );
                    *levelIt = tmp; // restore level, since FFTW may have changed it
                }
                //Compute covariances for each pair of levels using the power spectrum
                fftwf_execute(ft_forwards);
                cy = correlations.rows() / 2;
                cx = correlations.cols() / 2;
                #pragma omp parallel for private(p1, p2, o, i, j)
                for (p1 = 0; p1 < numFeat; p1++)
                {
                    MatrixXcf conjLevel = freq.block(p1 * levelIt->rows(), 0, levelIt->rows(), freq.cols()).conjugate();
                    MatrixXcf powerSpect(levelIt->rows(), freq.cols());
                    MatrixXf corr(correlations.rows(), correlations.cols());
                    for (p2 = 0; p2 < numFeat; p2++)
                    {
                        powerSpect = conjLevel.cwiseProduct(freq.block(p2 * levelIt->rows(), 0, levelIt->rows(), freq.cols()));
                        fftwf_execute_dft_c2r(ft_inverse, reinterpret_cast<fftwf_complex*>(powerSpect.data()), corr.data());
                        corr = fftshift(corr);
                        // Read out the correlations that belong to the offsets we need
                        for (o = 0; o < cov.size(); o++)
                        {
                            i = cy + this->offsets(o, 1);
                            j = cx + this->offsets(o, 0);
                            if (i >= 0 && j >= 0 && i < corr.rows() && j < corr.cols())
                            {
                                cov(o)(p1, p2) += static_cast<double>(corr(i, j))
                                                  / levelIt->numCells(); // division necessary, since FFTW computes an unnormalized DFT
                                if (p1 == 0 && p2 == 0)
                                    numSamples(o) += levelIt->numCells();
                            }
                        }
                    }
                }
                fftwf_destroy_plan(ft_forwards);
                fftwf_destroy_plan(ft_inverse);
            }
        }
    }
    if (progressCB != NULL && numImages > 0)
        progressCB(numImages, numImages, cbData);
    
    // Normalize and store computed autocorrelation function
    this->learnedAllOffsets = true;
    this->cov.resize(cov.size());
    for (o = 0; o < cov.size(); o++)
        if (numSamples(o) > 0)
            this->cov(o) = (cov(o) / static_cast<double>(numSamples(o))).cast<float>();
        else
        {
            this->cov(o) = CovMatrix::Zero(numFeat, numFeat);
            this->learnedAllOffsets = false;
        }
    
    // Save FFTW wisdom
    wisdom_file = fopen("wisdom.fftw", "w");
    if (wisdom_file)
    {
        fftwf_export_wisdom_to_file(wisdom_file);
        fclose(wisdom_file);
    }
}

void StationaryBackground::learnCovariance_accurate(ImageIterator & imgIt, const unsigned int numImages, const unsigned int maxOffset,
                                                    ProgressCallback progressCB, void * cbData)
{
    int numFeat = this->m_featureExtractor->numRelevantFeatures();
    if (this->mean.size() < numFeat)
        return;

    // Local declarations and variables
    typedef FeatureMatrix_<double> DoubleFeatureMatrix;
    typedef DoubleFeatureMatrix::ScalarMatrix DoubleCovMatrix;
    int i, j, l, dx, dy, o, t, x1, x2, y1, y2;
    vector<DoubleFeatureMatrix> levels;
    vector<DoubleFeatureMatrix>::const_iterator dLevelIt;
    
    // Initialize member variables
    this->cellSize = this->m_featureExtractor->cellSize();
    this->makeOffsetArray(maxOffset);
    
    // Iterate over the images and compute the autocorrelation function
    Eigen::Array< DoubleCovMatrix, Eigen::Dynamic, 1 > cov(this->offsets.rows());
    cov.setConstant(DoubleCovMatrix::Zero(numFeat, numFeat));
    Eigen::Matrix<unsigned long long, Eigen::Dynamic, 1> numSamples(cov.size());
    numSamples.setZero();
    for (imgIt.rewind(); imgIt.ready() && (numImages == 0 || (unsigned int) imgIt < numImages); ++imgIt)
    {
        if (progressCB != NULL && numImages > 0 && !progressCB((unsigned int) imgIt, numImages, cbData))
            break;
        JPEGImage img = (*imgIt).getImage();
        if (!img.empty())
        {
            FeaturePyramid pyra(img, this->m_featureExtractor);
            levels.resize(pyra.levels().size(), DoubleFeatureMatrix());
            // Subtract mean from all features
            #pragma omp parallel for private(l, i)
            for (l = 0; l < pyra.levels().size(); l++)
            {
                levels[l].resize(pyra.levels()[l].rows(), pyra.levels()[l].cols(), numFeat);
                for (i = 0; i < levels[l].numCells(); i++)
                    levels[l].cell(i) = (pyra.levels()[l].cell(i).head(numFeat) - this->mean).cast<double>();
            }
            // Loop over various scales and compute covariances
            for (dLevelIt = levels.begin(); dLevelIt != levels.end(); dLevelIt++)
            {
                #pragma omp parallel for private(o, dx, dy, y1, y2, x1, x2, i, j, l, t)
                for (o = 0; o < this->offsets.rows(); o++)
                {
                    dx = this->offsets(o, 0);
                    dy = this->offsets(o, 1);
                    if (dy > 0)
                    {
                        y1 = 0;
                        y2 = dLevelIt->rows() - 1 - dy;
                    }
                    else
                    {
                        y1 = -dy;
                        y2 = dLevelIt->rows() - 1;
                    }
                    if (dx > 0)
                    {
                        x1 = 0;
                        x2 = dLevelIt->cols() - 1 - dx;
                    }
                    else
                    {
                        x1 = -dx;
                        x2 = dLevelIt->cols() - 1;
                    }
                    
                    if (y2 >= y1 && x2 >= x1)
                    {
                        assert(y1 >= 0 && y2 < dLevelIt->rows() && x1 + dx >= 0 && x2 + dx < dLevelIt->cols());
                        t = (y2 - y1 + 1) * (x2 - x1 + 1);
                        DoubleCovMatrix feat1(t, numFeat);
                        DoubleCovMatrix feat2(t, numFeat);
                        for (i = y1, l = 0; i <= y2; i++)
                            for (j = x1; j <= x2; j++, l++)
                            {
                                feat1.row(l) = (*dLevelIt)(i, j).transpose();
                                feat2.row(l) = (*dLevelIt)(i + dy, j + dx).transpose();
                            }
                        cov(o).noalias() += feat1.transpose() * feat2;
                        numSamples(o) += t;
                    }
                }
            }
        }
    }
    if (progressCB != NULL && numImages > 0)
        progressCB(numImages, numImages, cbData);
    
    // Normalize and store computed autocorrelation function
    this->learnedAllOffsets = true;
    this->cov.resize(cov.size());
    for (o = 0; o < cov.size(); o++)
        if (numSamples(o) > 0)
            this->cov(o) = (cov(o) / static_cast<double>(numSamples(o))).cast<float>();
        else
        {
            this->cov(o) = CovMatrix::Zero(numFeat, numFeat);
            this->learnedAllOffsets = false;
        }
}

void StationaryBackground::makeOffsetArray(const unsigned int maxOffset)
{
    this->offsets.resize(maxOffset * (maxOffset + 1) * 2 + 1, 2);
    int dx, dy, o;
    for (dx = 0, o = 0; dx <= maxOffset; dx++)
        for (dy = 0; dy <= maxOffset; dy++)
        {
            this->offsets(o, 0) = dx;
            this->offsets(o, 1) = dy;
            o++;
            if (dx > 0 && dy > 0)
            {
                this->offsets(o, 0) = dx;
                this->offsets(o, 1) = -dy;
                o++;
            }
        }
}


template<class Derived>
static typename Derived::PlainObject fftshift(const Eigen::MatrixBase<Derived> & mat)
{
    typename Derived::Index rows = mat.rows(), cols = mat.cols(), halfRows = rows / 2, halfCols = cols / 2;
    typename Derived::PlainObject shifted(rows, cols);
    shifted.topLeftCorner    (       halfRows,        halfCols) = mat.bottomRightCorner(       halfRows,        halfCols);
    shifted.topRightCorner   (       halfRows, cols - halfCols) = mat.bottomLeftCorner (       halfRows, cols - halfCols);
    shifted.bottomLeftCorner (rows - halfRows,        halfCols) = mat.topRightCorner   (rows - halfRows,        halfCols);
    shifted.bottomRightCorner(rows - halfRows, cols - halfCols) = mat.topLeftCorner    (rows - halfRows, cols - halfCols);
    return shifted;
}
