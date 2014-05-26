#include "StationaryBackground.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "portable_endian.h"
using namespace ARTOS;
using namespace std;

StationaryBackground::StationaryBackground(const unsigned int numOffsets, const unsigned int numFeatures, const unsigned int cellSize)
: mean(numFeatures), cov(numOffsets), offsets(numOffsets, 2), cellSize(cellSize)
{
    this->cov.setConstant(CovMatrix(numFeatures, numFeatures));
}

bool StationaryBackground::readFromFile(const string & filename)
{
    ifstream file(filename.c_str(), ifstream::in | ifstream::binary);
    if (!file.is_open())
        return false;
    
    // Read parameters (cell size, number of features, number of offsets) from file
    unsigned int cs, nf, no;
    file.read(reinterpret_cast<char*>(&cs), 4);
    file.read(reinterpret_cast<char*>(&nf), 4);
    file.read(reinterpret_cast<char*>(&no), 4);
    cs = le32toh(cs);
    nf = le32toh(nf);
    no = le32toh(no);
    if (cs == 0 || nf == 0 || no == 0)
        return false;
    
    // Initialize matrices and arrays
    this->cellSize = cs;
    this->mean.resize(nf);
    this->cov.resize(no);
    this->offsets.resize(no, Eigen::NoChange);
    
    float buf;
    char * buf_p = reinterpret_cast<char*>(&buf);
    unsigned int i, j, k;
    // Read mean
    for (i = 0; i < nf; i++)
    {
        file.read(buf_p, 4);
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
                file.read(buf_p, 4);
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
    int ibuf;
    buf_p = reinterpret_cast<char*>(&ibuf);
    for (i = 0; i < no; i++)
    {
        file.read(buf_p, 4);
        if (!file.good())
        {
            this->clear();
            return false;
        }
        this->offsets(i,0) = le32toh(ibuf);
        file.read(buf_p, 4);
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
    
    // Write parameters (cell size, number of features, number of offsets)
    unsigned int cs, nf, no;
    cs = htole32(this->cellSize);
    no = htole32(this->getNumFeatures());
    nf = htole32(this->getNumOffsets());
    file.write(reinterpret_cast<char*>(&cs), 4);
    file.write(reinterpret_cast<char*>(&nf), 4);
    file.write(reinterpret_cast<char*>(&no), 4);
    
    float buf;
    char * buf_p = reinterpret_cast<char*>(&buf);
    unsigned int i, j, k;
    // Write mean
    for (i = 0; i < this->getNumFeatures(); i++)
    {
        buf = htole32(this->mean(i));
        file.write(buf_p, 4);
    }
    
    // Write covariance
    for (i = 0; i < this->getNumOffsets(); i++)
        for (j = 0; i < this->getNumFeatures(); j++)
            for (k = 0; i < this->getNumFeatures(); k++)
            {
                buf = htole32(this->cov(i)(j,k));
                file.write(buf_p, 4);
            }
    
    // Write offsets
    int ibuf;
    buf_p = reinterpret_cast<char*>(&ibuf);
    for (i = 0; i < this->getNumOffsets(); i++)
    {
        ibuf = htole32(this->offsets(i,0));
        file.write(buf_p, 4);
        ibuf = htole32(this->offsets(i,1));
        file.write(buf_p, 4);
    }
    
    return file.good();
}

void StationaryBackground::clear()
{
    this->mean.resize(0);
    this->cov.resize(0);
    this->offsets.resize(0, 2);
    this->cellSize = 0;
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

StationaryBackground::Matrix StationaryBackground::computeFlattenedCovariance(const int rows, const int cols, unsigned int features)
{
    // Check if number of features is at least as large as the number of features in this background model
    unsigned int ourFeatures = this->getNumFeatures();
    if (features == 0)
        features = ourFeatures;
    else if (features < ourFeatures)
        return Matrix();

    CovMatrixMatrix cov = this->computeCovariance(rows, cols);
    if (cov.size() == 0)
        return Matrix();
    
    unsigned int n = cov.rows() * features;
    Matrix flat(n, n);
    unsigned int i, j, k, l; // (i * features + j) gives the row, (k * features + l) the column of the flat matrix
    unsigned int p, q; // used for iterative access to the rows and columns of flat matrix to not evaluate the above expression every time
    for (i = 0, p = 0; i < cov.rows(); i++)
        for (j = 0; j < features; j++, p++)
            for (k = 0, q = 0; k < cov.cols(); k++)
                for (l = 0; l < features; l++, q++)
                    flat(p, q) = (j < ourFeatures && l < ourFeatures) ? cov(i, k)(j, l) : 0.0f;
    
    // Make sure the returned matrix is close to symmetric
    if ((flat - flat.transpose()).cwiseAbs().sum() > 1e-5)
        cerr << "Covariance matrix is not symmetric!" << endl;
    
    return (flat + flat.transpose()) / 2;
}
