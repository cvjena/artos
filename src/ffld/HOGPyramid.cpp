//--------------------------------------------------------------------------------------------------
// Implementation of the paper "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012.
//
// Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLD (the Fast Fourier Linear Detector)
//
// FFLD is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
//
// FFLD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with FFLD. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#include "HOGPyramid.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;
using namespace FFLD;
using namespace std;

HOGPyramid::HOGPyramid() : padx_(0), pady_(0), interval_(0)
{
}

HOGPyramid::HOGPyramid(int padx, int pady, int interval, const vector<Level> & levels) : padx_(0),
pady_(0), interval_(0)
{
	if ((padx < 1) || (pady < 1) || (interval < 1))
		return;
	
	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_ = levels;
}

HOGPyramid::HOGPyramid(const JPEGImage & image, int padx, int pady, int interval) : padx_(0),
pady_(0), interval_(0)
{
	if (image.empty() || (padx < 1) || (pady < 1) || (interval < 1))
		return;
	
	// Copmute the number of scales such that the smallest size of the last level is 5
	const int maxScale = ceil(log(min(image.width(), image.height()) / 40.0) / log(2.0)) * interval;
	
	// Cannot compute the pyramid on images too small
	if (maxScale < interval)
		return;
	
	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_.resize(maxScale + 1);
	
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < interval; ++i) {
		double scale = pow(2.0, static_cast<double>(-i) / interval);
		
		JPEGImage scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
		
		// First octave at twice the image resolution
#ifndef FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES
		Hog(scaled, levels_[i], padx, pady, 4);
		
		// Second octave at the original resolution
		if (i + interval <= maxScale)
			Hog(scaled, levels_[i + interval], padx, pady, 8);
		
		// Remaining octaves
		for (int j = 2; i + j * interval <= maxScale; ++j) {
			scale *= 0.5;
			scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
			Hog(scaled, levels_[i + j * interval], padx, pady, 8);
		}
#else
		Hog(scaled.scanLine(0), scaled.width(), scaled.height(), scaled.depth(), levels_[i], 4);
		
		// Second octave at the original resolution
		if (i + interval <= maxScale)
			Hog(scaled.scanLine(0), scaled.width(), scaled.height(), scaled.depth(),
				levels_[i + interval], 8);
		
		// Remaining octaves
		for (int j = 2; i + j * interval <= maxScale; ++j) {
			scale *= 0.5;
			scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
			Hog(scaled.scanLine(0), scaled.width(), scaled.height(), scaled.depth(),
				levels_[i + j * interval], 8);
		}
#endif
	}
	
	// Add padding
#ifdef FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES
	for (int i = 0; i <= maxScale; ++i) {
		Level tmp = Level::Constant(levels_[i].rows() + (pady + 1) * 2,
									levels_[i].cols() + (padx + 1) * 2, Cell::Zero());
		
		// Set the last feature to 1
		for (int y = 0; y < tmp.rows(); ++y)
			for (int x = 0; x < tmp.cols(); ++x)
				tmp(y, x)(31) = 1;
		
		tmp.block(pady + 1, padx + 1, levels_[i].rows(), levels_[i].cols()) = levels_[i];
		
		levels_[i].swap(tmp);
	}
#endif
}

int HOGPyramid::padx() const
{
	return padx_;
}

int HOGPyramid::pady() const
{
	return pady_;
}

int HOGPyramid::interval() const
{
	return interval_;
}

const vector<HOGPyramid::Level> & HOGPyramid::levels() const
{
	return levels_;
}

bool HOGPyramid::empty() const
{
	return levels().empty();
}

void HOGPyramid::convolve(const Level & filter, vector<Matrix> & convolutions) const
{
	// Resize convolutions to hold # levels
	convolutions.resize(levels_.size());
	
	// For each level
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < levels_.size(); ++i)
		Convolve(levels_[i], filter, convolutions[i]);
}

void HOGPyramid::convolve(const Level & filter, vector<SparseMatrix> & convolutions) const
{
	// Resize convolutions to hold # levels
	convolutions.resize(levels_.size());
	
	// For each level
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < levels_.size(); ++i)
		Convolve(levels_[i], filter, convolutions[i]);
}

void HOGPyramid::convolve(const vector<Matrix> & labels, Level & sum) const
{
	// Nothing to do if the levels or the labels are empty
	if (empty() || labels.empty()) {
		sum = Level();
		return;
	}
	
	// Resize sum to the filter size
	sum = Level::Constant(levels_[0].rows() - labels[0].rows() + 1,
						  levels_[0].cols() - labels[0].cols() + 1, Cell::Zero());
	
	// For each level
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < min(levels_.size(), labels.size()); ++i) {
		Level tmp;
		Convolve(levels_[i], labels[i], tmp);
		
		if (tmp.size())
#pragma omp critical
			sum += tmp;
	}
}

void HOGPyramid::convolve(const vector<SparseMatrix> & labels, Level & sum) const
{
	// Nothing to do if the levels or the labels are empty
	if (empty() || labels.empty()) {
		sum = Level();
		return;
	}
	
	// Resize sum to the filter size
	sum = Level::Constant(levels_[0].rows() - labels[0].rows() + 1,
						  levels_[0].cols() - labels[0].cols() + 1, Cell::Zero());
	
	// For each level
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < min(levels_.size(), labels.size()); ++i) {
		Level tmp;
		Convolve(levels_[i], labels[i], tmp);
		
		if (tmp.size())
#pragma omp critical
			sum += tmp;
	}
}

Map<HOGPyramid::Matrix, Aligned> HOGPyramid::Convert(Level & level)
{
	return Map<Matrix, Aligned>(level.data()->data(), level.rows(),
											  level.cols() * NbFeatures);
}

Map<const HOGPyramid::Matrix, Aligned> HOGPyramid::Convert(const Level & level)
{
	return Map<const Matrix, Aligned>(level.data()->data(), level.rows(),
													level.cols() * NbFeatures);
}

FFLD::HOGPyramid::Level HOGPyramid::Flip(const HOGPyramid::Level & filter)
{
	// Symmetric features
	const int symmetry[NbFeatures] = {
		9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
		18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
		28, 27, 30, 29, // Texture
		31 // Truncation
	};
	
	// Symmetric filter
	HOGPyramid::Level result(filter.rows(), filter.cols());
	
	for (int y = 0; y < filter.rows(); ++y)
		for (int x = 0; x < filter.cols(); ++x)
			for (int i = 0; i < NbFeatures; ++i)
				result(y, x)(i) = filter(y, filter.cols() - 1 - x)(symmetry[i]);
	
	return result;
}

#ifndef FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES
namespace FFLD
{
namespace detail
{
// Bilinear interpolation among the 4 neighboring cells
template <class Matrix, int CellSize>
	inline void interpolate(int x, int y, int bin0, int bin1, HOGPyramid::Scalar magnitude0,
							HOGPyramid::Scalar magnitude1, Matrix & matrix)
{
	// Find the bin into which (x, y) falls
	const int i = (y - CellSize / 2) / CellSize;
	const int j = (x - CellSize / 2) / CellSize;
	const int k = (y - CellSize / 2) & (CellSize - 1);
	const int l = (x - CellSize / 2) & (CellSize - 1);
	
	// Bilinear interpolation
	const int a = k * 2 + 1;
	const int b = CellSize * 2 - a;
	const int c = l * 2 + 1;
	const int d = CellSize * 2 - c;
	
	matrix(i    , j    )(bin0) += magnitude0 * (b * d);
	matrix(i    , j    )(bin1) += magnitude1 * (b * d);
	matrix(i    , j + 1)(bin0) += magnitude0 * (b * c);
	matrix(i    , j + 1)(bin1) += magnitude1 * (b * c);
	matrix(i + 1, j    )(bin0) += magnitude0 * (a * d);
	matrix(i + 1, j    )(bin1) += magnitude1 * (a * d);
	matrix(i + 1, j + 1)(bin0) += magnitude0 * (a * c);
	matrix(i + 1, j + 1)(bin1) += magnitude1 * (a * c);
}
}
}

void HOGPyramid::Hog(const JPEGImage & image, Level & level, int padx, int pady,
					 int cellSize)
{
	// Table of all the possible tangents (1MB)
	static Scalar ATAN2_TABLE[512][512] = {{0}};
	
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
	
	// Get all the image members
	const int width = image.width();
	const int height = image.height();
	const int depth = image.depth();
	
	// Make sure the image is big enough
	assert(width >= cellSize / 2);
	assert(height >= cellSize / 2);
	assert(depth >= 1);
	assert(padx >= 1);
	assert(pady >= 1);
	assert((cellSize == 8) || (cellSize == 4));
	
	// Resize the feature matrix
	level = Level::Constant((height + cellSize / 2) / cellSize + pady * 2,
							(width + cellSize / 2) / cellSize + padx * 2, Cell::Zero());
	
	for (int y = 0; y < height; ++y) {
		const int yp = min(y + 1, height - 1);
		const int ym = max(y - 1, 0);
		
		const uint8_t * linep = reinterpret_cast<const uint8_t *>(image.scanLine(yp));
		const uint8_t * line = reinterpret_cast<const uint8_t *>(image.scanLine(y));
		const uint8_t * linem = reinterpret_cast<const uint8_t *>(image.scanLine(ym));
		
		for (int x = 0; x < width; ++x) {
			const int xp = min(x + 1, width - 1);
			const int xm = max(x - 1, 0);
			
			// Use the channel with the largest gradient magnitude
			Scalar magnitude = 0;
			Scalar theta = 0;
			
			for (int i = 0; i < depth; ++i) {
				const int dx = static_cast<int>(line[xp * depth + i]) -
							   static_cast<int>(line[xm * depth + i]);
				const int dy = static_cast<int>(linep[x * depth + i]) -
							   static_cast<int>(linem[x * depth + i]);
				
				if (dx * dx + dy * dy > magnitude) {
					magnitude = dx * dx + dy * dy;
					theta = ATAN2_TABLE[dy + 255][dx + 255];
				}
			}
			
			magnitude = sqrt(magnitude);
			
			// Bilinear interpolation
			const int theta0 = theta;
			const int theta1 = (theta0 < 17) ? (theta0 + 1) : 0;
			const Scalar alpha = theta - theta0;
			
			if (cellSize == 8)
				detail::interpolate<Level, 8>(x + padx * cellSize, y + pady * cellSize,
												theta0, theta1, magnitude * (1 - alpha),
												magnitude * alpha, level);
			else // cellSize == 4
				detail::interpolate<Level, 4>(x + padx * cellSize, y + pady * cellSize,
												theta0, theta1, magnitude * (1 - alpha),
												magnitude * alpha, level);
		}
	}
	
	// Compute the "gradient energy" of each cell, i.e. ||C(i,j)||^2
	for (int y = 0; y < level.rows(); ++y) {
		for (int x = 0; x < level.cols(); ++x) {
			Scalar sumSq = 0;
			
			for (int i = 0; i < 9; ++i)
				sumSq += (level(y, x)(i) + level(y, x)(i + 9)) *
						 (level(y, x)(i) + level(y, x)(i + 9));
			
			level(y, x)(NbFeatures - 1) = sumSq;
		}
	}
	
	// Compute the four normalization factors then normalize and clamp everything
	const Scalar EPS = numeric_limits<Scalar>::epsilon();
	
	for (int y = pady; y < level.rows() - pady; ++y) {
		for (int x = padx; x < level.cols() - padx; ++x) {
			// Normalization factors
			const Scalar n0 = 1 / sqrt(level(y - 1, x - 1)(NbFeatures - 1) +
									   level(y - 1, x    )(NbFeatures - 1) +
									   level(y    , x - 1)(NbFeatures - 1) +
									   level(y    , x    )(NbFeatures - 1) + EPS);
			const Scalar n1 = 1 / sqrt(level(y - 1, x    )(NbFeatures - 1) +
									   level(y - 1, x + 1)(NbFeatures - 1) +
									   level(y    ,     x)(NbFeatures - 1) +
									   level(y    , x + 1)(NbFeatures - 1) + EPS);
			const Scalar n2 = 1 / sqrt(level(y    , x - 1)(NbFeatures - 1) +
									   level(y    , x    )(NbFeatures - 1) +
									   level(y + 1, x - 1)(NbFeatures - 1) +
									   level(y + 1, x    )(NbFeatures - 1) + EPS);
			const Scalar n3 = 1 / sqrt(level(y    , x    )(NbFeatures - 1) +
									   level(y    , x + 1)(NbFeatures - 1) +
									   level(y + 1, x    )(NbFeatures - 1) +
									   level(y + 1, x + 1)(NbFeatures - 1) + EPS);
			
			// Contrast-insensitive features
			for (int i = 0; i < 9; ++i) {
				const Scalar sum = level(y, x)(i) + level(y, x)(i + 9);
				const Scalar h0 = min(sum * n0, Scalar(0.2));
				const Scalar h1 = min(sum * n1, Scalar(0.2));
				const Scalar h2 = min(sum * n2, Scalar(0.2));
				const Scalar h3 = min(sum * n3, Scalar(0.2));
				level(y, x)(i + 18) = (h0 + h1 + h2 + h3) * Scalar(0.5);
			}
			
			// Contrast-sensitive features
			Scalar t0 = 0;
			Scalar t1 = 0;
			Scalar t2 = 0;
			Scalar t3 = 0;
			
			for (int i = 0; i < 18; ++i) {
				const Scalar sum = level(y, x)(i);
				const Scalar h0 = min(sum * n0, Scalar(0.2));
				const Scalar h1 = min(sum * n1, Scalar(0.2));
				const Scalar h2 = min(sum * n2, Scalar(0.2));
				const Scalar h3 = min(sum * n3, Scalar(0.2));
				level(y, x)(i) = (h0 + h1 + h2 + h3) * Scalar(0.5);
				t0 += h0;
				t1 += h1;
				t2 += h2;
				t3 += h3;
			}
			
			// Texture features
			level(y, x)(27) = t0 * Scalar(0.2357);
			level(y, x)(28) = t1 * Scalar(0.2357);
			level(y, x)(29) = t2 * Scalar(0.2357);
			level(y, x)(30) = t3 * Scalar(0.2357);
		}
	}
	
	// Truncation features
	for (int y = 0; y < level.rows(); ++y) {
		for (int x = 0; x < level.cols(); ++x) {
			if ((y < pady) || (y >= level.rows() - pady) || (x < padx) ||
				(x >= level.cols() - padx)) {
				level(y, x).setZero();
				level(y, x)(NbFeatures - 1) = 1;
			}
			else {
				level(y, x)(NbFeatures - 1) = 0;
			}
		}
	}
}
#else
void HOGPyramid::Hog(const uint8_t * bits, int width, int height, int depth, Level & level,
					 int cellSize)
{
	// Adapted from voc-release4.01/features.cc
	const Scalar EPS = 0.0001;
	
	const Scalar UU[9] = {
		1.0000, 0.9397, 0.7660, 0.5000, 0.1736,-0.1736,-0.5000,-0.7660,-0.9397
	};
	
	const Scalar VV[9] = {
		0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420
	};
	
	// Make sure all sizes are strictly positive
	assert(width > 0);
	assert(height > 0);
	assert(depth > 0);
	assert(cellSize > 0);
	
	// Memory for caching orientation histograms & their norms
	int blocks[2];
	blocks[0] = static_cast<double>(height) / cellSize + 0.5;
	blocks[1] = static_cast<double>(width) / cellSize + 0.5;
	MatrixXf hist = MatrixXf::Zero(blocks[0], blocks[1] * 18);
	MatrixXf norm = MatrixXf::Zero(blocks[0], blocks[1]);
	
	// Memory for HOG features
	int out[3];
	out[0] = max(blocks[0] - 2, 0);
	out[1] = max(blocks[1] - 2, 0);
	out[2] = 27 + 4 + 1;
	level.resize(out[0], out[1]);
	
	int visible[2];
	visible[0] = blocks[0] * cellSize;
	visible[1] = blocks[1] * cellSize;
	
	for (int y = 1; y < visible[0] - 1; ++y) {
		for (int x = 1; x < visible[1] - 1; ++x) {
			const int x2 = min(x, width - 2);
			const int y2 = min(y, height - 2);
			
			// Use the channel with the largest gradient magnitude
			Scalar magnitude = 0;
			int argDx = 0;
			int argDy = 0;
			
			for (int i = 0; i < depth; ++i) {
				const int dx = static_cast<int>(bits[(y2 * width + x2 + 1) * depth + i]) -
							   static_cast<int>(bits[(y2 * width + x2 - 1) * depth + i]);
				const int dy = static_cast<int>(bits[((y2 + 1) * width + x2) * depth + i]) -
							   static_cast<int>(bits[((y2 - 1) * width + x2) * depth + i]);
				
				if (dx * dx + dy * dy > magnitude) {
					magnitude = dx * dx + dy * dy;
					argDx = dx;
					argDy = dy;
				}
			}
			
			// Snap to one of 18 orientations
			int theta = 0;
			Scalar best = 0;
			
			for (int i = 0; i < 9; ++i) {
				const Scalar dot = UU[i] * argDx + VV[i] * argDy;
				
				if (dot > best) {
					best = dot;
					theta = i;
				}
				else if (-dot > best) {
					best = -dot;
					theta = i + 9;
				}
			}
			
			// Add to 4 histograms around pixel using linear interpolation
			Scalar xp = (x + Scalar(0.5)) / cellSize - Scalar(0.5);
			Scalar yp = (y + Scalar(0.5)) / cellSize - Scalar(0.5);
			int ixp = floor(xp);
			int iyp = floor(yp);
			Scalar vx0 = xp - ixp;
			Scalar vy0 = yp - iyp;
			Scalar vx1 = 1 - vx0;
			Scalar vy1 = 1 - vy0;
			
			magnitude = sqrt(magnitude);
			
			if ((ixp >= 0) && (iyp >= 0))
				hist(iyp, ixp * 18 + theta) += vx1 * vy1 * magnitude;
			
			if ((ixp + 1 < blocks[1]) && (iyp >= 0))
				hist(iyp, (ixp + 1) * 18 + theta) += vx0 * vy1 * magnitude;
			
			if ((ixp >= 0) && (iyp + 1 < blocks[0]))
				hist(iyp + 1, ixp * 18 + theta) += vx1 * vy0 * magnitude;
			
			if ((ixp + 1 < blocks[1]) && (iyp + 1 < blocks[0]))
				hist(iyp + 1, (ixp + 1) * 18 + theta) += vx0 * vy0 * magnitude;
		}
	}
	
	// Compute energy in each block by summing over orientations
	for (int y = 0; y < blocks[0]; ++y) {
		for (int x = 0; x < blocks[1]; ++x) {
			Scalar sumSq = 0;
			
			for (int i = 0; i < 9; ++i)
				sumSq += (hist(y, x * 18 + i) + hist(y, x * 18 + i + 9)) *
						 (hist(y, x * 18 + i) + hist(y, x * 18 + i + 9));
			
			norm(y, x) = sumSq;
		}
	}
	
	// Compute features
	for (int y = 0; y < out[0]; ++y) {
		for (int x = 0; x < out[1]; ++x) {
			// Normalization factors
			const Scalar n0 = 1 / sqrt(norm(y    , x    ) + norm(y    , x + 1) +
											norm(y + 1, x    ) + norm(y + 1, x + 1) + EPS);
			const Scalar n1 = 1 / sqrt(norm(y    , x + 1) + norm(y    , x + 2) +
											norm(y + 1, x + 1) + norm(y + 1, x + 2) + EPS);
			const Scalar n2 = 1 / sqrt(norm(y + 1, x    ) + norm(y + 1, x + 1) +
											norm(y + 2, x    ) + norm(y + 2, x + 1) + EPS);
			const Scalar n3 = 1 / sqrt(norm(y + 1, x + 1) + norm(y + 1, x + 2) +
											norm(y + 2, x + 1) + norm(y + 2, x + 2) + EPS);
			
			// Contrast-insensitive features
			for (int i = 0; i < 9; ++i) {
				const Scalar sum = hist(y + 1, (x + 1) * 18 + i) +
								   hist(y + 1, (x + 1) * 18 + i + 9);
				const Scalar h0 = min(sum * n0, Scalar(0.2));
				const Scalar h1 = min(sum * n1, Scalar(0.2));
				const Scalar h2 = min(sum * n2, Scalar(0.2));
				const Scalar h3 = min(sum * n3, Scalar(0.2));
				level(y, x)(i + 18) = (h0 + h1 + h2 + h3) / 2;
			}
			
			// Contrast-sensitive features
			Scalar t0 = 0;
			Scalar t1 = 0;
			Scalar t2 = 0;
			Scalar t3 = 0;
			
			for (int i = 0; i < 18; ++i) {
				const Scalar sum = hist(y + 1, (x + 1) * 18 + i);
				const Scalar h0 = min(sum * n0, Scalar(0.2));
				const Scalar h1 = min(sum * n1, Scalar(0.2));
				const Scalar h2 = min(sum * n2, Scalar(0.2));
				const Scalar h3 = min(sum * n3, Scalar(0.2));
				level(y, x)(i) = (h0 + h1 + h2 + h3) / 2;
				t0 += h0;
				t1 += h1;
				t2 += h2;
				t3 += h3;
			}
			
			// Texture features
			level(y, x)(27) = t0 * Scalar(0.2357);
			level(y, x)(28) = t1 * Scalar(0.2357);
			level(y, x)(29) = t2 * Scalar(0.2357);
			level(y, x)(30) = t3 * Scalar(0.2357);
		}
	}
	
	// Truncation feature
	for (int y = 0; y < level.rows(); ++y)
		for (int x = 0; x < level.cols(); ++x)
			level(y, x)(31) = 0;
}
#endif

void HOGPyramid::Convolve(const Level & x, const Level & y, Matrix & z)
{
	// Nothing to do if x is smaller than y
	if ((x.rows() < y.rows()) || (x.cols() < y.cols())) {
		z = Matrix();
		return;
	}
	
	z = Matrix::Zero(x.rows() - y.rows() + 1, x.cols() - y.cols() + 1);
	
	for (int i = 0; i < z.rows(); ++i) {
		for (int j = 0; j < y.rows(); ++j) {
			const Map<const Matrix, Aligned, OuterStride<NbFeatures> >
				mapx(reinterpret_cast<const Scalar *>(x.row(i + j).data()), z.cols(),
					 y.cols() * NbFeatures);
#ifndef FFLD_HOGPYRAMID_DOUBLE
			const Map<const RowVectorXf, Aligned>
#else
			const Map<const RowVectorXd, Aligned>
#endif
				mapy(reinterpret_cast<const Scalar *>(y.row(j).data()), y.cols() * NbFeatures);
			
			z.row(i).noalias() += mapy * mapx.transpose();
		}
	}
}

void HOGPyramid::Convolve(const Level & x, const Level & y, SparseMatrix & z)
{
	// Nothing to do if x is smaller than y
	if ((x.rows() < y.rows()) || (x.cols() < y.cols())) {
		z = SparseMatrix();
		return;
	}
	
	assert(z.rows() == x.rows() - y.rows() + 1);
	assert(z.cols() == x.cols() - y.cols() + 1);
	
	// Iterate over the non-zero entries of the samples matrix
	for (int i = 0; i < z.rows(); ++i) {
		z.startVec(i);
		
		for (SparseMatrix::InnerIterator it(z, i); it; ++it) {
			Scalar dot = 0;
			
			for (int j = 0; j < y.rows(); ++j) {
#ifndef FFLD_HOGPYRAMID_DOUBLE
				const Map<const RowVectorXf, Aligned>
#else
				const Map<const RowVectorXd, Aligned>
#endif
					mapx(reinterpret_cast<const Scalar *>(x.row(i + j).data() + it.col()),
						 y.cols() * NbFeatures);
				
#ifndef FFLD_HOGPYRAMID_DOUBLE
				const Map<const RowVectorXf, Aligned>
#else
				const Map<const RowVectorXd, Aligned>
#endif
					mapy(reinterpret_cast<const Scalar *>(y.row(j).data()), y.cols() * NbFeatures);
				
				dot += mapx.dot(mapy);
			}
			
			it.valueRef() = dot;
		}
	}
	
	z.finalize();
}

void HOGPyramid::Convolve(const Level & x, const Matrix & z, Level & y)
{
	// Nothing to do if x is smaller than z
	if ((x.rows() < z.rows()) || (x.cols() < z.cols())) {
		y = Level();
		return;
	}
	
	y = Level::Constant(x.rows() - z.rows() + 1, x.cols() - z.cols() + 1, Cell::Zero());
	
	for (int i = 0; i < z.rows(); ++i) {
		for (int j = 0; j < y.rows(); ++j) {
			const Map<const Matrix, Aligned, OuterStride<NbFeatures> >
				mapx(reinterpret_cast<const Scalar *>(x.row(i + j).data()), z.cols(),
					 y.cols() * NbFeatures);
			
#ifndef FFLD_HOGPYRAMID_DOUBLE
			Map<RowVectorXf, Aligned>
#else
			Map<RowVectorXd, Aligned>
#endif
				mapy(reinterpret_cast<Scalar *>(y.row(j).data()), y.cols() * NbFeatures);
			
			mapy.noalias() += z.row(i) * mapx;
		}
	}
}

void HOGPyramid::Convolve(const Level & x, const SparseMatrix & z, Level & y)
{
	// Nothing to do if x is smaller than z
	if ((x.rows() < z.rows()) || (x.cols() < z.cols())) {
		y = Level();
		return;
	}
	
	const Map<const Matrix, Aligned>
		mapx(reinterpret_cast<const Scalar *>(x.data()), x.rows(), x.cols() * NbFeatures);
	
	y = Level::Constant(x.rows() - z.rows() + 1, x.cols() - z.cols() + 1, Cell::Zero());
	
	Map<Matrix, Aligned>
		mapy(reinterpret_cast<Scalar *>(y.data()), y.rows(), y.cols() * NbFeatures);
	
	// Iterate over the non-zero entries of the z matrix
	for (int i = 0; i < z.rows(); ++i)
		for (SparseMatrix::InnerIterator it(z, i); it; ++it)
			mapy.noalias() += it.value() * mapx.block(i, it.col() * NbFeatures,
													  mapy.rows(), mapy.cols());
}
