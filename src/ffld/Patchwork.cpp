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

#include "Patchwork.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <set>

using namespace Eigen;
using namespace FFLD;
using namespace std;

int Patchwork::MaxRows_(0);
int Patchwork::MaxCols_(0);
int Patchwork::HalfCols_(0);

#ifndef FFLD_HOGPYRAMID_DOUBLE
	fftwf_plan Patchwork::Forwards_(0);
	fftwf_plan Patchwork::Inverse_(0);
#else
	fftw_plan Patchwork::Forwards_(0);
	fftw_plan Patchwork::Inverse_(0);
#endif

Patchwork::Patchwork() : padx_(0), pady_(0), interval_(0)
{
}

Patchwork::Patchwork(const HOGPyramid & pyramid) : padx_(pyramid.padx()), pady_(pyramid.pady()),
interval_(pyramid.interval())
{
	// Remove the padding from the bottom/right sides since convolutions with Fourier wrap around
	const int nbLevels = pyramid.levels().size();
	
	rectangles_.resize(nbLevels);
	
	for (int i = 0; i < nbLevels; ++i) {
		rectangles_[i].first.setWidth(pyramid.levels()[i].cols() - padx_);
		rectangles_[i].first.setHeight(pyramid.levels()[i].rows() - pady_);
	}
	
	// Build the patchwork planes
	const int nbPlanes = BLF(rectangles_);
	
	// Constructs an empty patchwork in case of error
	if (nbPlanes <= 0)
		return;
	
	planes_.resize(nbPlanes);
	
	for (int i = 0; i < nbPlanes; ++i) {
		planes_[i] = Plane::Constant(MaxRows_, HalfCols_, Cell::Zero());
		
		Map<HOGPyramid::Level, Aligned>
			plane(reinterpret_cast<HOGPyramid::Cell *>(planes_[i].data()), MaxRows_, HalfCols_ * 2);
		
		// Set the last feature to 1
		for (int y = 0; y < MaxRows_; ++y)
			for (int x = 0; x < MaxCols_; ++x)
				plane(y, x)(HOGPyramid::NbFeatures - 1) = 1.0f;
	}
	
	// Recopy the pyramid levels into the planes
	for (int i = 0; i < nbLevels; ++i) {
		Map<HOGPyramid::Level, Aligned>
			plane(reinterpret_cast<HOGPyramid::Cell *>(planes_[rectangles_[i].second].data()),
				  MaxRows_, HalfCols_ * 2);
		
		plane.block(rectangles_[i].first.y(), rectangles_[i].first.x(),
					rectangles_[i].first.height(), rectangles_[i].first.width()) =
			pyramid.levels()[i].topLeftCorner(rectangles_[i].first.height(),
											  rectangles_[i].first.width());
	}
	
	// Transform the planes
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < nbPlanes; ++i)
#ifndef FFLD_HOGPYRAMID_DOUBLE
		fftwf_execute_dft_r2c(Forwards_, reinterpret_cast<float *>(planes_[i].data()->data()),
							  reinterpret_cast<fftwf_complex *>(planes_[i].data()->data()));
#else
		fftw_execute_dft_r2c(Forwards_, reinterpret_cast<double *>(planes_[i].data()->data()),
							 reinterpret_cast<fftw_complex *>(planes_[i].data()->data()));
#endif
}

int Patchwork::padx() const
{
	return padx_;
}

int Patchwork::pady() const
{
	return pady_;
}

int Patchwork::interval() const
{
	return interval_;
}

bool Patchwork::empty() const
{
	return planes_.empty();
}

void Patchwork::convolve(const vector<Filter> & filters,
						 vector<vector<HOGPyramid::Matrix> > & convolutions) const
{
	const int nbFilters = filters.size();
	const int nbPlanes = planes_.size();
	const int nbLevels = rectangles_.size();
	
	// Early return if the patchwork or the filters are empty
	if (empty() || !nbFilters) {
		convolutions.clear();
		return;
	}
	
	// Pointwise multiply the transformed filters with the patchwork's planes
	// The performace measurements reported in the paper were done without reallocating the sums
	// each time by making them static
	// Even though it was faster (~10%) I removed it as it was not clean/thread safe
	vector<vector<Patchwork::Matrix> > sums(nbFilters);
	
	for (int i = 0; i < nbFilters; ++i) {
		sums[i].resize(nbPlanes);
		
		for (int j = 0; j < nbPlanes; ++j)
			sums[i][j].resize(MaxRows_, HalfCols_);
	}
	
	// The following assumptions are not dangerous in the sense that the program will only work
	// slower if they do not hold
	const int cacheSize = 32768; // Assume L1 cache of 32K
	const int fragmentsSize = (nbPlanes + 1) * sizeof(Cell); // Assume nbPlanes < nbFilters
	const int step = min(cacheSize / fragmentsSize,
#ifdef _OPENMP
						 MaxRows_ * HalfCols_ / omp_get_max_threads());
#else
						 MaxRows_ * HalfCols_);
#endif
	
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i <= MaxRows_ * HalfCols_ - step; i += step)
		for (int j = 0; j < nbFilters; ++j)
			for (int k = 0; k < nbPlanes; ++k)
				for (int l = 0; l < step; ++l)
					sums[j][k](i + l) =
						filters[j].first(i + l).cwiseProduct(planes_[k](i + l)).sum();
	
	for (i = MaxRows_ * HalfCols_ - ((MaxRows_ * HalfCols_) % step); i < MaxRows_ * HalfCols_; ++i)
		for (int j = 0; j < nbFilters; ++j)
			for (int k = 0; k < nbPlanes; ++k)
				sums[j][k](i) = filters[j].first(i).cwiseProduct(planes_[k](i)).sum();
	
	// Transform back the results and store them in convolutions
	convolutions.resize(nbFilters);
	
	for (int i = 0; i < nbFilters; ++i)
		convolutions[i].resize(nbLevels);
	
#pragma omp parallel for private(i)
	for (i = 0; i < nbFilters * nbPlanes; ++i) {
		const int k = i / nbPlanes; // Filter index
		const int l = i % nbPlanes; // Plane index
		
		Map<HOGPyramid::Matrix, Aligned>
			output(reinterpret_cast<HOGPyramid::Scalar *>(sums[k][l].data()), MaxRows_,
				   HalfCols_ * 2);
		
#ifndef FFLD_HOGPYRAMID_DOUBLE
		fftwf_execute_dft_c2r(Inverse_, reinterpret_cast<fftwf_complex *>(sums[k][l].data()),
							  output.data());
#else
		fftw_execute_dft_c2r(Inverse_, reinterpret_cast<fftw_complex *>(sums[k][l].data()),
							 output.data());
#endif
		
		for (int j = 0; j < nbLevels; ++j) {
			const int rows = rectangles_[j].first.height() + pady_ - filters[k].second.first + 1;
			const int cols = rectangles_[j].first.width() + padx_ - filters[k].second.second + 1;
			
			if ((rows > 0) && (cols > 0) && (rectangles_[j].second == l)) {
				const int x = rectangles_[j].first.x();
				const int y = rectangles_[j].first.y();
				const int width = rectangles_[j].first.width();
				const int height = rectangles_[j].first.height();
				
				if ((rows <= height) && (cols <= width)) {
					convolutions[k][j] = output.block(y, x, rows, cols);
				}
				else {
					convolutions[k][j].resize(rows, cols);
					convolutions[k][j].topLeftCorner(min(rows, height), min(cols, width)) =
						output.block(y, x, min(rows, height), min(cols, width));
					
					if (rows > height)
						convolutions[k][j].bottomRows(rows - height).fill(output(y, x));
					
					if (cols > width)
						convolutions[k][j].rightCols(cols - width).fill(output(y, x));
				}
			}
		}
	}
}

bool Patchwork::Init(int maxRows, int maxCols)
{
	// It is an error if maxRows or maxCols are too small
	if ((maxRows < 2) || (maxCols < 2))
		return false;
	
	// Temporary matrices
	HOGPyramid::Matrix tmp(maxRows * HOGPyramid::NbFeatures, maxCols + 2);
	
	int dims[2] = {maxRows, maxCols};
	
#ifndef FFLD_HOGPYRAMID_DOUBLE
	// Use fftwf_import_wisdom_from_file and not fftwf_import_wisdom_from_filename as old versions
	// of fftw seem to not include it
	FILE * file = fopen("wisdom.fftw", "r");
	
	if (file) {
		fftwf_import_wisdom_from_file(file);
		fclose(file);
	}
	
	const fftwf_plan forwards =
		fftwf_plan_many_dft_r2c(2, dims, HOGPyramid::NbFeatures, tmp.data(), 0,
								HOGPyramid::NbFeatures, 1,
								reinterpret_cast<fftwf_complex *>(tmp.data()), 0,
								HOGPyramid::NbFeatures, 1, FFTW_PATIENT);
	
	const fftwf_plan inverse =
		fftwf_plan_dft_c2r_2d(dims[0], dims[1], reinterpret_cast<fftwf_complex *>(tmp.data()),
							  tmp.data(), FFTW_PATIENT);
	
	file = fopen("wisdom.fftw", "w");
	
	if (file) {
		fftwf_export_wisdom_to_file(file);
		fclose(file);
	}
#else
	FILE * file = fopen("wisdom.fftw", "r");
	
	if (file) {
		fftw_import_wisdom_from_file(file);
		fclose(file);
	}
	
	const fftw_plan forwards =
		fftw_plan_many_dft_r2c(2, dims, HOGPyramid::NbFeatures, tmp.data(), 0,
							   HOGPyramid::NbFeatures, 1,
							   reinterpret_cast<fftw_complex *>(tmp.data()), 0,
							   HOGPyramid::NbFeatures, 1, FFTW_PATIENT);
	
	const fftw_plan inverse =
		fftw_plan_dft_c2r_2d(dims[0], dims[1], reinterpret_cast<fftw_complex *>(tmp.data()),
							 tmp.data(), FFTW_PATIENT);
	
	file = fopen("wisdom.fftw", "w");
	
	if (file) {
		fftw_export_wisdom_to_file(file);
		fclose(file);
	}
#endif
	
	// If successful, set MaxRows_, MaxCols_, HalfCols_, Forwards_ and Inverse_
	if (forwards && inverse) {
		MaxRows_ = maxRows;
		MaxCols_ = maxCols;
		HalfCols_ = maxCols / 2 + 1;
		Forwards_ = forwards;
		Inverse_ = inverse;
		return true;
	}
	
	return false;
}

int Patchwork::MaxRows()
{
	return MaxRows_;
}

int Patchwork::MaxCols()
{
	return MaxCols_;
}

void Patchwork::TransformFilter(const HOGPyramid::Level & filter, Filter & result)
{
	// Early return if no filter given or if Init was not called or if the filter is too large
	if (!filter.size() || !MaxRows_ || (filter.rows() > MaxRows_) || (filter.cols() > MaxCols_)) {
		result = Filter();
		return;
	}
	
	// Recopy the filter into a plane
	result.first = Plane::Constant(MaxRows_, HalfCols_, Cell::Zero());
	result.second = pair<int, int>(filter.rows(), filter.cols());
	
	Map<HOGPyramid::Level, Aligned> plane(reinterpret_cast<HOGPyramid::Cell *>(result.first.data()),
										  MaxRows_, HalfCols_ * 2);
	
	for (int y = 0; y < filter.rows(); ++y)
		for (int x = 0; x < filter.cols(); ++x)
			plane((MaxRows_ - y) % MaxRows_, (MaxCols_ - x) % MaxCols_) = filter(y, x) /
																		  (MaxRows_ * MaxCols_);
	
	// Transform that plane	
#ifndef FFLD_HOGPYRAMID_DOUBLE
	fftwf_execute_dft_r2c(Forwards_, reinterpret_cast<float *>(plane.data()->data()),
						  reinterpret_cast<fftwf_complex *>(result.first.data()->data()));
#else
	fftw_execute_dft_r2c(Forwards_, reinterpret_cast<double *>(plane.data()->data()),
						 reinterpret_cast<fftw_complex *>(result.first.data()->data()));
#endif
}

namespace FFLD
{
namespace detail
{
// Order rectangles by decreasing area.
class AreaComparator
{
public:
	AreaComparator(const vector<pair<Rectangle, int> > & rectangles) :
	rectangles_(rectangles)
	{
	}
	
	/// Returns whether rectangle @p a comes before @p b.
	bool operator()(int a, int b) const
	{
		const int areaA = rectangles_[a].first.area();
		const int areaB = rectangles_[b].first.area();
		
		return (areaA > areaB) || ((areaA == areaB) && (rectangles_[a].first.height() >
														rectangles_[b].first.height()));
	}
	
private:
	const vector<pair<Rectangle, int> > & rectangles_;
};

// Order free gaps (rectangles) by position and then by size
struct PositionComparator
{
	// Returns whether rectangle @p a comes before @p b
	bool operator()(const Rectangle & a, const Rectangle & b) const
	{
		return (a.y() < b.y()) ||
			   ((a.y() == b.y()) &&
				((a.x() < b.x()) ||
				 ((a.x() == b.x()) &&
				  ((a.height() > b.height()) ||
				   ((a.height() == b.height()) && (a.width() > b.width()))))));
	}
};
}
}

int Patchwork::BLF(vector<pair<Rectangle, int> > & rectangles)
{
	// Order the rectangles by decreasing area. If a rectangle is bigger than MaxRows x MaxCols
	// return -1
	vector<int> ordering(rectangles.size());
	
	for (int i = 0; i < rectangles.size(); ++i) {
		if ((rectangles[i].first.width() > MaxCols_) || (rectangles[i].first.height() > MaxRows_))
			return -1;
		
		ordering[i] = i;
	}
	
	sort(ordering.begin(), ordering.end(), detail::AreaComparator(rectangles));
	
	// Index of the plane containing each rectangle
	for (int i = 0; i < rectangles.size(); ++i)
		rectangles[i].second = -1;
	
	vector<set<Rectangle, detail::PositionComparator> > gaps;
	
	// Insert each rectangle in the first gap big enough
	for (int i = 0; i < rectangles.size(); ++i) {
		pair<Rectangle, int> & rect = rectangles[ordering[i]];
		
		// Find the first gap big enough
		set<Rectangle, detail::PositionComparator>::iterator g;
		
		for (int i = 0; (rect.second == -1) && (i < gaps.size()); ++i) {
			for (g = gaps[i].begin(); g != gaps[i].end(); ++g) {
				if ((g->width() >= rect.first.width()) && (g->height() >= rect.first.height())) {
					rect.second = i;
					break;
				}
			}
		}
		
		// If no gap big enough was found, add a new plane
		if (rect.second == -1) {
			set<Rectangle, detail::PositionComparator> plane;
			plane.insert(Rectangle(MaxCols_, MaxRows_)); // The whole plane is free
			gaps.push_back(plane);
			g = gaps.back().begin();
			rect.second = gaps.size() - 1;
		}
		
		// Insert the rectangle in the gap
		rect.first.setX(g->x());
		rect.first.setY(g->y());
		
		// Remove all the intersecting gaps, and add newly created gaps
		for (g = gaps[rect.second].begin(); g != gaps[rect.second].end();) {
			if (!((rect.first.right() < g->left()) || (rect.first.bottom() < g->top()) ||
				  (rect.first.left() > g->right()) || (rect.first.top() > g->bottom()))) {
				// Add a gap to the left of the new rectangle if possible
				if (g->x() < rect.first.x())
					gaps[rect.second].insert(Rectangle(g->x(), g->y(), rect.first.x() - g->x(),
													   g->height()));
				
				// Add a gap on top of the new rectangle if possible
				if (g->y() < rect.first.y())
					gaps[rect.second].insert(Rectangle(g->x(), g->y(), g->width(),
													   rect.first.y() - g->y()));
				
				// Add a gap to the right of the new rectangle if possible
				if (g->right() > rect.first.right())
					gaps[rect.second].insert(Rectangle(rect.first.right() + 1, g->y(),
													   g->right() - rect.first.right(),
													   g->height()));
				
				// Add a gap below the new rectangle if possible
				if (g->bottom() > rect.first.bottom())
					gaps[rect.second].insert(Rectangle(g->x(), rect.first.bottom() + 1, g->width(),
													   g->bottom() - rect.first.bottom()));
				
				// Remove the intersecting gap
				gaps[rect.second].erase(g++);
			}
			else {
				++g;
			}
		}
	}
	
	return gaps.size();
}
