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

#ifndef FFLD_SCENE_H
#define FFLD_SCENE_H

#include "Object.h"

#include <string>
#include <vector>


namespace FFLD
{
/// The Scene class represents a Pascal scene, consisting of a (filename to a) jpeg image and a list
/// of Pascal objects. It stores most of the information present in a Pascal VOC 2007 .xml
/// annotation file. Missing are the <source>, <owner>, and <segmented> fields, as they are
/// irrelevant to the training or testing of object detectors. The <folder> and (image) <filename>
/// fields are also merged together into an absolute filename, derived from the scene filename.
class Scene
{
public:
	/// Constructs an empty scene. An empty scene has an empty image and no object.
	Scene();
	
	/// Constructs a scene from informations about a jpeg image and a list of objects.
	/// @param[in] width Width of the image.
	/// @param[in] height Height of the image.
	/// @param[in] depth Depth of the image.
	/// @param[in] filename Filename of the image.
	/// @param[in] objects List of objects present in the scene.
	Scene(int width, int height, int depth, const std::string & filename,
		  const std::vector<Object> & objects);
	
	/// Constructs a scene and tries to load the scene from the xml file with the given
	/// @p filename.
	Scene(const std::string & filename);
	
	/// Constructs a scene and tries to load the scene from in-memory xml data in
	/// @p buffer of @p size bytes.
	Scene(const char * buffer, int size);
	
	/// Returns the width of the image.
	int width() const;
	
	/// Sets the width of the image.
	void setWidth(int width);
	
	/// Returns the height of the image.
	int height() const;
	
	/// Sets the height of the image.
	void setHeight(int height);
	
	/// Returns the depth of the image. The image depth is the number of color channels.
	int depth() const;
	
	/// Sets the depth of the image.
	void setDepth(int depth);
	
	/// Returns the filename of the image.
	const std::string & filename() const;
	
	/// Sets the filename of the image.
	void setFilename(const std::string & filename);
	
	/// Returns the list of objects present in the scene.
	const std::vector<Object> & objects() const;
	
	/// Sets the list of objects present in the scene.
	void setObjects(const std::vector<Object> & objects);
	
	/// Returns whether the scene is empty. An empty scene has an empty image and no object.
	bool empty() const;
	
private:
	int width_;
	int height_;
	int depth_;
	std::string filename_;
	std::vector<Object> objects_;
	
	void parseXmlDoc(void * xmldoc);
};

/// Serializes a scene to a stream.
std::ostream & operator<<(std::ostream & os, const Scene & scene);

/// Unserializes a scene from a stream.
std::istream & operator>>(std::istream & is, Scene & scene);
}

#endif
