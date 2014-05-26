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

#ifndef FFLD_OBJECT_H
#define FFLD_OBJECT_H

#include "Rectangle.h"

#include <iosfwd>

namespace FFLD
{
/// The Object class represents an object in a Scene. It stores all the information present
/// inbetween <object> tags in a Pascal VOC 2007 .xml annotation file, although the bounding box is
/// represented slightly differently (top left coordinates of (0, 0) instead of (1, 1)).
class Object
{
public:
	/// The possible object labels.
	enum Name
	{
		AEROPLANE, BICYCLE, BIRD, BOAT, BOTTLE, BUS, CAR, CAT, CHAIR, COW, DININGTABLE, DOG,
		HORSE, MOTORBIKE, PERSON, POTTEDPLANT, SHEEP, SOFA, TRAIN, TVMONITOR, UNKNOWN
	};
	
	/// The possible object views.
	enum Pose
	{
		FRONTAL, LEFT, REAR, RIGHT, UNSPECIFIED
	};
	
	/// Constructs an empty object. An empty object has name 'unknown', pose 'unspecified', and all
	/// other parameters set to their default values.
	Object();
	
	/// Constructs an object from a name, a pose, annotation flags and a bounding box.
	/// @param[in] name Label of the object.
	/// @param[in] pose View of the object.
	/// @param[in] truncated Whether the object is annotated as being truncated.
	/// @param[in] difficult Whether the object is annotated as being difficult.
	/// @param[in] bndbox Bounding box of the object.
	Object(Name name, Pose pose, bool truncated, bool difficult, Rectangle bndbox);
	
	/// Returns the name (label) of the object.
	Name name() const;
	
	/// Sets the name (label) of the object.
	void setName(Name name);
	
	/// Returns the pose (view) of the object.
	Pose pose() const;
	
	/// Sets the pose (view) of the object.
	void setPose(Pose pose);
	
	/// Returns whether the object is annotated as being truncated.
	bool truncated() const;
	
	/// Annotates the object as being truncated.
	void setTruncated(bool truncated);
	
	/// Returns whether the object is annotated as being difficult.
	bool difficult() const;
	
	/// Annotates the object as being difficult.
	void setDifficult(bool difficult);
	
	/// Returns the bounding box of the object.
	Rectangle bndbox() const;
	
	/// Sets the bounding box of the object.
	void setBndbox(Rectangle bndbox);
	
	/// Returns whether the object is empty.
	/// An empty object has name 'unknown', pose 'unspecified', and all other parameters set to
	/// their default values.
	bool empty() const;
	
private:
	Name name_;
	Pose pose_;
	bool truncated_;
	bool difficult_;
	Rectangle bndbox_;
};

/// Serializes an object to a stream.
std::ostream & operator<<(std::ostream & os, const Object & obj);

/// Unserializes an object from a stream.
std::istream & operator>>(std::istream & is, Object & obj);
}

#endif
