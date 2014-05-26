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

#include "Object.h"

#include <iostream>

using namespace FFLD;
using namespace std;

Object::Object() : name_(UNKNOWN), pose_(UNSPECIFIED), truncated_(false), difficult_(false)
{
}

Object::Object(Name name, Pose pose, bool truncated, bool difficult, Rectangle bndbox) :
name_(name), pose_(pose), truncated_(truncated), difficult_(difficult), bndbox_(bndbox)
{
}

Object::Name Object::name() const
{
	return name_;
}

void Object::setName(Name name)
{
	name_ = name;
}

Object::Pose Object::pose() const
{
	return pose_;
}

void Object::setPose(Pose pose)
{
	pose_ = pose;
}

bool Object::truncated() const
{
	return truncated_;
}

void Object::setTruncated(bool truncated)
{
	truncated_ = truncated;
}

bool Object::difficult() const
{
	return difficult_;
}

void Object::setDifficult(bool difficult)
{
	difficult_ = difficult;
}

Rectangle Object::bndbox() const
{
	return bndbox_;
}

void Object::setBndbox(Rectangle bndbox)
{
	bndbox_ = bndbox;
}

bool Object::empty() const
{
	return (name() == UNKNOWN) && (pose() == UNSPECIFIED) && !truncated() && !difficult() &&
		   bndbox().empty();
}

ostream & FFLD::operator<<(ostream & os, const Object & obj)
{
	return os << static_cast<int>(obj.name()) << ' ' << static_cast<int>(obj.pose()) << ' '
			  << obj.truncated() << ' ' << obj.difficult() << ' ' << obj.bndbox();
}

istream & FFLD::operator>>(istream & is, Object & obj)
{
	int name, pose;
	bool truncated, difficult;
	Rectangle bndbox;
	
    is >> name >> pose >> truncated >> difficult >> bndbox;
	
	if (!is) {
		obj = Object();
		return is;
	}
	
	obj.setName(static_cast<Object::Name>(name));
	obj.setPose(static_cast<Object::Pose>(pose));
	obj.setTruncated(truncated);
	obj.setDifficult(difficult);
	obj.setBndbox(bndbox);
	
    return is;
}
