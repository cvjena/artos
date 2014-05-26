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

#include "Scene.h"

#include <algorithm>
#include <sstream>

#include <libxml/parser.h>

using namespace FFLD;
using namespace std;

Scene::Scene() : width_(0), height_(0), depth_(0)
{
}

Scene::Scene(int width, int height, int depth, const string & filename,
			 const vector<Object> & objects) : width_(width), height_(height), depth_(depth),
filename_(filename), objects_(objects)
{
}

namespace FFLD
{
namespace detail
{
template <typename Result>
inline Result content(const xmlNodePtr cur)
{
	if ((cur == NULL) || (cur->xmlChildrenNode == NULL))
		return Result();
	
	istringstream iss(reinterpret_cast<const char *>(cur->xmlChildrenNode->content));
	Result result;
	iss >> result;
	return result;
}
}
}

Scene::Scene(const string & filename)
{
	xmlDoc * doc = xmlParseFile(filename.c_str());
	if (doc != NULL)
	{
		this->parseXmlDoc(reinterpret_cast<void*>(doc));
		xmlFreeDoc(doc);
	}
}

Scene::Scene(const char * buffer, int size)
{
	xmlDoc * doc = xmlParseMemory(buffer, size);
	if (doc != NULL)
	{
		this->parseXmlDoc(reinterpret_cast<void*>(doc));
		xmlFreeDoc(doc);
	}
}

void Scene::parseXmlDoc(void * xmldoc)
{
	if (xmldoc == NULL)
		return;
	xmlDoc * doc = reinterpret_cast<xmlDoc*>(xmldoc);
	
	const string Names[20] =
	{
		"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
		"train", "tvmonitor"
	};
	
	const string Poses[4] =
	{
		"Frontal", "Left", "Rear", "Right"
	};
	
	xmlNodePtr cur = xmlDocGetRootElement(doc);
	
	if (cur == NULL) {
		return;
	}
	
	if (xmlStrcmp(cur->name, reinterpret_cast<const xmlChar *>("annotation"))) {
		return;
	}
	
	cur = cur->xmlChildrenNode;
	
	while (cur != NULL) {
		if (!xmlStrcmp(cur->name, reinterpret_cast<const xmlChar *>("size"))) {
			xmlNodePtr cur2 = cur->xmlChildrenNode;
			
			while (cur2 != NULL) {
				if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("width")))
					width_ = detail::content<int>(cur2);
				else if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("height")))
					height_ = detail::content<int>(cur2);
				else if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("depth")))
					depth_ = detail::content<int>(cur2);
				
				cur2 = cur2->next;
			}
		}
		else if (!xmlStrcmp(cur->name, reinterpret_cast<const xmlChar *>("object"))) {
			objects_.push_back(Object());
			
			xmlNodePtr cur2 = cur->xmlChildrenNode;
			
			while (cur2 != NULL) {
				if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("name"))) {
					const string * iter =
						find(Names, Names + 20, detail::content<string>(cur2));
					
					if (iter != Names + 20)
						objects_.back().setName(static_cast<Object::Name>(iter - Names));
				}
				else if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("pose"))) {
					const string * iter =
						find(Poses, Poses + 4, detail::content<string>(cur2));
					
					if (iter != Poses + 4)
						objects_.back().setPose(static_cast<Object::Pose>(iter - Poses));
				}
				else if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("truncated"))) {
					objects_.back().setTruncated(detail::content<bool>(cur2));
				}
				else if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("difficult"))) {
					objects_.back().setDifficult(detail::content<bool>(cur2));
				}
				else if (!xmlStrcmp(cur2->name, reinterpret_cast<const xmlChar *>("bndbox"))) {
					Rectangle bndbox;
					
					xmlNodePtr cur3 = cur2->xmlChildrenNode;
					
					while (cur3 != NULL) {
						if (!xmlStrcmp(cur3->name, reinterpret_cast<const xmlChar *>("xmin")))
							bndbox.setX(detail::content<int>(cur3));
						else if (!xmlStrcmp(cur3->name, reinterpret_cast<const xmlChar *>("ymin")))
							bndbox.setY(detail::content<int>(cur3));
						else if (!xmlStrcmp(cur3->name, reinterpret_cast<const xmlChar *>("xmax")))
							bndbox.setWidth(detail::content<int>(cur3));
						else if (!xmlStrcmp(cur3->name, reinterpret_cast<const xmlChar *>("ymax")))
							bndbox.setHeight(detail::content<int>(cur3));
						
						cur3 = cur3->next;
					}
					
					// Only set the bounding box if all values have been assigned
					if (bndbox.width() && bndbox.height()) {
						bndbox.setX(bndbox.x());
						bndbox.setY(bndbox.y());
						bndbox.setWidth(bndbox.width() - bndbox.x() + 1);
						bndbox.setHeight(bndbox.height() - bndbox.y() + 1);
						objects_.back().setBndbox(bndbox);
					}
				}
				
				cur2 = cur2->next;
			}
		}
		
		cur = cur->next;
	}
}

int Scene::width() const
{
	return width_;
}

void Scene::setWidth(int width)
{
	width_ = width;
}

int Scene::height() const
{
	return height_;
}

void Scene::setHeight(int height)
{
	height_ = height;
}

int Scene::depth() const
{
	return depth_;
}

void Scene::setDepth(int depth)
{
	depth_ = depth;
}

const string & Scene::filename() const
{
	return filename_;
}

void Scene::setFilename(const string &filename)
{
	filename_ = filename;
}

const vector<Object> & Scene::objects() const
{
	return objects_;
}

void Scene::setObjects(const vector<Object> &objects)
{
	objects_ = objects;
}

bool Scene::empty() const
{
	return ((width() <= 0) || (height() <= 0) || (depth() <= 0) || filename().empty()) &&
		   objects().empty();
}

ostream & FFLD::operator<<(ostream & os, const Scene & scene)
{
	os << scene.width() << ' ' << scene.height() << ' ' << scene.depth() << ' '
	   << scene.objects().size() << ' ' << scene.filename() << endl;
	
	for (unsigned int i = 0; i < scene.objects().size(); ++i)
		os << scene.objects()[i] << endl;
	
	return os;
}

istream & FFLD::operator>>(istream & is, Scene & scene)
{
	int width, height, depth, nbObjects;
    
    is >> width >> height >> depth >> nbObjects;
	is.get(); // Remove the space
	
	string filename;
	getline(is, filename);
	
	vector<Object> objects(nbObjects);
	
	for (int i = 0; i < nbObjects; ++i)
		is >> objects[i];
	
	if (!is) {
		scene = Scene();
		return is;
	}
	
	scene = Scene(width, height, depth, filename, objects);
	
	return is;
}
