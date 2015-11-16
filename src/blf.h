#ifndef ARTOS_BLF_H
#define ARTOS_BLF_H

#include <vector>
#include "Rectangle.h"

namespace ARTOS
{

/**
* A rectangle on a plane of a patchwork.
*/
class PatchworkRectangle : public Rectangle
{

public:

    /**
    * Constructs an empty rectangle. An empty rectangle has no area.
    */
    PatchworkRectangle() : Rectangle(), plane_(-1) {};
    
    /**
    * Constructs a rectangle with the given @p width and @p height.
    */
    PatchworkRectangle(int width, int height) : Rectangle(width, height), plane_(-1) {};
    
    /**
    * Constructs a rectangle with coordinates (@p x, @p y) and the given @p width and @p height.
    */
    PatchworkRectangle(int x, int y, int width, int height) : Rectangle(x, y, width, height), plane_(-1) {};
    
    /**
    * Constructs a rectangle with coordinates (@p x, @p y) and the given @p width and @p height on a given @p plane.
    */
    PatchworkRectangle(int x, int y, int width, int height, int plane) : Rectangle(x, y, width, height), plane_(plane) {};
    
    /**
    * @return Returns the index of the plane this rectangle is placed on. -1 will be returned if no plane is assigned.
    */
    int plane() const { return plane_; };
    
    /**
    * @param[in] plane Index of the plane this rectangle is to be assigned to.
    */
    void setPlane(int plane) { plane_ = plane; };

private:

    int plane_;

};


/**
* @brief Bottom-Left fill algorithm.
*
* Fills planes of a fixed size with a number of rectangles of given size.
*
* @param[in,out] rectangles A vector of rectangles with plane indices.
* The width and height of the rectangles must be set and this function will change
* their x and y coordinate as well as the plane index.
*
* @param[in] maxWidth Width of the planes.
*
* @param[in] maxHeight Height of the planes.
*
* @return Returns the number of planes needed for the rectangles. A negative number
* will be returned on error.
*/
int BLF(std::vector<PatchworkRectangle> & rectangles, unsigned int maxWidth, unsigned int maxHeight);

}

#endif