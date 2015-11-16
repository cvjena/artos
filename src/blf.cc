#include "blf.h"
#include <algorithm>
#include <set>
using namespace std;

namespace ARTOS
{

namespace detail
{
// Order rectangles by decreasing area.
class AreaComparator
{
public:
    AreaComparator(const vector<PatchworkRectangle> & rectangles) :
    rectangles_(rectangles)
    {
    }
    
    /// Returns whether rectangle @p a comes before @p b.
    bool operator()(int a, int b) const
    {
        const int areaA = rectangles_[a].area();
        const int areaB = rectangles_[b].area();
        
        return (areaA > areaB) || ((areaA == areaB) && (rectangles_[a].height() >
                                                        rectangles_[b].height()));
    }
    
private:
    const vector<PatchworkRectangle> & rectangles_;
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

int BLF(vector<PatchworkRectangle> & rectangles, unsigned int maxWidth, unsigned int maxHeight)
{
    // Order the rectangles by decreasing area. If a rectangle is bigger than MaxRows x MaxCols
    // return -1
    vector<int> ordering(rectangles.size());
    
    for (int i = 0; i < rectangles.size(); ++i) {
        if ((rectangles[i].width() > maxWidth) || (rectangles[i].height() > maxHeight))
            return -1;
        
        ordering[i] = i;
    }
    
    sort(ordering.begin(), ordering.end(), detail::AreaComparator(rectangles));
    
    // Index of the plane containing each rectangle
    for (int i = 0; i < rectangles.size(); ++i)
        rectangles[i].setPlane(-1);
    
    vector< set<Rectangle, detail::PositionComparator> > gaps;
    
    // Insert each rectangle in the first gap big enough
    for (int i = 0; i < rectangles.size(); ++i)
    {
        PatchworkRectangle & rect = rectangles[ordering[i]];
        
        // Find the first gap big enough
        set<Rectangle, detail::PositionComparator>::iterator g;
        
        for (int i = 0; (rect.plane() == -1) && (i < gaps.size()); ++i)
            for (g = gaps[i].begin(); g != gaps[i].end(); ++g)
                if ((g->width() >= rect.width()) && (g->height() >= rect.height()))
                {
                    rect.setPlane(i);
                    break;
                }
        
        // If no gap big enough was found, add a new plane
        if (rect.plane() == -1)
        {
            set<Rectangle, detail::PositionComparator> plane;
            plane.insert(Rectangle(maxWidth, maxHeight)); // The whole plane is free
            gaps.push_back(plane);
            g = gaps.back().begin();
            rect.setPlane(gaps.size() - 1);
        }
        
        // Insert the rectangle in the gap
        rect.setX(g->x());
        rect.setY(g->y());
        
        // Remove all the intersecting gaps, and add newly created gaps
        for (g = gaps[rect.plane()].begin(); g != gaps[rect.plane()].end();)
        {
            if (!((rect.right() < g->left()) || (rect.bottom() < g->top()) ||
                  (rect.left() > g->right()) || (rect.top() > g->bottom())))
            {
                // Add a gap to the left of the new rectangle if possible
                if (g->x() < rect.x())
                    gaps[rect.plane()].insert(Rectangle(g->x(), g->y(), rect.x() - g->x(),
                                                       g->height()));
                
                // Add a gap on top of the new rectangle if possible
                if (g->y() < rect.y())
                    gaps[rect.plane()].insert(Rectangle(g->x(), g->y(), g->width(),
                                                       rect.y() - g->y()));
                
                // Add a gap to the right of the new rectangle if possible
                if (g->right() > rect.right())
                    gaps[rect.plane()].insert(Rectangle(rect.right() + 1, g->y(),
                                                       g->right() - rect.right(),
                                                       g->height()));
                
                // Add a gap below the new rectangle if possible
                if (g->bottom() > rect.bottom())
                    gaps[rect.plane()].insert(Rectangle(g->x(), rect.bottom() + 1, g->width(),
                                                       g->bottom() - rect.bottom()));
                
                // Remove the intersecting gap
                gaps[rect.plane()].erase(g++);
            }
            else
                ++g;
        }
    }
    
    return gaps.size();
}

}