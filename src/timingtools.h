#ifndef TIMINGTOOLS_H
#define TIMINGTOOLS_H

#include <chrono>
#include <vector>

extern std::vector<std::chrono::high_resolution_clock::time_point> TimingStarts;

inline void start()
{
    TimingStarts.push_back(std::chrono::high_resolution_clock::now());
}

inline unsigned int stop()
{
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    
    if (TimingStarts.empty())
        return 0;

    unsigned int duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - TimingStarts.back()).count();
    TimingStarts.pop_back();
    return duration;
}

#endif

