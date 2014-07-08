#include "Random.h"
#include <ctime>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace ARTOS;

bool Random::seeded = false;

void Random::seedOnce()
{
    if (!Random::seeded)
    {
#ifndef _OPENMP
        std::srand(std::time(0));
#else
        #pragma omp parallel
        std::srand(std::time(0) + omp_get_thread_num());
#endif
        Random::seeded = true;
    }
}