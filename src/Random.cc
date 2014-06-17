#include "Random.h"
#include <ctime>
using namespace ARTOS;

bool Random::seeded = false;

void Random::seedOnce()
{
    if (!Random::seeded)
    {
        std::srand(std::time(0));
        Random::seeded = true;
    }
}