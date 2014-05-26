#include "timingtools.h"

#ifndef _WIN32
timeval Start, Stop;
#else
ULARGE_INTEGER Start, Stop;
#endif