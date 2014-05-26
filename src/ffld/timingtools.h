#ifndef TIME_INCLUDE
#define TIME_INCLUDE

#ifndef _WIN32
#include <sys/time.h>

extern timeval Start, Stop;

inline void start()
{
	gettimeofday(&Start, 0);
}

inline int stop()
{
	gettimeofday(&Stop, 0);
	
	timeval duration;
	timersub(&Stop, &Start, &duration);
	
	return duration.tv_sec * 1000 + (duration.tv_usec + 500) / 1000;
}
#else
#include <time.h>
#include <windows.h>

extern ULARGE_INTEGER Start, Stop;

inline void start()
{
	GetSystemTimeAsFileTime((FILETIME *)&Start);
}

inline int stop()
{
	GetSystemTimeAsFileTime((FILETIME *)&Stop);
	Stop.QuadPart -= Start.QuadPart;
	return (Stop.QuadPart + 5000) / 10000;
}

#endif
#endif

