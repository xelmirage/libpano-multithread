#include "filter.h"
#include <float.h>


//---------for time stat------------

#ifdef WIN32
typedef DWORD os_TIME1;
#else
typedef struct timeval os_TIME;
#endif
void os_GetTime1(os_TIME1* time)
{
#ifdef WIN32
    *time = GetTickCount();
#else
    struct timezone tz;
    gettimeofday(time, &tz);
#endif
}

int os_TimeDiff1(os_TIME1* time1, os_TIME1* time2)
{
#ifdef WIN32
    return *time1 - *time2;
#else
    return (int)((double)time1->tv_sec*1000 + ((double)time1->tv_usec)*1e-3 -
                 (double)time2->tv_sec*1000 - ((double)time2->tv_usec)*1e-3);
#endif
}

#define TIMETRACE(TEXT, CODE) { os_TIME1 t1,t2; os_GetTime1(&t1); CODE; \
        os_GetTime1(&t2); \
		printf("%s took %f seconds.\n",TEXT,os_TimeDiff1(&t2,&t1)/1000.0); }

int getCPUCount()
{
#ifdef WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
#elif defined(HW_NCPU) || defined(__APPLE__)
    // BSD and OSX like system
    int mib[2];
    int numCPUs = 1;
    size_t len = sizeof(numCPUs);

    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    sysctl(mib, 2, &numCPUs, &len, 0, 0);
    return numCPUs;

#elif defined(_SC_NPROCESSORS_ONLN)
    // Linux and Solaris
    long nProcessorsOnline = sysconf(_SC_NPROCESSORS_ONLN);
    return nProcessorsOnline;
#else
    return 1;
#endif
}

#ifdef _WINDOWS
unsigned long long getTotalMemory()
{
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
#ifndef _WIN64
    // when compiled as 32 bit version, we can only use about 2 GB
    // even if we have more memory available on a 64 bit system
    return std::min<unsigned long long>(status.ullTotalPhys, 1500*1024*1024);
#else
    return status.ullTotalPhys;
#endif
};
#elif defined __APPLE__
unsigned long long utils::getTotalMemory()
{
    SInt32 ramSize;
    if(Gestalt(gestaltPhysicalRAMSizeInMegabytes, &ramSize)==noErr)
    {
        return ramSize * 1024 * 1024;
    }
    else
    {
        // if query was not successful return 1 GB, 
        // return 0 would result in crash in calling function
        return 1024*1024*1024;
    }
};
#else
unsigned long long utils::getTotalMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
#endif