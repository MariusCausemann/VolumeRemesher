#include "parallel.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#ifdef __linux__
#include <sched.h>
#endif

namespace vol_rem {

namespace {
// 0: no explicit setting (see num_threads).
std::atomic<unsigned> requested_num_threads{0};

#ifdef __linux__
// The number of CPUs in this process's affinity mask, 0 if it cannot be read. cpu_set_t holds
// 1024 CPUs; on a bigger machine sched_getaffinity fails with EINVAL, so the set is grown until
// the call succeeds.
unsigned affinity_cpu_count()
{
    for (int ncpus = CPU_SETSIZE; ncpus <= (1 << 20); ncpus *= 2) {
        cpu_set_t* set = CPU_ALLOC(ncpus);
        if (!set) return 0;
        const size_t size = CPU_ALLOC_SIZE(ncpus);
        CPU_ZERO_S(size, set);
        const int rc = sched_getaffinity(0, size, set);
        const int n = rc == 0 ? CPU_COUNT_S(size, set) : 0;
        CPU_FREE(set);
        if (rc == 0) return n > 0 ? (unsigned)n : 0;
        if (errno != EINVAL) return 0;
    }
    return 0;
}
#endif
} // namespace

unsigned num_threads()
{
    if (unsigned n = requested_num_threads.load(std::memory_order_relaxed)) return n;
    if (const char* env = std::getenv("VOLUMEREMESHER_NUM_THREADS")) {
        char* end = nullptr;
        const long n = std::strtol(env, &end, 10);
        if (n > 0 && end != env && *end == '\0') return (unsigned)n;
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "VOLUMEREMESHER_NUM_THREADS=\"%s\" is not a positive integer: ignored\n", env);
        }
    }
#ifdef __linux__
    if (const unsigned n = affinity_cpu_count()) return n;
#endif
    const unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1;
}

void set_num_threads(unsigned n)
{
    requested_num_threads.store(n, std::memory_order_relaxed);
}

} // namespace vol_rem
