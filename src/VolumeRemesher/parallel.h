#ifndef VOLUMEREMESHER_PARALLEL_H
#define VOLUMEREMESHER_PARALLEL_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

namespace vol_rem {

// 0: no explicit setting (see num_threads).
inline std::atomic<unsigned> requested_num_threads{0};

// The number of threads parallel_blocks runs on: set_num_threads(n) when n > 0, else the
// VOLUMEREMESHER_NUM_THREADS environment variable, else the CPUs this process may run on. On
// Linux that is its affinity mask (taskset, cgroups, a batch scheduler's binding), which
// std::thread::hardware_concurrency() ignores: it counts every CPU of the machine.
inline unsigned num_threads()
{
    if (unsigned n = requested_num_threads.load(std::memory_order_relaxed)) return n;
    if (const char* env = std::getenv("VOLUMEREMESHER_NUM_THREADS")) {
        const long n = std::strtol(env, nullptr, 10);
        if (n > 0) return (unsigned)n;
    }
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return (unsigned)n;
    }
#endif
    const unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1;
}

// Set the number of threads parallel_blocks runs on; 0 restores the default (num_threads).
inline void set_num_threads(unsigned n)
{
    requested_num_threads.store(n, std::memory_order_relaxed);
}

//
//
// Run fn over blocks of [0,n) on num_threads() threads (at most one per block), dynamically
// scheduled (atomic block-fetch) so uneven per-item cost stays balanced. Falls back to a
// serial call for small n or a single hardware thread. Compiling with VOLUMEREMESHER_SERIAL_TET
// (CMake: -DVOLUMEREMESHER_PARALLEL_TETRAHEDRALIZATION=OFF) forces the serial path everywhere.
//
// CAREFUL with exact arithmetic in here: NFG's bignatural (the storage under bigfloat and
// bigrational) and expansion allocate from thread_local pools, so such a value must be born,
// used and destroyed inside ONE call of fn -- let only indices and other plain data escape.
// Two mistakes compile cleanly and assert nothing:
//   - computing values in parallel and reading them after the join is a use-after-free (the
//     worker's pool dies with the worker);
//   - destroying a value on a thread other than the one that allocated it pushes the block
//     onto the wrong thread's free stack, which is sized for its own pool and can overflow.
// So a worker must not even assign into a caller-owned bigrational: the caller destroys it.
// See exact_coords.h for how exact results are carried out of a worker as plain limbs.
//
// An exception thrown by fn on a worker (ip_error throws when NFG_IP_ERROR_THROWS is defined,
// as it is in the Python module) is caught there, the remaining blocks are abandoned, and the
// first such exception is rethrown on the calling thread after every worker has joined. Left
// uncaught it would escape a std::thread and terminate the process.
template <class F>
inline void parallel_blocks(uint64_t n, F&& fn)
{
#ifdef VOLUMEREMESHER_SERIAL_TET
    fn(uint64_t(0), n);
#else
    const uint64_t block = 64;
    const unsigned nthreads = (unsigned)std::min<uint64_t>(num_threads(), (n + block - 1) / block);
    if (nthreads <= 1 || n < 512) {
        fn(uint64_t(0), n);
        return;
    }
    std::atomic<uint64_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto runner = [&]() {
        try {
            uint64_t i;
            while (!failed.load(std::memory_order_relaxed) && (i = next.fetch_add(block)) < n)
                fn(i, std::min(n, i + block));
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; t++) pool.emplace_back(runner);
    for (std::thread& t : pool) t.join();
    if (error) std::rethrow_exception(error);
#endif
}

//
//
// std::sort(first, last, comp) on up to num_threads() threads: each thread sorts one
// contiguous run, then pairs of runs are merged in parallel until one is left. Only for plain
// data whose comparison cannot throw. Elements that compare equal may end up in any order, as
// with std::sort.
template <class T, class Compare>
inline void parallel_sort(T* first, T* last, Compare comp)
{
    const uint64_t n = (uint64_t)(last - first);
#ifdef VOLUMEREMESHER_SERIAL_TET
    const uint64_t nruns = 1;
#else
    // Below 2^16 elements per thread the threads cost more than they save.
    const uint64_t nruns = std::max<uint64_t>(1, std::min<uint64_t>(num_threads(), n >> 16));
#endif
    if (nruns == 1) {
        std::sort(first, last, comp);
        return;
    }
    auto run_tasks = [](uint64_t ntasks, auto&& task) {
        std::vector<std::thread> pool;
        pool.reserve(ntasks);
        for (uint64_t t = 0; t < ntasks; t++) pool.emplace_back(task, t);
        for (std::thread& t : pool) t.join();
    };
    std::vector<uint64_t> runs(nruns + 1);
    for (uint64_t r = 0; r <= nruns; r++) runs[r] = n * r / nruns;
    run_tasks(nruns, [&](uint64_t r) { std::sort(first + runs[r], first + runs[r + 1], comp); });

    std::vector<T> buffer(n);
    T* src = first;
    T* dst = buffer.data();
    while (runs.size() > 2) {
        const uint64_t m = runs.size() - 1;
        run_tasks((m + 1) / 2, [&](uint64_t p) {
            const uint64_t b = runs[2 * p], mid = runs[std::min(2 * p + 1, m)],
                           e = runs[std::min(2 * p + 2, m)];
            std::merge(src + b, src + mid, src + mid, src + e, dst + b, comp);
        });
        std::vector<uint64_t> merged;
        for (uint64_t r = 0; r < m; r += 2) merged.push_back(runs[r]);
        merged.push_back(runs[m]);
        runs.swap(merged);
        std::swap(src, dst);
    }
    if (src != first) std::copy(src, src + n, first);
}

} // namespace vol_rem

#endif // VOLUMEREMESHER_PARALLEL_H
