#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <sys/resource.h>
#include <sys/time.h>
#include <type_traits>
#include <unistd.h>
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif

#ifdef __linux__
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#endif

namespace ppc {
class timer {
  public:
    using time_point = decltype(std::chrono::high_resolution_clock::now());

    timer() {
        start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end = std::chrono::high_resolution_clock::now();
    }

    double seconds() const {
        return (end - start).count() / double(1E9);
    }

  private:
    time_point start;
    time_point end;
};

class random {
  public:
    random() {}
    random(int seed) {
        rng = std::mt19937(seed);
    }
    float get_float() {
        static_assert(std::numeric_limits<float>::radix == 2, "Floating point numbers must be in radix 2");
        static_assert(std::numeric_limits<float>::digits == 24, "Floating point numbers exactly 24 bits in mantissa");
        uint32_t bits = rng() >> (32 - std::numeric_limits<float>::digits);
        return float(bits) * (1.f / 16777216.f /* 2^24 */);
    }

    double get_double() {
        static_assert(std::numeric_limits<double>::radix == 2, "Floating point numbers must be in radix 2");
        static_assert(std::numeric_limits<double>::digits == 53, "Double-precision floating point numbers exactly 53 bits in mantissa");
        uint64_t bits = get_uint64(0, std::numeric_limits<uint64_t>::max()) >> (64 - std::numeric_limits<double>::digits);
        return double(bits) * (1. / 9007199254740992. /* 2^53 */);
    }

    float get_float(float a, float b) {
        return a + get_float() * (b - a);
    }

    double get_double(double a, double b) {
        return a + get_double() * (b - a);
    }

    float get_float_normal() {
        if (float_normal_available) {
            float_normal_available = false;
            return next_float_normal;
        }
        float u1 = get_float();
        float u2 = get_float();
        float r = sqrtf(-2. * logf(u1));
        if (!std::isfinite(r))
            r = 6; // ~sqrt(-2*log(1/2^24))
        next_float_normal = r * sinf(2 * M_PI * u2);
        float_normal_available = true;
        return r * cosf(2. * M_PI * u2);
    }

    double get_double_normal() {
        if (double_normal_available) {
            double_normal_available = false;
            return next_double_normal;
        }
        double u1 = get_double();
        double u2 = get_double();
        double r = sqrt(-2. * log(u1));
        if (!std::isfinite(r))
            r = 9; // ~sqrt(-2*log(1/2^53))
        next_double_normal = r * sin(2 * M_PI * u2);
        double_normal_available = true;
        return r * cos(2. * M_PI * u2);
    }

    uint64_t get_uint64(uint64_t a, uint64_t b) {
        assert(a <= b);
        if (a == 0 && b == std::numeric_limits<uint64_t>::max()) {
            return ((uint64_t)rng() << 32) | (uint64_t)rng();
        }

        uint64_t length = b - a + 1;
        while (true) {
            uint64_t bits = ((uint64_t)rng() << 32) | (uint64_t)rng();
            uint64_t n = bits / length;
            uint64_t begin = n * length;

            if (begin <= begin + (length - 1)) {
                return a + (bits % length);
            }
        }
    }

    int64_t get_int64(int64_t a, int64_t b) {
        assert(a <= b);
        // Convert to unsigned values and offset them such that their order is preserved.
        const uint64_t ua = a;
        const uint64_t ub = b;
        constexpr uint64_t offset = uint64_t(std::numeric_limits<int64_t>::max()) + 1;
        // Check that the offset is correct
        static_assert(uint64_t(std::numeric_limits<int64_t>::min()) + offset < uint64_t(std::numeric_limits<int64_t>::max()) + offset);
        const uint64_t c = get_uint64(ua + offset, ub + offset);
        const int64_t res = c - offset;
        assert(a <= res);
        assert(res <= b);
        return res;
    }

    uint32_t get_uint32(uint32_t a, uint32_t b) {
        return get_uint64(a, b);
    }

    int32_t get_int32(int32_t a, int32_t b) {
        return get_int64(a, b);
    }

  private:
    std::mt19937 rng;
    double next_double_normal;
    double next_float_normal;
    bool double_normal_available = false;
    bool float_normal_available = false;
};

class fdostreambuf : public std::basic_streambuf<char> {
  public:
    fdostreambuf(int fd) {
        file = fdopen(fd, "w");
    }

    ~fdostreambuf() {
        fclose(file);
        file = nullptr;
    }

  protected:
    virtual std::streamsize xsputn(const char_type *s, std::streamsize count) {
        return fwrite(s, 1, count, file);
    }
    virtual int_type overflow(int_type ch) {
        if (!std::char_traits<char>::eq_int_type(ch, std::char_traits<char>::eof())) {
            char c = ch;
            assert(fwrite(&c, 1, 1, file) == 1);
        }
        return ch;
    }

  private:
    std::FILE *file;
};

class fdostream : public std::basic_ostream<char> {
  public:
    fdostream(std::unique_ptr<fdostreambuf> buf) : std::basic_ostream<char>(buf.get()), buf(std::move(buf)) {
    }
    fdostream(int fd) : fdostream(std::make_unique<fdostreambuf>(fd)) {
    }

  private:
    std::unique_ptr<fdostreambuf> buf;
};

inline std::ostream &operator<<(std::ostream &out, const ppc::timer &timer) {
    std::ios_base::fmtflags oldf = out.flags(std::ios::right | std::ios::fixed);
    std::streamsize oldp = out.precision(3);
    out << timer.seconds();
    out.flags(oldf);
    out.precision(oldp);
    out.copyfmt(std::ios(NULL));
    return out;
}

#ifdef __linux__
namespace {
int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                    int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct read_format {
    long long value;        /* The value of the event */
    long long time_enabled; /* if PERF_FORMAT_TOTAL_TIME_ENABLED */
    long long time_running; /* if PERF_FORMAT_TOTAL_TIME_RUNNING */
};
} // namespace
#endif

class perf {
  private:
    bool running;
    std::chrono::nanoseconds elapsed;
    std::chrono::nanoseconds elapsed_cputime;
    std::chrono::time_point<std::chrono::steady_clock> begin;
    std::chrono::nanoseconds begin_cputime;

#ifdef __linux__

    // helper class to manage file handles
    class FD {
      public:
        int fd;

        FD() {
            fd = -1;
        }

        ~FD() {
            if (fd != -1) {
                close(fd);
            }
        }

        FD(const FD &other) = delete;
        FD(FD &&other)
        noexcept {
            fd = other.fd;
            other.fd = -1;
        }
        FD &operator=(const FD &other) = delete;
        FD &operator=(FD &&other) noexcept {
            if (fd != -1) {
                close(fd);
            }
            fd = other.fd;
            other.fd = -1;
            return *this;
        }
    };

    FD inst;
    FD cycles;
    FD branches;
    FD branch_misses;

#endif // __linux__

    std::chrono::nanoseconds get_cputime() {
        using namespace std::chrono;
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        auto dur = seconds(ru.ru_utime.tv_sec) + microseconds(ru.ru_utime.tv_usec);
        return duration_cast<nanoseconds>(dur);
    }

  public:
    perf() {
        running = false;
        elapsed = std::chrono::nanoseconds::zero();
        elapsed_cputime = std::chrono::nanoseconds::zero();

#ifdef __linux__

        char *cfg = std::getenv("PPC_PERF");
        if (!cfg || strcmp(cfg, "") == 0)
            return;

        struct perf_event_attr pe;

        memset(&pe, 0, sizeof(pe));
        //attributes for all events
        pe.size = sizeof(pe);
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        pe.inherit = 1;

        // specific to group leader (created first)
        pe.pinned = 1;
        pe.disabled = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        pe.type = PERF_TYPE_HARDWARE;
        pe.config = PERF_COUNT_HW_INSTRUCTIONS;

        inst.fd = perf_event_open(&pe, 0, -1, -1, 0);

        // others will pass the group leader's fd to be part of the same group
        int group_fd = inst.fd;
        pe.disabled = 0;
        pe.pinned = 0;
        pe.read_format = 0;

        pe.type = PERF_TYPE_HARDWARE;
        pe.config = PERF_COUNT_HW_CPU_CYCLES;
        cycles.fd = perf_event_open(&pe, 0, -1, group_fd, 0);

        pe.type = PERF_TYPE_HARDWARE;
        pe.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
        branches.fd = perf_event_open(&pe, 0, -1, group_fd, 0);

        pe.type = PERF_TYPE_HARDWARE;
        pe.config = PERF_COUNT_HW_BRANCH_MISSES;
        branch_misses.fd = perf_event_open(&pe, 0, -1, group_fd, 0);

#endif // __linux__
    }

    void start() {
        if (running)
            return;
        running = true;
        begin_cputime = get_cputime();
        begin = std::chrono::steady_clock::now();

#ifdef __linux__
        if (inst.fd != -1) {
            ioctl(inst.fd, PERF_EVENT_IOC_ENABLE, 0);
        }
#endif
    }

    void stop() {
        if (!running)
            return;

#ifdef __linux__
        if (inst.fd != -1) {
            ioctl(inst.fd, PERF_EVENT_IOC_DISABLE, 0);
        }
#endif

        auto end = std::chrono::steady_clock::now();
        auto end_cputime = get_cputime();
        elapsed += end - begin;
        elapsed_cputime += end_cputime - begin_cputime;
        running = false;
    }

    void print_to(std::ostream &stream) {
        bool resume = running;
        if (running) {
            stop();
        }

        stream << "perf_wall_clock_ns\t" << elapsed.count() << '\n';

        bool got_cpu_time = false;

#ifdef __linux__
        struct read_format buf;
        if (inst.fd != -1 && read(inst.fd, &buf, 24) == 24) {
            got_cpu_time = true;
            stream << "perf_cpu_time_ns\t" << buf.time_enabled << '\n';
            stream << "perf_instructions\t" << buf.value << '\n';
            if (cycles.fd != -1 && read(cycles.fd, &buf.value, 8) == 8) {
                stream << "perf_cycles\t" << buf.value << '\n';
            }
            if (branches.fd != -1 && read(branches.fd, &buf.value, 8) == 8) {
                stream << "perf_branches\t" << buf.value << '\n';
            }
            if (branch_misses.fd != -1 && read(branch_misses.fd, &buf.value, 8) == 8) {
                stream << "perf_branch_misses\t" << buf.value << '\n';
            }
        }
#endif // __linux__

        if (!got_cpu_time) {
            stream << "perf_cpu_time_ns\t" << elapsed_cputime.count() << '\n';
        }

        if (resume) {
            start();
        }
    }
};

template <typename T>
inline void print_matrix(int ny, int nx, T *data, std::unique_ptr<ppc::fdostream> &stream) {
    *stream << "[";
    for (int i = 0; i < ny - 1; ++i) {
        for (int j = 0; j < nx; ++j)
            *stream << " " << std::scientific << data[j + i * nx];
        *stream << ";";
    }

    for (int j = 0; j < nx; ++j)
        *stream << " " << std::scientific << data[j + (ny - 1) * nx];

    *stream << " ]";
}

#ifdef __NVCC__
inline void setup_cuda_device() {
    cudaError_t err = cudaFree(0); // Documentation promises a no-op
    if (err != cudaSuccess) {
        std::cerr << "Failed to setup cuda device:" << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void reset_cuda_device() {
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "Failed to reset cuda device:" << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#else
inline void setup_cuda_device() {
    // Nothing to do without cuda
}

inline void reset_cuda_device() {
    // Nothing to do without cuda
}
#endif
} // namespace ppc
