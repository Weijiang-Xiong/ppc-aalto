#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <random>
#include <sstream>
#include <type_traits>
#include <unistd.h>

#include "ppc.h"
#include "so.h"

typedef unsigned long long data_t;

struct input {
    int n;
    std::vector<data_t> data;
};

#define CHECK_READ(x) \
    do {              \
        if (!(x)) {   \
            exit(4);  \
        }             \
    } while (false)

#define CHECK_END(x)      \
    do {                  \
        std::string _tmp; \
        if (x >> _tmp) {  \
            exit(4);      \
        }                 \
    } while (false)

enum class gen_type : int {
    rand = 0,
    rand_small,
    constant,
    incr,
    decr,
    benchmark,
};

static constexpr data_t MAGIC = 12345;

template <typename iter>
static void generate(iter begin, iter end, ppc::random rng, gen_type type) {
    int64_t i = 0;
    const auto n = std::distance(begin, end);

    switch (type) {
    case gen_type::rand:
        for (auto it = begin; it != end; ++it) {
            *it = rng.get_uint64(0, std::numeric_limits<data_t>::max());
        }
        break;
    case gen_type::rand_small:
        for (auto it = begin; it != end; ++it) {
            *it = rng.get_uint64(0, 3);
        }
        break;
    case gen_type::constant:
        std::fill(begin, end, MAGIC);
        break;
    case gen_type::incr:
        for (auto it = begin; it != end; ++it) {
            *it = MAGIC + i++;
        }
        break;
    case gen_type::decr:
        for (auto it = begin; it != end; ++it) {
            *it = MAGIC + n - i++;
        }
        break;
    case gen_type::benchmark: {
        const auto sizes = (n + 4) / 5;
        generate(begin + std::min(0 * sizes, n), begin + std::min(1 * sizes, n), rng, gen_type::rand);
        generate(begin + std::min(1 * sizes, n), begin + std::min(2 * sizes, n), rng, gen_type::rand_small);
        generate(begin + std::min(2 * sizes, n), begin + std::min(3 * sizes, n), rng, gen_type::constant);
        generate(begin + std::min(3 * sizes, n), begin + std::min(4 * sizes, n), rng, gen_type::incr);
        generate(begin + std::min(4 * sizes, n), begin + std::min(5 * sizes, n), rng, gen_type::decr);
        break;
    }
    default:
        std::cerr << "unknown MODE\n";
        std::exit(3);
    }
}

data_t calculate_checksum(const input &in) {
    constexpr int MOD_BUCKETS = 13;
    constexpr int MAG_BUCKETS = 11;
    data_t mod_buckets[MOD_BUCKETS] = {0};
    data_t mag_buckets[MAG_BUCKETS] = {0};

    for (int i = 0; i < in.n; i++) {
        data_t x = in.data[i];
        // Scatter values by large primes
        data_t h1 = ((x ^ (x << 31) ^ 6205131147806566763ULL) + (x >> 37) * 12397291431827980453ULL) * 15735802595588038961ULL;
        data_t h2 = ((x ^ (x << 35) ^ 15735802595588038961ULL) + (x >> 18) * 1675862878778982461ULL) * 1847422058446529393ULL;
        // Add to correct buckets
        mod_buckets[h1 % MOD_BUCKETS]++;
        mag_buckets[h2 / (std::numeric_limits<data_t>::max() / MAG_BUCKETS + 1)]++;
    }

    data_t h1 = 0;
    data_t m1 = 1;
    for (int i = 0; i < MOD_BUCKETS; i++) {
        h1 += m1 * mod_buckets[i];
        m1 *= 11958607313276352923ULL;
    }

    data_t h2 = 0;
    data_t m2 = 1;
    for (int i = 0; i < MAG_BUCKETS; i++) {
        h2 += m2 * mag_buckets[i];
        m2 *= 5785869592929004343ULL;
    }

    return h1 ^ h2;
}

input generate_random_input(std::ifstream &input_file) {
    int len;
    std::string mode_s;
    CHECK_READ(input_file >> len >> mode_s);
    gen_type mode;
    if (mode_s == "rand")
        mode = gen_type::rand;
    else if (mode_s == "rand_small")
        mode = gen_type::rand_small;
    else if (mode_s == "constant")
        mode = gen_type::constant;
    else if (mode_s == "incr")
        mode = gen_type::incr;
    else if (mode_s == "decr")
        mode = gen_type::decr;
    else if (mode_s == "benchmark")
        mode = gen_type::benchmark;
    else {
        std::cerr << "Unknown input mode " << mode_s << std::endl;
        std::exit(3);
    }

    ppc::random rng(len * 6 + (int)mode);
    std::vector<data_t> input(len);

    generate(input.begin(), input.end(), rng, (gen_type)mode);

    return {len, input};
}

input generate_raw_input(std::ifstream &input_file) {
    input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string raw_line;
    std::getline(input_file, raw_line);
    CHECK_READ(input_file);
    std::stringstream raw_reader(raw_line);

    std::vector<data_t> input;
    data_t element;
    while (raw_reader >> element) {
        input.push_back(element);
    }
    CHECK_END(raw_reader);

    return {(int)input.size(), input};
}

int main(int argc, const char **argv) {
    const char *ppc_output = std::getenv("PPC_OUTPUT");
    int ppc_output_fd = 0;
    if (ppc_output) {
        ppc_output_fd = std::stoi(ppc_output);
    }
    if (ppc_output_fd <= 0) {
        ppc_output_fd = 1;
    }
    std::unique_ptr<ppc::fdostream> stream =
        std::unique_ptr<ppc::fdostream>(new ppc::fdostream(ppc_output_fd));

    argc--;
    argv++;
    if (argc < 1 || argc > 2) {
        std::cerr << "Invalid usage" << std::endl;
        return 1;
    }

    bool test = false;
    if (argv[0] == std::string("--test")) {
        test = true;
        argc--;
        argv++;
    }

    std::ifstream input_file(argv[0]);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        return 2;
    }

    std::string input_type;
    CHECK_READ(input_file >> input_type);
    if (input_type == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        CHECK_READ(input_file >> input_type);
    }

    input input;
    if (input_type == "random") {
        input = generate_random_input(input_file);
    } else if (input_type == "raw") {
        input = generate_raw_input(input_file);
    } else {
        std::cerr << "Invalid input type: " << input_type << std::endl;
        return 3;
    }

    if (input_file >> input_type) {
        if (input_type == "threads") {
            size_t thread_count;
            CHECK_READ(input_file >> thread_count);
            assert(thread_count > 0);
            omp_set_num_threads(thread_count);
        } else {
            std::cerr << "Unknown input: " << input_type << std::endl;
            return 3;
        }
    }

    data_t checksum = 0;
    if (test) {
        checksum = calculate_checksum(input);
    }

    bool small = input.n <= 200;
    std::vector<data_t> original;
    if (small)
        original = input.data;

    ppc::setup_cuda_device();
    ppc::perf timer;
    timer.start();
    psort(input.n, input.data.data());
    timer.stop();
    timer.print_to(*stream);
    ppc::reset_cuda_device();

    if (test) {
        data_t new_checksum = calculate_checksum(input);

        int error_type = 0;
        if (new_checksum != checksum) {
            error_type = 2;
        } else {
            for (int i = 1; i < input.n; i++) {
                if (input.data[i] < input.data[i - 1]) {
                    error_type = 1;
                }
            }
        }

        if (error_type) {
            *stream << "error_type\t" << error_type << "\n";
            *stream << "result\tfail\n";

            *stream << "n\t" << input.n << '\n'
                    << "size\t" << (small ? "small" : "large") << '\n';

            if (small) {
                *stream << "input\t";
                ppc::print_matrix(1, input.n, original.data(), stream);
                *stream << '\n';

                *stream << "output\t";
                ppc::print_matrix(1, input.n, input.data.data(), stream);
                *stream << '\n';

                std::sort(original.begin(), original.begin() + input.n);
                *stream << "correct\t";
                ppc::print_matrix(1, input.n, original.data(), stream);
                *stream << '\n';
            }
        } else {
            *stream << "result\tpass\n";
        }
    } else {
        *stream << "result\tdone\n";
    }
    *stream << std::endl;
}
