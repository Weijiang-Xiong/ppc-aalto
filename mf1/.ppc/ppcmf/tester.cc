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
#include <random>
#include <sstream>
#include <type_traits>
#include <unistd.h>

#include "mf.h"
#include "ppc.h"

const uint64_t FULL_TEST_SIZE = 1'000'000'000;

struct input {
    int ny;
    int nx;
    int hy;
    int hx;
    std::vector<float> input;
};

static bool verify_pixel(int y, int x, const input &input, float *output) {
    int smallcnt = 0;
    int bigcnt = 0;
    int equalcnt = 0;
    float small = -std::numeric_limits<float>::infinity();
    float big = std::numeric_limits<float>::infinity();
    float median = output[x + input.nx * y];
    for (int j = std::max(0, y - input.hy); j < std::min(input.ny, y + input.hy + 1); j++) {
        for (int i = std::max(0, x - input.hx); i < std::min(input.nx, x + input.hx + 1); i++) {
            float val = input.input[i + input.nx * j];
            if (val == median) {
                equalcnt++;
            } else if (val < median) {
                smallcnt++;
                small = std::max(small, val);
            } else {
                bigcnt++;
                big = std::min(big, val);
            }
        }
    }
    // move some equals to one side to make them the same size
    if (smallcnt < bigcnt) {
        int transfer = bigcnt - smallcnt;
        equalcnt -= transfer;
        smallcnt += transfer;
        small = median;
    } else if (bigcnt < smallcnt) {
        int transfer = smallcnt - bigcnt;
        equalcnt -= transfer;
        bigcnt += transfer;
        big = median;
    }
    // check if we have enough equals in the middle to stop here
    if (equalcnt > 0) {
        return true;
    }
    // check if we weren't in the middle
    if (equalcnt < 0) {
        return false;
    }
    // even counts: compute and check mean
    return (big + small) * 0.5f == median;
}

static bool verify(const input &input, float *output, int *errors) {
    uint64_t operation_count = uint64_t(input.ny) * input.nx * (2 * input.hy + 1) * (2 * input.hx + 1);
    bool pass = true;

    if (operation_count <= FULL_TEST_SIZE) {
        for (int y = 0; y < input.ny; y++) {
            for (int x = 0; x < input.nx; x++) {
                bool p = verify_pixel(y, x, input, output);
                if (!p) {
                    errors[x + input.nx * y] = 1;
                    pass = false;
                }
            }
        }
    } else {
        ppc::random rng;
        for (uint64_t i = 0; i < FULL_TEST_SIZE; i += (2 * input.hy + 1) * (2 * input.hx + 1)) {
            int y = rng.get_int32(0, input.ny - 1);
            int x = rng.get_int32(0, input.nx - 1);
            bool p = verify_pixel(y, x, input, output);
            if (!p) {
                errors[x + input.nx * y] = 1;
                pass = false;
            }
        }
    }
    return pass;
}

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

input generate_raw_input(std::ifstream &input_file) {
    std::string header;
    CHECK_READ(getline(input_file, header));
    std::stringstream header_reader(header);
    int ny, nx, hy, hx;
    CHECK_READ(header_reader >> ny >> nx >> hy >> hx);
    CHECK_END(header_reader);

    std::vector<float> data(ny * nx);
    for (int y = 0; y < ny; y++) {
        std::string line;
        CHECK_READ(getline(input_file, line));
        std::stringstream line_reader(line);
        for (int x = 0; x < nx; x++) {
            float value;
            CHECK_READ(line_reader >> value);
            data[y * nx + x] = value;
        }
        CHECK_END(line_reader);
    }
    CHECK_END(input_file);

    return {ny, nx, hy, hx, data};
}

input generate_random_input(std::ifstream &input_file) {
    ppc::random rng;
    int ny, nx, hy, hx;
    CHECK_READ(input_file >> ny >> nx >> hy >> hx);
    CHECK_END(input_file);

    std::vector<float> data(ny * nx);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            data[y * nx + x] = rng.get_float();
        }
    }

    return {ny, nx, hy, hx, data};
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
    std::unique_ptr<ppc::fdostream> stream = std::unique_ptr<ppc::fdostream>(new ppc::fdostream(ppc_output_fd));

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
    if (input_type == "raw") {
        input = generate_raw_input(input_file);
    } else if (input_type == "random") {
        input = generate_random_input(input_file);
    } else {
        std::cerr << "Invalid input type" << std::endl;
        return 3;
    }

    std::vector<float> output(input.nx * input.ny);

    ppc::perf timer;
    timer.start();
    mf(input.ny, input.nx, input.hy, input.hx, input.input.data(), output.data());
    timer.stop();
    timer.print_to(*stream);

    if (test) {
        std::vector<int> errors(input.ny * input.nx);
        bool pass = verify(input, output.data(), errors.data());

        if (!pass) {
            bool small = input.ny * input.nx <= 200;
            stream->precision(std::numeric_limits<float>::max_digits10 - 1);
            *stream
                << "result\tfail\n"
                << "ny\t" << input.ny << '\n'
                << "nx\t" << input.nx << '\n'
                << "hy\t" << input.hy << '\n'
                << "hx\t" << input.hx << '\n'
                << "size\t" << (small ? "small" : "large") << '\n';

            if (small) {

                *stream << "input\t";
                ppc::print_matrix(input.ny, input.nx, input.input.data(), stream);
                *stream << '\n';

                *stream << "output\t";
                ppc::print_matrix(input.ny, input.nx, output.data(), stream);
                *stream << '\n';

                *stream << "locations\t";
                ppc::print_matrix(input.ny, input.nx, errors.data(), stream);
                *stream << '\n';
            }
        } else {
            *stream << "result\tpass\n";
        }
    } else {
        *stream << "result\tdone\n";
    }
    if (stream)
        *stream << std::flush;
}
