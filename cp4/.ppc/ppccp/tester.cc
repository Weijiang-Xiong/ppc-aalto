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

#include "cp.h"
#include "ppc.h"

typedef long double pfloat;

struct input {
    int ny;
    int nx;
    std::vector<float> input;
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

static float verify(const input &input, const float *result, float *errors) {
    bool nans = false;
    double worst = 0.0;

    std::vector<pfloat> normalized(input.ny * input.nx);

    for (int j = 0; j < input.ny; j++) {
        pfloat s = 0.0;
        pfloat ss = 0.0;
        for (int i = 0; i < input.nx; i++) {
            pfloat x = input.input[j * input.nx + i];
            s += x;
            ss += x * x;
        }
        pfloat mean = s / input.nx;
        pfloat mult = 1.0 / (pfloat)std::sqrt((long double)(ss - s * mean));
        for (int i = 0; i < input.nx; i++)
            normalized[j * input.nx + i] = (input.input[j * input.nx + i] - mean) * mult;
    }

    pfloat err_sqsum = 0.0;

    for (int j = 0; j < input.ny; ++j) {
        for (int i = j; i < input.ny; ++i) {
            float q = result[i + input.ny * j];
            if (q != q) // q is NaN
            {
                nans = true;
                errors[i + input.ny * j] = 0.0 / 0.0;
                continue;
            }
            pfloat temp = 0.0;
            for (int x = 0; x < input.nx; ++x) {
                pfloat a = normalized[x + input.nx * i];
                pfloat b = normalized[x + input.nx * j];
                temp += a * b;
            }
            pfloat err = q - temp;
            err_sqsum += err * err;
            double abserr = std::abs((double)err);
            errors[i + input.ny * j] = abserr;

            worst = std::max(abserr, worst);
        }
    }
    if (nans)
        worst = 0.0 / 0.0;

    return worst;
}

// Does 'iter' iterations of Freivald's algorithm and returns the largest
// difference over all vector elements and iterations.
static float verify_gvfa(const input &input, const float *result, int iter) {
    std::vector<double> normalized(input.ny * input.nx);

    for (int j = 0; j < input.ny; j++) {
        double s = 0.0;
        for (int i = 0; i < input.nx; i++) {
            double x = input.input[j * input.nx + i];
            s += x;
        }
        double mean = s / input.nx;
        double ss = 0.0;
        for (int i = 0; i < input.nx; i++) {
            double x = input.input[j * input.nx + i];
            x -= mean;
            normalized[j * input.nx + i] = x;
            ss += x * x;
        }
        double mult = 1.0 / std::sqrt(ss);
        for (int i = 0; i < input.nx; i++)
            normalized[j * input.nx + i] *= mult;
    }

    ppc::random rng;
    std::vector<double> x(input.ny * iter);
    std::vector<double> ATx(input.nx * iter, 0.0);
    std::vector<double> AATx(input.ny * iter, 0.0);
    std::vector<double> Bx(input.ny * iter);
    for (int j = 0; j < input.ny; j++) {
        for (int k = 0; k < iter; k++) {
            x[j * iter + k] = rng.get_double_normal();
        }
    }

    for (int j = 0; j < input.ny; j++) {
        for (int i = 0; i < input.nx; i++) {
            double left = normalized[j * input.nx + i];
            for (int k = 0; k < iter; k++)
                ATx[i * iter + k] += left * x[j * iter + k];
        }
    }

    for (int j = 0; j < input.ny; j++) {
        for (int i = 0; i < input.nx; i++) {
            for (int k = 0; k < iter; k++)
                AATx[j * iter + k] += normalized[j * input.nx + i] * ATx[i * iter + k];
        }
    }

    for (int j = 0; j < input.ny; j++) {
        for (int i = j; i < input.ny; i++) {
            double left = result[j * input.ny + i];
            for (int k = 0; k < iter; k++)
                Bx[j * iter + k] += left * x[i * iter + k];
            if (i != j) {
                for (int k = 0; k < iter; k++)
                    Bx[i * iter + k] += left * x[j * iter + k];
            }
        }
    }

    float worst = 0.0f;
    for (int j = 0; j < input.ny; j++) {
        for (int k = 0; k < iter; k++) {
            float diff = AATx[j * iter + k] - Bx[j * iter + k];
            float err = std::abs(diff);
            if (err != err)
                return err; // NaN
            worst = std::max(err, worst);
        }
    }
    return worst;
}

static void generate(int ny, int nx, float *data) {
    ppc::random rng;
    for (int y = 0; y < ny; ++y) {
        if (y > 0 && rng.get_uint64(0, 1)) {
            // Introduce some correlations
            int row = std::min(static_cast<int>(y * rng.get_double()), y - 1);
            double offset = 2.0 * (rng.get_double() - 0.5f);
            double mult = 2.0 * rng.get_double();
            for (int x = 0; x < nx; ++x) {
                double error = 0.1 * rng.get_double();
                data[x + nx * y] = mult * data[x + nx * row] + offset + error;
            }
        } else {
            // Generate random data
            for (int x = 0; x < nx; ++x) {
                double v = rng.get_double();
                data[x + nx * y] = v;
            }
        }
    }
}

// Generates random rows in a random 3D subspace
static void generate_subspace(int ny, int nx, float *data) {
    ppc::random rng;

    std::vector<float> a(nx);
    std::vector<float> b(nx);
    std::vector<float> c(nx);
    for (int x = 0; x < nx; x++) {
        a[x] = rng.get_float_normal();
        b[x] = rng.get_float_normal();
        c[x] = rng.get_float_normal();
    }

    for (int y = 0; y < ny; y++) {
        float am = rng.get_float_normal();
        float bm = rng.get_float_normal();
        float cm = rng.get_float_normal();
        float len = std::sqrt(am * am + bm * bm + cm * cm);
        am /= len;
        bm /= len;
        cm /= len;
        for (int x = 0; x < nx; x++)
            data[x + nx * y] = am * a[x] + bm * b[x] + cm * c[x];
    }
}

static void generate_normal(int ny, int nx, float *data) {
    ppc::random rng;
    std::generate(data, data + nx * ny, [&] { return rng.get_float_normal(); });
}

static void generate_special(int ny, int nx, float *data) {
    ppc::random rng;
    const float a = std::numeric_limits<float>::max();
    std::generate(data, data + nx * ny, [&] { return rng.get_double(-a, a); });
}

static void generate_measurement(int ny, int nx, float *data) {
    ppc::random rng;
    std::vector<double> target(nx);
    std::generate(target.begin(), target.end(), [&] { return rng.get_double(); });
    for (int j = 0; j < ny; j++) {
        // parameters for y = ax + b
        double a = 4.0 * rng.get_double() + 0.25; // [0.25, 4.25]
        if (rng.get_uint64(0, 1))
            a = -a;
        double b = 4.0 * rng.get_double() - 2.0; // [-2.0, 2.0]
        double error = rng.get_double();
        for (int i = 0; i < nx; i++) // generate related data with error
            data[j * nx + i] = a * target[i] + b + (rng.get_double() - 0.5) * 2.0 * error;
    }
}

static void generate_benchmark(int ny, int nx, float *data) {
    ppc::random rng;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            float v = rng.get_float();
            data[x + nx * y] = v;
        }
    }
}

static input generate_random_input(std::ifstream &input_file) {
    ppc::random rng;
    int ny, nx;
    int mode;
    CHECK_READ(input_file >> ny >> nx >> mode);
    CHECK_END(input_file);

    std::vector<float> data(ny * nx);

    switch (mode) {
    case 0:
        generate(ny, nx, data.data());
        break;
    case 1:
        generate_normal(ny, nx, data.data());
        break;
    case 2:
        generate_subspace(ny, nx, data.data());
        break;
    case 3:
        generate_measurement(ny, nx, data.data());
        break;
    case 4:
        generate_special(ny, nx, data.data());
        break;
    case 5:
        generate_benchmark(ny, nx, data.data());
        break;
    default:
        std::cerr << "unknown MODE\n";
        std::exit(3);
    }

    return {ny, nx, data};
}

static input generate_raw_input(std::ifstream &input_file) {
    int ny, nx;
    std::string header;
    CHECK_READ(getline(input_file, header));
    std::stringstream header_reader(header);
    CHECK_READ(header_reader >> ny >> nx);
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

    return {ny, nx, data};
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
    bool allow_float = false;
    if (argv[0] == std::string("--test-single") || argv[0] == std::string("--test-double")) {
        test = true;
        if (argv[0] == std::string("--test-single"))
            allow_float = true;
        argc--;
        argv++;
    }

    const float allowed_error = allow_float
                                    ? 1e-5
                                    : std::numeric_limits<float>::epsilon() * 0.6;
    constexpr float gvfa_limit = 1e-3;

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

    std::vector<float> output(input.ny * input.ny);

    ppc::setup_cuda_device();
    ppc::perf timer;
    timer.start();
    correlate(input.ny, input.nx, input.input.data(), output.data());
    timer.stop();
    timer.print_to(*stream);
    ppc::reset_cuda_device();

    if (test) {
        std::vector<float> errors(input.ny * input.ny);
        float gvfa_error = verify_gvfa(input, output.data(), 20);
        bool pass = gvfa_error < gvfa_limit;
        float max_error = 0;
        bool has_max_error = false;
        if ((double)input.nx * input.ny * input.ny < 1e8) {
            max_error = verify(input, output.data(), errors.data());
            has_max_error = true;
            bool full_pass = max_error < allowed_error;
            // gvfa should never claim error if the result is actually ok
            assert(pass || !full_pass);
            pass = full_pass;
        }
        if (!pass) {
            bool small = input.ny * input.nx <= 200 && input.ny * input.ny <= 200;
            stream->precision(std::numeric_limits<float>::max_digits10 - 1);
            *stream
                << "result\tfail\n"
                << "gvfa_error\t" << std::scientific << gvfa_error << '\n'
                << "gvfa_error_limit\t" << std::scientific << gvfa_limit << '\n';
            if (has_max_error) {
                *stream
                    << "max_error\t" << std::scientific << max_error << '\n'
                    << "max_error_limit\t" << std::scientific << allowed_error << '\n';
            }
            *stream
                << "ny\t" << input.ny << '\n'
                << "nx\t" << input.nx << '\n'
                << "size\t" << (small ? "small" : "large") << '\n';

            if (small) {
                *stream << "input\t";
                ppc::print_matrix(input.ny, input.nx, input.input.data(), stream);
                *stream << '\n';

                *stream << "output\t";
                ppc::print_matrix(input.ny, input.ny, output.data(), stream);
                *stream << '\n';

                *stream << "locations\t";
                ppc::print_matrix(input.ny, input.ny, errors.data(), stream);
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
