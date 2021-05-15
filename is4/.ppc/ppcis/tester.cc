#include "is.h"
#include "ppc.h"

#include <random>
#include <vector>

#ifdef __clang__
typedef long double pfloat;
#else
typedef __float128 pfloat;
#endif

static constexpr float THRESHOLD = 0.0001;
static constexpr float MINDIFF = 0.001f;
std::unique_ptr<ppc::fdostream> stream;

#define CHECK_READ(x)     \
    do {                  \
        if (!(x)) {       \
            std::exit(1); \
        }                 \
    } while (false)

#define CHECK_END(x)      \
    do {                  \
        std::string _tmp; \
        if (x >> _tmp) {  \
            std::exit(1); \
        }                 \
    } while (false)

static void colours(ppc::random &rng, float a[3], float b[3]) {
    float maxdiff = 0;
    do {
        bool done = false;
        while (!done) {
            for (int k = 0; k < 3; ++k) {
                a[k] = rng.get_double();
                b[k] = rng.get_uint64(0, 1) ? rng.get_double() : a[k];
                if (a[k] != b[k]) {
                    done = true;
                }
            }
        }
        maxdiff = std::max({std::abs(a[0] - b[0]),
                            std::abs(a[1] - b[1]),
                            std::abs(a[2] - b[2])});
    } while (maxdiff < MINDIFF);
}

static void dump(const float (&a)[3]) {
    *stream << std::scientific << a[0] << "," << std::scientific << a[1] << "," << std::scientific << a[2];
}

static void dump(const Result &r) {
    *stream
        << "y0\t" << r.y0 << "\n"
        << "x0\t" << r.x0 << "\n"
        << "y1\t" << r.y1 << "\n"
        << "x1\t" << r.x1 << "\n"
        << "outer\t";
    dump(r.outer);
    *stream << "\n";
    *stream << "inner\t";
    dump(r.inner);
    *stream << "\n";
}

static bool close(float a, float b) {
    return std::abs(a - b) < THRESHOLD;
}

static bool equal(const float (&a)[3], const float (&b)[3]) {
    return close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2]);
}

static void compare(bool is_test, int ny, int nx, const Result &e, const Result &r, const float *data) {
    if (is_test) {
        if (e.y0 == r.y0 && e.x0 == r.x0 && e.y1 == r.y1 && e.x1 == r.x1 && equal(e.outer, r.outer) && equal(e.inner, r.inner)) {
            *stream << "result\tpass\n";
        } else {
            bool small = ny * nx <= 200;
            stream->precision(std::numeric_limits<float>::max_digits10 - 1);
            *stream
                << "result\tfail\n"
                << "threshold\t" << std::scientific << THRESHOLD << '\n'
                << "ny\t" << ny << "\n"
                << "nx\t" << nx << "\n"
                << "what\texpected\n";
            dump(e);
            *stream << "what\tgot\n";
            dump(r);
            *stream << "size\t" << (small ? "small" : "large") << '\n';
            if (small) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        const float *p = &data[3 * x + 3 * nx * y];
                        const float v[3] = {p[0], p[1], p[2]};
                        *stream << "triple\t";
                        dump(v);
                        *stream << "\n";
                    }
                }
            }
        }
    } else {
        *stream << "result\tdone\n";
    }
    *stream << std::flush;
}

static void test(bool is_test, ppc::random &rng, int ny, int nx, int y0, int x0,
                 int y1, int x1, bool binary, bool worstcase) {
    Result e;
    e.y0 = y0;
    e.x0 = x0;
    e.y1 = y1;
    e.x1 = x1;
    if (binary) {
        bool flip = rng.get_uint64(0, 1);
        for (int c = 0; c < 3; ++c) {
            e.inner[c] = flip ? 0.0f : 1.0f;
            e.outer[c] = flip ? 1.0f : 0.0f;
        }
    } else {
        if (worstcase) {
            // Test worst-case scenario
            for (int c = 0; c < 3; ++c) {
                e.inner[c] = 1.0f;
                e.outer[c] = 1.0f;
            }
            e.outer[0] -= MINDIFF;
        } else {
            // Random but distinct colours
            colours(rng, e.inner, e.outer);
        }
    }

    std::vector<float> data(3 * ny * nx);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            for (int c = 0; c < 3; ++c) {
                bool inside = y0 <= y && y < y1 && x0 <= x && x < x1;
                data[c + 3 * x + 3 * nx * y] = inside ? e.inner[c] : e.outer[c];
            }
        }
    }

    Result r;
    {
        ppc::setup_cuda_device();
        ppc::perf timer;
        timer.start();
        r = segment(ny, nx, data.data());
        timer.stop();
        timer.print_to(*stream);
        ppc::reset_cuda_device();
    }
    compare(is_test, ny, nx, e, r, data.data());
}

static void test(bool is_test, ppc::random &rng, int ny, int nx, bool binary, bool worstcase) {
    if (ny * nx <= 2) {
        std::cerr << "Invalid dimensions" << std::endl;
        std::exit(1);
    }

    bool ok = false;
    int y0, x0, y1, x1;
    do {
        // Random box location
        y0 = rng.get_int32(0, ny - 1);
        x0 = rng.get_int32(0, nx - 1);

        y1 = rng.get_int32(y0 + 1, ny);
        x1 = rng.get_int32(x0 + 1, nx);
        // Avoid ambiguous cases
        if (y0 == 0 && y1 == ny && x0 == 0) {
            ok = false;
        } else if (y0 == 0 && y1 == ny && x1 == nx) {
            ok = false;
        } else if (x0 == 0 && x1 == nx && y0 == 0) {
            ok = false;
        } else if (x0 == 0 && x1 == nx && y1 == ny) {
            ok = false;
        } else {
            ok = true;
        }
    } while (!ok);

    test(is_test, rng, ny, nx, y0, x0, y1, x1, binary, worstcase);
}

static std::vector<float> generate_gradient(
    int ny, int nx, int x0, int x1, int y0, int y1, int y2,
    float color_outer, float color_inner) {
    std::vector<float> bitmap(nx * ny * 3);
    const float fact = 1.0f / float(y2 - y1);

    for (int y = 0; y < ny; ++y) {
        const bool yinside = y >= y0 && y < y1;
        const bool yinside_gradient = y >= y1 && y < y2;
        for (int x = 0; x < nx; ++x) {
            const auto pixel_base = (nx * y + x) * 3;
            const bool xinside = x >= x0 && x < x1;
            const bool inside = yinside && xinside;
            const bool inside_gradient = yinside_gradient && xinside;

            if (inside) {
                for (int c = 0; c < 3; ++c) {
                    bitmap[pixel_base + c] = color_inner;
                }
            } else if (inside_gradient) {
                const float val = float(y2 - y) * fact * (color_inner - color_outer) + color_outer;
                for (int c = 0; c < 3; ++c) {
                    bitmap[pixel_base + c] = val;
                }
            } else {
                for (int c = 0; c < 3; ++c) {
                    bitmap[pixel_base + c] = color_outer;
                }
            }
        }
    }
    return bitmap;
}

static Result segment_gradient(
    int ny, int nx, int x0, int x1,
    int y0, int y1, int y2, const float *data) {
    // We know all the boundaries, except inside the gradient
    pfloat color_outer;
    pfloat color_inner = data[3 * (nx * y0 + x0)];

    if (x0 > 0 || y0 > 0) {
        const pfloat gr_color = data[0];
        color_outer = gr_color;
    } else if (x1 < nx || y2 < ny) {
        const pfloat gr_color = data[3 * (nx * ny) - 1];
        color_outer = gr_color;
    } else {
        throw;
    } // situation should not exist

    const pfloat sumcolor_top = (x1 - x0) * (y1 - y0) * color_inner;
    pfloat min_sqerror = std::numeric_limits<double>::max();
    Result e;

    // calculate all end positions (naively)
    for (int yend = y1; yend <= y2; ++yend) {
        pfloat sumcolor_inside = sumcolor_top;
        for (int y = y1; y < yend; ++y) {
            const int pixel_base = 3 * (nx * y + x0);
            const pfloat gr_color = data[pixel_base];
            sumcolor_inside += (x1 - x0) * gr_color;
        }

        pfloat sumcolor_outside = (ny * nx - (x1 - x0) * (y2 - y0)) * color_outer;
        for (int y = yend; y < y2; ++y) {
            const int pixel_base = 3 * (nx * y + x0);
            const pfloat gr_color = data[pixel_base];
            sumcolor_outside += (x1 - x0) * gr_color;
        }

        const pfloat pixels_inside = pfloat((yend - y0) * (x1 - x0));
        const pfloat pixels_outside = pfloat(ny * nx) - pixels_inside;

        const pfloat color_inside = sumcolor_inside / pixels_inside;
        const pfloat color_outside = sumcolor_outside / pixels_outside;

        pfloat sqerror_inside = (x1 - x0) * (y1 - y0) * (color_inner - color_inside) * (color_inner - color_inside);
        for (int y = y1; y < yend; ++y) {
            const int pixel_base = 3 * (nx * y + x0);
            const pfloat gr_color = data[pixel_base];
            sqerror_inside += (x1 - x0) * (gr_color - color_inside) * (gr_color - color_inside);
        }

        pfloat sqerror_outside = ((ny * nx) - (x1 - x0) * (y2 - y0)) * (color_outer - color_outside) * (color_outer - color_outside);
        for (int y = yend; y < y2; ++y) {
            const int pixel_base = 3 * (nx * y + x0);
            const pfloat gr_color = data[pixel_base];
            sqerror_outside += (x1 - x0) * (gr_color - color_outside) * (gr_color - color_outside);
        }

        const pfloat sqerror = 3.0 * (sqerror_inside + sqerror_outside);
        if (sqerror < min_sqerror) {
            min_sqerror = sqerror;
            for (int c = 0; c < 3; ++c) {
                e.outer[c] = color_outside;
                e.inner[c] = color_inside;
            }
            e.y0 = y0;
            e.y1 = yend;

            e.x0 = x0;
            e.x1 = x1;
        }
    }
    return e;
}

static void test_gradient(bool is_test, ppc::random &rng,
                          int ny, int nx, int x0, int x1,
                          int y0, int y1, int y2) {
    float color_outer;
    float color_inner;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    do {
        color_outer = rng.get_float(0.0f, 1.0f);
        color_inner = rng.get_float(0.0f, 1.0f);
    } while (std::abs(color_outer - color_inner) < MINDIFF);

    const auto data = generate_gradient(
        ny, nx, x0, x1, y0, y1, y2, color_outer, color_inner);

    Result e = segment_gradient(ny, nx, x0, x1, y0, y1, y2, data.data());
    Result r;
    {
        ppc::setup_cuda_device();
        ppc::perf timer;
        timer.start();
        r = segment(ny, nx, data.data());
        timer.stop();
        timer.print_to(*stream);
        ppc::reset_cuda_device();
    }
    compare(is_test, ny, nx, e, r, data.data());
}

static void test_gradient(bool is_test, ppc::random &rng, int ny, int nx) {
    if (ny <= 2 || nx <= 2) {
        std::cerr << "Invalid dimensions" << std::endl;
        std::exit(1);
    }
    bool ok = false;
    int x0, x1, y0, y1, y2;
    while (!ok) {
        // Random box location
        x0 = rng.get_int32(0, nx - 1);
        x1 = rng.get_int32(x0 + 1, nx);
        y0 = rng.get_int32(0, ny - 1);
        y1 = rng.get_int32(y0 + 1, ny);
        y2 = rng.get_int32(y1, ny);
        // Avoid ambiguous cases
        if ((x0 > 0 && x1 < nx && y0 > 0 && y2 < ny))
            ok = true;
    }
    test_gradient(is_test, rng, ny, nx, x0, x1, y0, y1, y2);
}

int main(int argc, char **argv) {
    const char *ppc_output = std::getenv("PPC_OUTPUT");
    int ppc_output_fd = 0;
    if (ppc_output) {
        ppc_output_fd = std::stoi(ppc_output);
    }
    if (ppc_output_fd <= 0) {
        ppc_output_fd = 1;
    }
    stream = std::unique_ptr<ppc::fdostream>(new ppc::fdostream(ppc_output_fd));

    argc--;
    argv++;
    if (argc < 1 || argc > 3) {
        std::cerr << "Invalid usage" << std::endl;
        std::exit(1);
    }

    bool is_test = false;
    if (argv[0] == std::string("--test")) {
        is_test = true;
        argc--;
        argv++;
    }

    std::ifstream input_file(argv[0]);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        std::exit(1);
    }

    std::string input_type;
    CHECK_READ(input_file >> input_type);
    if (input_type == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        CHECK_READ(input_file >> input_type);
    }

    int ny;
    int nx;
    CHECK_READ(input_file >> ny >> nx);

    ppc::random rng(uint32_t(ny) * 0x1234567 + uint32_t(nx));

    if (input_type == "structured-color") {
        test(is_test, rng, ny, nx, false, false);
    } else if (input_type == "structured-worstcase") {
        test(is_test, rng, ny, nx, false, true);
    } else if (input_type == "structured-binary") {
        test(is_test, rng, ny, nx, true, false);
    } else if (input_type == "gradient") {
        test_gradient(is_test, rng, ny, nx);
    } else {
        std::cerr << "Invalid input type" << std::endl;
        std::exit(1);
    }
}
