#include "is.h"
#include "ppc.h"

#include <random>
#include <vector>

#ifdef __clang__
typedef long double pfloat;
#else
typedef __float128 pfloat;
#endif

static constexpr float RELATIVE_THRESHOLD = 0.000001;
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

static double total_cost(int ny, int nx, const float *data, const Result &res) {
    double error[3] = {};
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            const float *color;
            if (res.x0 <= x && x < res.x1 && res.y0 <= y && y < res.y1) {
                color = res.inner;
            } else {
                color = res.outer;
            }
            for (int c = 0; c < 3; c++) {
                double diff = (double)color[c] - (double)data[c + 3 * x + 3 * nx * y];
                error[c] += diff * diff;
            }
        }
    }
    return error[0] + error[1] + error[2];
}

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
            double expected_cost = total_cost(ny, nx, data, e);
            double returned_cost = total_cost(ny, nx, data, r);
            double ub = expected_cost * (1.0 + RELATIVE_THRESHOLD);
            double lb = expected_cost * (1.0 - RELATIVE_THRESHOLD);
            if (lb < returned_cost && returned_cost < ub) {
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

// Takes a pointer to three floats and normalizes them so that
// -1 <= a[c] <= 1 with at least one equality (approx.)
static void normalize_uniform(float *a) {
    float m = 1.0 / std::max({std::abs(a[0]), std::abs(a[1]), std::abs(a[2])});
    a[0] *= m;
    a[1] *= m;
    a[2] *= m;
}

static void offset_rect(ppc::random &rng, int ny, int nx, float *image,
                        bool binary, const float offset_max,
                        const float directional_max) {
    float a[3];
    float b[3];
    for (int c = 0; c < 3; ++c) {
        a[c] = rng.get_float(-1.0, 1.0);
        b[c] = rng.get_float(-1.0, 1.0);
    }
    if (binary) {
        // for binary inputs we only care about the first channel, so the
        // directional effect needs to be in that direction
        b[0] = b[0] > 0 ? 1 : -1;
    }
    normalize_uniform(b);

    for (int c = 0; c < 3; ++c) {
        float bt = b[c];
        b[c] = offset_max * a[c] + directional_max * bt;
        a[c] = offset_max * a[c] - directional_max * bt;
    }

    int h = rng.get_int32(ny / 3, ny);
    int w = rng.get_int32(nx / 3, nx);
    int y0 = rng.get_int32(0, ny - h);
    int x0 = rng.get_int32(0, nx - w);

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const float *color;
            if (x0 <= x && x < x0 + w && y0 <= y && y < y0 + h) {
                color = a;
            } else {
                color = b;
            }
            for (int c = 0; c < 3; ++c) {
                image[c + 3 * x + 3 * nx * y] += color[c];
            }
        }
    }
}

static std::vector<float> generate_voronoi(ppc::random &rng, int ny, int nx,
                                           bool binary, int n_points,
                                           float color_offset, float rect_strength) {
    const float voronoi_max = 0.5 - (color_offset + rect_strength);
    std::vector<float> image(nx * ny * 3, 0.5);
    std::vector<int> ys(n_points);
    std::vector<int> xs(n_points);
    std::vector<float> colors(3 * n_points);
    for (int k = 0; k < n_points; k++) {
        ys[k] = rng.get_int32(0, ny - 1);
        xs[k] = rng.get_int32(0, nx - 1);
        colors[3 * k + 0] = rng.get_float(-1.0, 1.0);
        colors[3 * k + 1] = rng.get_float(-1.0, 1.0);
        colors[3 * k + 2] = rng.get_float(-1.0, 1.0);
        normalize_uniform(&colors[3 * k]);
    }

    offset_rect(rng, ny, nx, image.data(), binary, color_offset, rect_strength);

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int dx0 = xs[0] - x;
            int dy0 = ys[0] - y;
            int dsq = dx0 * dx0 + dy0 * dy0;
            int closest = 0;
            for (int k = 1; k < n_points; k++) {
                int dx = xs[k] - x;
                int dy = ys[k] - y;
                int ndsq = dx * dx + dy * dy;
                if (ndsq < dsq) {
                    dsq = ndsq;
                    closest = k;
                }
            }
            image[0 + 3 * x + 3 * nx * y] += colors[3 * closest + 0] * voronoi_max;
            image[1 + 3 * x + 3 * nx * y] += colors[3 * closest + 1] * voronoi_max;
            image[2 + 3 * x + 3 * nx * y] += colors[3 * closest + 2] * voronoi_max;
        }
    }

    // for bitmaps, use the red channel as the probability P(color == 1.0)
    if (binary) {
        for (int w = 0; w < nx * ny; w++) {
            float p = image[3 * w];
            float val = rng.get_float(0.0, 1.0) < p ? 1.0 : 0.0;
            image[3 * w] = val;
            image[3 * w + 1] = val;
            image[3 * w + 2] = val;
        }
    }

    return image;
}
static void find_avgs(int ny, int nx, const float *data, Result &res) {
    double inner[3] = {};
    double outer[3] = {};
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            double *color;
            if (res.x0 <= x && x < res.x1 && res.y0 <= y && y < res.y1) {
                color = inner;
            } else {
                color = outer;
            }
            for (int c = 0; c < 3; c++) {
                color[c] += (double)data[c + 3 * x + 3 * nx * y];
            }
        }
    }
    double size = (res.y1 - res.y0) * (res.x1 - res.x0);
    for (int c = 0; c < 3; c++) {
        res.inner[c] = inner[c] / (size);
        res.outer[c] = outer[c] / (nx * ny - size);
    }
}

static void test_voronoi(bool is_test, ppc::random &rng, int ny, int nx,
                         Result &e, bool binary, int n_points,
                         float color_offset, float rect_strength) {
    std::vector<float> data = generate_voronoi(
        rng, ny, nx, binary, n_points, color_offset, rect_strength);
    find_avgs(ny, nx, data.data(), e);
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
    } else if (input_type == "precomputed-voronoi" || input_type == "precomputed-voronoi-binary") {
        bool binary = input_type == "precomputed-voronoi-binary";
        int n_points;
        float color_offset, rect_strength;
        CHECK_READ(input_file >> n_points >> color_offset >> rect_strength);
        std::string label;
        CHECK_READ(input_file >> label);
        if (label == "expected-rectangle") {
            Result expected;
            CHECK_READ(input_file >> expected.y0 >> expected.x0 >> expected.y1 >> expected.x1);
            test_voronoi(is_test, rng, ny, nx, expected, binary,
                         n_points, color_offset, rect_strength);
        } else {
            std::cerr << "Invalid test syntax for given type" << std::endl;
            std::exit(1);
        }
    } else {
        std::cerr << "Invalid input type" << std::endl;
        std::exit(1);
    }

    return 0;
}
