#include "ppc.h"
#include "prereq.h"

static constexpr float THRESHOLD = 1e-6;

struct Rect {
    int y0;
    int x0;
    int y1;
    int x1;
};

struct TestCase {
    float expected[3];
    std::vector<float> input;
    int ny;
    int nx;
    Rect rect;
};

static std::pair<int, int> random_interval(ppc::random &rng, int n, int s) {
    if (s > 0) {
        int v0 = rng.get_int32(0, n - s);
        int v1 = v0 + s;
        return {v0, v1};
    } else {
        int v0 = rng.get_int32(0, n - 1);
        int v1 = rng.get_int32(0, n - 1);
        if (v0 > v1) {
            std::swap(v0, v1);
        } else if (v0 == v1) {
            if (v0 == 0)
                v1++;
            else
                v0--;
        }
        return {v0, v1};
    }
}

static Rect random_rect(ppc::random &rng, int ny, int nx, int sy, int sx) {
    auto [x0, x1] = random_interval(rng, nx, sx);
    auto [y0, y1] = random_interval(rng, ny, sy);
    return {y0, x0, y1, x1};
}

// One rectangle
static TestCase generate_simple(int ny, int nx, int sy, int sx) {
    ppc::random rng;
    Rect rect = random_rect(rng, ny, nx, sy, sx);

    std::vector<float> data(3 * nx * ny);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            data[(y * nx + x) * 3 + 0] = 0.0;
            data[(y * nx + x) * 3 + 1] = 0.0;
            data[(y * nx + x) * 3 + 2] = 0.0;
        }
    }
    for (int y = rect.y0; y < rect.y1; y++) {
        for (int x = rect.x0; x < rect.x1; x++) {
            data[(y * nx + x) * 3 + 0] = 1.0;
            data[(y * nx + x) * 3 + 1] = 1.0;
            data[(y * nx + x) * 3 + 2] = 1.0;
        }
    }

    return {
        {1.0, 1.0, 1.0},
        data,
        ny,
        nx,
        rect,
    };
}

// All pixels have the same color
static TestCase generate_all_equal(int ny, int nx, int sy, int sx) {
    ppc::random rng;
    Rect rect = random_rect(rng, ny, nx, sy, sx);

    float color[3] = {rng.get_float(), rng.get_float(), rng.get_float()};
    std::vector<float> data(3 * nx * ny);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            data[(y * nx + x) * 3 + 0] = color[0];
            data[(y * nx + x) * 3 + 1] = color[1];
            data[(y * nx + x) * 3 + 2] = color[2];
        }
    }

    return {
        {color[0], color[1], color[2]},
        data,
        ny,
        nx,
        rect,
    };
}

static TestCase generate_gradient(int ny, int nx, int sy, int sx) {
    ppc::random rng;
    Rect rect = random_rect(rng, ny, nx, sy, sx);
    // Generate random colors for the corners
    float top_left[3] = {rng.get_float(), rng.get_float(), rng.get_float()};
    float top_right[3] = {rng.get_float(), rng.get_float(), rng.get_float()};
    float bottom_left[3] = {rng.get_float(), rng.get_float(), rng.get_float()};
    float bottom_right[3] = {rng.get_float(), rng.get_float(), rng.get_float()};

    std::vector<float> data(3 * nx * ny);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            // Linearly interpolate the colors
            double x_fact = double(x) / nx;
            double y_fact = double(y) / ny;
            data[(y * nx + x) * 3 + 0] = float(
                y_fact * (x_fact * top_left[0] + (1.0 - x_fact) * top_right[0]) +
                (1.0 - y_fact) *
                    (x_fact * bottom_left[0] + (1.0 - x_fact) * bottom_right[0]));
            data[(y * nx + x) * 3 + 1] = float(
                y_fact * (x_fact * top_left[1] + (1.0 - x_fact) * top_right[1]) +
                (1.0 - y_fact) *
                    (x_fact * bottom_left[1] + (1.0 - x_fact) * bottom_right[1]));
            data[(y * nx + x) * 3 + 2] = float(
                y_fact * (x_fact * top_left[2] + (1.0 - x_fact) * top_right[2]) +
                (1.0 - y_fact) *
                    (x_fact * bottom_left[2] + (1.0 - x_fact) * bottom_right[2]));
        }
    }

    // Compute the expected color. It is the average of the corners.
    double x_fact = 0.5 * double(rect.x0 + rect.x1 - 1) / nx;
    double y_fact = 0.5 * double(rect.y0 + rect.y1 - 1) / ny;
    float r =
        float(y_fact * (x_fact * top_left[0] + (1.0 - x_fact) * top_right[0]) +
              (1.0 - y_fact) *
                  (x_fact * bottom_left[0] + (1.0 - x_fact) * bottom_right[0]));
    float g =
        float(y_fact * (x_fact * top_left[1] + (1.0 - x_fact) * top_right[1]) +
              (1.0 - y_fact) *
                  (x_fact * bottom_left[1] + (1.0 - x_fact) * bottom_right[1]));
    float b =
        float(y_fact * (x_fact * top_left[2] + (1.0 - x_fact) * top_right[2]) +
              (1.0 - y_fact) *
                  (x_fact * bottom_left[2] + (1.0 - x_fact) * bottom_right[2]));

    return {
        {r, g, b},
        data,
        ny,
        nx,
        rect,
    };
}

static TestCase generate_small_noise(int ny, int nx, int sy, int sx) {
    ppc::random rng;
    Rect rect = random_rect(rng, ny, nx, sy, sx);

    float color[3] = {rng.get_float(), rng.get_float(), rng.get_float()};
    std::vector<float> data(3 * nx * ny);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            data[(y * nx + x) * 3 + 0] = color[0];
            data[(y * nx + x) * 3 + 1] = color[1];
            data[(y * nx + x) * 3 + 2] = color[2];
        }
    }

    // Add some small noise
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 3 * nx * ny; i++) {
        retry_plus:
            int x = rng.get_int32(rect.x0, rect.x1 - 1);
            int y = rng.get_int32(rect.y0, rect.y1 - 1);
            if (!(data[(y * nx + x) * 3 + c] < 1.0f))
                goto retry_plus;
            data[(y * nx + x) * 3 + c] += std::numeric_limits<float>::epsilon();
        }
        for (int i = 0; i < 3 * nx * ny; i++) {
        retry_minus:
            int x = rng.get_int32(rect.x0, rect.x1 - 1);
            int y = rng.get_int32(rect.y0, rect.y1 - 1);
            if (!(data[(y * nx + x) * 3 + c] > std::numeric_limits<float>::epsilon()))
                goto retry_minus;
            data[(y * nx + x) * 3 + c] -= std::numeric_limits<float>::epsilon();
        }
    }

    return {
        {color[0], color[1], color[2]},
        data,
        ny,
        nx,
        rect,
    };
}

static TestCase generate_color_rects(int ny, int nx, int sy, int sx) {
    ppc::random rng;
    Rect rect = random_rect(rng, ny, nx, sy, sx);

    std::vector<double> data(3 * nx * ny);
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    for (int i = 0; i < 100; i++) {
        Rect new_rect = random_rect(rng, ny, nx, -1, -1);
        float color[3] = {rng.get_float(), rng.get_float(), rng.get_float()};
        for (int y = new_rect.y0; y < new_rect.y1; y++) {
            for (int x = new_rect.x0; x < new_rect.x1; x++) {
                data[(y * nx + x) * 3 + 0] += color[0];
                data[(y * nx + x) * 3 + 1] += color[1];
                data[(y * nx + x) * 3 + 2] += color[2];
            }
        }
        double size = std::max(0, std::min(new_rect.x1, rect.x1) -
                                      std::max(new_rect.x0, rect.x0)) *
                      std::max(0, std::min(new_rect.y1, rect.y1) -
                                      std::max(new_rect.y0, rect.y0));
        r += size * color[0];
        g += size * color[1];
        b += size * color[2];
    }

    double r_scale = 0.0;
    double g_scale = 0.0;
    double b_scale = 0.0;
    for (int i = 0; i < nx * ny; i++) {
        r_scale = std::max(r_scale, data[3 * i + 0]);
        g_scale = std::max(g_scale, data[3 * i + 1]);
        b_scale = std::max(b_scale, data[3 * i + 2]);
    }
    r_scale = 1.0 / r_scale;
    g_scale = 1.0 / g_scale;
    b_scale = 1.0 / b_scale;
    for (int i = 0; i < nx * ny; i++) {
        data[3 * i + 0] *= r_scale;
        data[3 * i + 1] *= g_scale;
        data[3 * i + 2] *= b_scale;
    }
    r_scale /= (rect.x1 - rect.x0) * (rect.y1 - rect.y0);
    g_scale /= (rect.x1 - rect.x0) * (rect.y1 - rect.y0);
    b_scale /= (rect.x1 - rect.x0) * (rect.y1 - rect.y0);

    std::vector<float> fdata(3 * nx * ny);
    std::copy(data.begin(), data.end(), fdata.begin());

    return {
        {float(r * r_scale), float(g * g_scale), float(b * b_scale)},
        fdata,
        ny,
        nx,
        rect,
    };
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
    int ny;
    int nx;
    int sy;
    int sx;
    CHECK_READ(input_file >> input_type);
    if (input_type == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        CHECK_READ(input_file >> input_type);
    }
    CHECK_READ(input_file >> ny);
    CHECK_READ(input_file >> nx);
    CHECK_READ(input_file >> sy);
    CHECK_READ(input_file >> sx);
    CHECK_END(input_file);

    if (nx <= 0 || ny <= 0) {
        std::cerr << "Invalid input size" << std::endl;
        return 3;
    }

    TestCase test_case;
    if (input_type == "simple") {
        test_case = generate_simple(ny, nx, sy, sx);
    } else if (input_type == "all_equal") {
        test_case = generate_all_equal(ny, nx, sy, sx);
    } else if (input_type == "gradient") {
        test_case = generate_gradient(ny, nx, sy, sx);
    } else if (input_type == "small_noise") {
        test_case = generate_small_noise(ny, nx, sy, sx);
    } else if (input_type == "color_rects") {
        test_case = generate_color_rects(ny, nx, sy, sx);
    } else {
        std::cerr << "Invalid input type" << std::endl;
        return 3;
    }

    ppc::perf timer;
    timer.start();
    const Result result = calculate(
        ny, nx, test_case.input.data(),
        test_case.rect.y0, test_case.rect.x0, test_case.rect.y1, test_case.rect.x1);
    timer.stop();
    timer.print_to(*stream);

    if (test) {
        const float error = std::max({std::abs(result.avg[0] - test_case.expected[0]),
                                      std::abs(result.avg[1] - test_case.expected[1]),
                                      std::abs(result.avg[2] - test_case.expected[2])});
        const bool pass = error <= THRESHOLD;
        if (!pass) {
            bool small = ny * nx <= 200;
            stream->precision(std::numeric_limits<float>::max_digits10 - 1);
            *stream
                << "result\tfail\n"
                << "error_magnitude\t" << std::scientific << error << '\n'
                << "threshold\t" << std::scientific << THRESHOLD << '\n'
                << "ny\t" << ny << '\n'
                << "nx\t" << nx << '\n'
                << "y0\t" << test_case.rect.y0 << '\n'
                << "x0\t" << test_case.rect.x0 << '\n'
                << "y1\t" << test_case.rect.y1 << '\n'
                << "x1\t" << test_case.rect.x1 << '\n'
                << "avg\t" << std::scientific << result.avg[0]
                << '\t' << std::scientific << result.avg[1]
                << '\t' << std::scientific << result.avg[2] << '\n'
                << "size\t" << (small ? "small" : "large") << '\n';
            if (small) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        *stream << "triple";
                        for (int c = 0; c < 3; ++c) {
                            *stream << '\t' << std::scientific << test_case.input[c + 3 * x + 3 * nx * y];
                        }
                        *stream << '\n';
                    }
                }
            }
        } else {
            *stream << "result\tpass\n";
        }
    } else {
        *stream << "result\tdone\n";
    }
    *stream << std::flush;
}
