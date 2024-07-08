#include <iostream>
#include <immintrin.h> // AVX-512 intrinsics
#include <chrono>
#include <vector>
#include <limits>
#include <memory> // For posix_memalign and free
#include <algorithm> // For std::sort
#include <cmath> // For std::sqrt
// Custom aligned allocator with rebind support
template <typename T, std::size_t Alignment>
class aligned_allocator {
public:
    using value_type = T;

    aligned_allocator() noexcept {}

    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc();

        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    // Additional required traits
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    T* address(T& x) const noexcept { return &x; }
    const T* address(const T& x) const noexcept { return &x; }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p) {
        p->~U();
    }

    size_type max_size() const noexcept {
        return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) { return true; }

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) { return false; }
class MyCppyyModule {
public:
    // Scalar implementation of Euclidean distance computation
    void euclidean_distance(const float* x1, const float* y1, const float* z1, const float* x2, const float* y2, const float* z2, float* result, int n) {
        for (int i = 0; i < n; ++i) {
            float dx = x1[i] - x2[i];
            float dy = y1[i] - y2[i];
            float dz = z1[i] - z2[i];
            result[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    // AVX-512 implementation of Euclidean distance computation
    void euclidean_distance_avx512(const float* x1, const float* y1, const float* z1, const float* x2, const float* y2, const float* z2, float* result, int n) {
        int i;
        for (i = 0; i <= n - 16; i += 16) {
            __m512 x1_simd = _mm512_load_ps(&x1[i]);
            __m512 y1_simd = _mm512_load_ps(&y1[i]);
            __m512 z1_simd = _mm512_load_ps(&z1[i]);

            __m512 x2_simd = _mm512_load_ps(&x2[i]);
            __m512 y2_simd = _mm512_load_ps(&y2[i]);
            __m512 z2_simd = _mm512_load_ps(&z2[i]);

            __m512 dx = _mm512_sub_ps(x1_simd, x2_simd);
            __m512 dy = _mm512_sub_ps(y1_simd, y2_simd);
            __m512 dz = _mm512_sub_ps(z1_simd, z2_simd);

            __m512 dx2 = _mm512_mul_ps(dx, dx);
            __m512 dy2 = _mm512_mul_ps(dy, dy);
            __m512 dz2 = _mm512_mul_ps(dz, dz);

            __m512 dist2 = _mm512_add_ps(dx2, _mm512_add_ps(dy2, dz2));
            __m512 dist = _mm512_sqrt_ps(dist2);

            _mm512_store_ps(&result[i], dist);
        }

        // Handle remaining elements
        for (; i < n; ++i) {
            float dx = x1[i] - x2[i];
            float dy = y1[i] - y2[i];
            float dz = z1[i] - z2[i];
            result[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
};

double calculate_percentile(std::vector<double>& sorted_times, double percentile) {
    size_t index = static_cast<size_t>(percentile * sorted_times.size());
    if (index >= sorted_times.size()) {
        index = sorted_times.size() - 1;
    }
    return sorted_times[index];
}

int main() {
    MyCppyyModule module;
    const int n = 1000000; // Data size
    const int alignment = 64; // AVX-512 requires 64-byte alignment
    const int repetitions = 1000; // Number of repetitions for benchmarking

    // Use aligned allocation for vectors
    std::vector<float, aligned_allocator<float, alignment>> x1(n, 1.0f), y1(n, 2.0f), z1(n, 3.0f), x2(n, 4.0f), y2(n, 5.0f), z2(n, 6.0f), result_scalar(n, 0.0f), result_avx512(n, 0.0f);

    std::vector<double> times_scalar, times_avx512;
    times_scalar.reserve(repetitions);
    times_avx512.reserve(repetitions);

    // Warm-up
    for (int i = 0; i < 100; ++i) {
        module.euclidean_distance(x1.data(), y1.data(), z1.data(), x2.data(), y2.data(), z2.data(), result_scalar.data(), 100);
    }
    module.euclidean_distance_avx512(x1.data(), y1.data(), z1.data(), x2.data(), y2.data(), z2.data(), result_avx512.data(), 100);

    // Benchmarking loop
    for (int rep = 0; rep < repetitions; ++rep) {
        // Measure scalar version
        auto start = std::chrono::high_resolution_clock::now();
        module.euclidean_distance(x1.data(), y1.data(), z1.data(), x2.data(), y2.data(), z2.data(), result_scalar.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        times_scalar.push_back(std::chrono::duration<double, std::milli>(end - start).count());

        // Measure AVX-512 version
        start = std::chrono::high_resolution_clock::now();
        module.euclidean_distance_avx512(x1.data(), y1.data(), z1.data(), x2.data(), y2.data(), z2.data(), result_avx512.data(), n);
        end = std::chrono::high_resolution_clock::now();
        times_avx512.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Calculate percentiles
    std::sort(times_scalar.begin(), times_scalar.end());
    std::sort(times_avx512.begin(), times_avx512.end());

    double median_scalar = calculate_percentile(times_scalar, 0.5);
    double median_avx512 = calculate_percentile(times_avx512, 0.5);
    double p95_scalar = calculate_percentile(times_scalar, 0.95);
    double p95_avx512 = calculate_percentile(times_avx512, 0.95);

    // Results
    std::cout << "Median euclidean_distance time: " << median_scalar << " ms\n";
    std::cout << "Median euclidean_distance_avx512 time: " << median_avx512 << " ms\n";
    std::cout << "95th percentile euclidean_distance time: " << p95_scalar << " ms\n";
    std::cout << "95th percentile euclidean_distance_avx512 time: " << p95_avx512 << " ms\n";

    // Verify results
    bool is_same = true;
    for (int i = 0; i < n; ++i) {
        if (result_scalar[i] != result_avx512[i]) {
            is_same = false;
            std::cout << "Mismatch at index " << i << ": scalar=" << result_scalar[i] << ", avx512=" << result_avx512[i] << "\n";
            break;
        }
    }

    if (is_same) {
        std::cout << "Results are the same.\n";
    } else {
        std::cout << "Results are different.\n";
    }

    return 0;
}
