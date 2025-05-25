#ifndef FOREST_UTILS_HPP
#define FOREST_UTILS_HPP

#include <vector>
#include <random>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <immintrin.h>
#include "timing.hpp"

namespace py = pybind11;

// Sketch utility class (bit manipulation helper functions)
class SketchUtils {
public:
    // Set specified bit to 1
    static inline void setBit(uint8_t* data, int pos) {
        data[pos >> 3] |= (1 << (pos & 7));
    }
    
    // Set specified bit to 0
    static inline void clearBit(uint8_t* data, int pos) {
        data[pos >> 3] &= ~(1 << (pos & 7));
    }
    
    // Get value of specified bit
    static inline bool getBit(const uint8_t* data, int pos) {
        return (data[pos >> 3] & (1 << (pos & 7))) != 0;
    }
    
    // Calculate Hamming distance (number of different bits)
    static inline int hammingDistance(const uint8_t* a, const uint8_t* b, int bytes, int max_acceptable_score) {
        int i = 0;
        int distance = 0;
        
        #ifdef __POPCNT__
        // Fast path when _mm_popcnt_u64 is available
        // Process in 8-byte (64-bit) units
        if (i + 8 <= bytes) {
            uint64_t a_val = *(uint64_t*)(void*)(a + i);
            uint64_t b_val = *(uint64_t*)(void*)(b + i);
            uint64_t xor_result = a_val ^ b_val;
            int cnt = _mm_popcnt_u64(xor_result);
            if(30 <= cnt) {
                return bytes * 16;
            }
            distance += cnt;
            i += 8;
            if (i + 8 <= bytes) {
                uint64_t a_val = *(uint64_t*)(void*)(a + i);
                uint64_t b_val = *(uint64_t*)(void*)(b + i);
                uint64_t xor_result = a_val ^ b_val;
                int cnt = _mm_popcnt_u64(xor_result);
                if(30 <= cnt) {
                    return bytes * 16;
                }
                distance += cnt;
                i += 8;
                if (i + 8 <= bytes) {
                    uint64_t a_val = *(uint64_t*)(void*)(a + i);
                    uint64_t b_val = *(uint64_t*)(void*)(b + i);
                    uint64_t xor_result = a_val ^ b_val;
                    int cnt = _mm_popcnt_u64(xor_result);
                    if(30 <= cnt) {
                        return bytes * 16;
                    }
                    distance += cnt;
                    i += 8;
                    for (; i + 8 <= bytes; i += 8) {
                        uint64_t a_val = *(uint64_t*)(void*)(a + i);
                        uint64_t b_val = *(uint64_t*)(void*)(b + i);
                        uint64_t xor_result = a_val ^ b_val;
                        int cnt = _mm_popcnt_u64(xor_result);
                        distance += cnt;
                        if(max_acceptable_score < distance) {
                            return bytes * 16;
                        }
                    }
                }
            }
        }
        
        // Process remaining 4 bytes (32 bits) if any
        if (i + 4 <= bytes) {
            uint32_t a_val = *(uint32_t*)(void*)(a + i);
            uint32_t b_val = *(uint32_t*)(void*)(b + i);
            uint32_t xor_result = a_val ^ b_val;
            
            // Use 32-bit population count instruction
            distance += _mm_popcnt_u32(xor_result);
            i += 4;
        }
        #endif
        
        // Process remaining bytes (0-3 bytes) or when POPCNT instruction is not available
        for (; i < bytes; i++) {
            uint8_t diff = a[i] ^ b[i];
            
            #ifdef __POPCNT__
            // 8-bit population count
            distance += _mm_popcnt_u32(diff);
            #else
            // Count bits one by one
            while (diff) {
                distance++;
                diff &= diff - 1; // Clear the lowest 1 bit
            }
            #endif
        }
        
        return distance;
    }
};

// Utility functions for spatial indexing
#ifndef FOREST_UTILS_CLASS_DEFINED
class ForestUtils {
public:
    // Perform quantization only
    static void transform_point(const float* point, uint8_t* result, int in_dim, float rate) {
        for (int i = 0; i < in_dim; i++) {
            result[i] = (uint8_t)std::min(std::max(0, (int)(point[i] * rate + 128)), 255);
        }
    }
    
    // Generate sketch from quantized point (positive if >= 128, negative if < 128)
    static void generate_sketch_from_quantized(const uint8_t* point, uint8_t* sketch_data, int dimensions, int sketch_qword_bytes) {
        // Initialize sketch with zeros
        std::memset(sketch_data, 0, sketch_qword_bytes);
        
        // Set sign bit for each dimension (based on quantized value)
        for (int i = 0; i < dimensions; i++) {
            if (point[i] >= 128) {  // Positive value if >= 128
                SketchUtils::setBit(sketch_data, i);
            }
        }
    }
    
    // Generate sketch batch from quantized data
    static void generate_sketches_batch(const uint8_t* quantized_data, 
                                          int num_points, int dimensions, uint8_t * sketches, ForestTiming& timing) {
        ScopedTimer timer(timing.fit_sketch);
        
        // Calculate sketch byte size
        size_t sketch_qword_bytes = (size_t((dimensions + 7) >> 3) + 7) & ~7;
        
        // Parallel processing with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < num_points; i++) {
            // Cast to size_t to prevent overflow
            size_t point_offset = size_t(i) * dimensions;
            size_t sketch_offset = size_t(i) * sketch_qword_bytes;
            
            const uint8_t* point_ptr = quantized_data + point_offset;
            uint8_t* sketch_ptr = sketches + sketch_offset;
            
            // Generate sketch for this point
            generate_sketch_from_quantized(point_ptr, sketch_ptr, dimensions, sketch_qword_bytes);
        }
    }
    
    
    // Method to quantize multiple queries at once
    static std::vector<uint8_t> quantize_queries_batch(
                                                    const py::array_t<float>& queries,
                                                    int dimensions,
                                                    float rate,
                                                    ForestTiming& timing) {
        py::buffer_info buf = queries.request();
        int num_queries = buf.shape[0];
        float* query_ptr = (float*)buf.ptr;

        // Result buffer (calculated with 64-bit integers)
        std::vector<uint8_t> result(size_t(num_queries) * dimensions);

        // Process and quantize each query
        #pragma omp parallel for
        for (int i = 0; i < num_queries; i++) {
            // Transform query point (calculated with 64-bit integers)
            float* query_point = query_ptr + size_t(i) * dimensions;

            // Quantize (offset calculated with 64-bit integers)
            transform_point(query_point, &result[size_t(i) * dimensions], dimensions, rate);
        }

        return result;
    }

    // Create batch from prepared quantized queries using BitWriter
    static std::vector<uint8_t> prepare_queries_batch(const std::vector<uint8_t>& quantized_queries,
                                                    int num_queries, int dimensions,
                                                    ForestTiming& timing) {
        // Result buffer
        BitWriter writer;

        // Process each query
        for (int i = 0; i < num_queries; i++) {
            // Write quantized data according to axis mapping
            for (int d = 0; d < dimensions; d++) {
                writer.writeBits(quantized_queries[size_t(i) * dimensions + d], 8);
            }
        }

        // Return result
        writer.finalizeBuffer();
        std::vector<uint8_t> result(writer.size());
        std::copy(writer.data(), writer.data() + writer.size(), result.data());

        return result;
    }
};
#endif // FOREST_UTILS_CLASS_DEFINED

#endif // FOREST_UTILS_HPP