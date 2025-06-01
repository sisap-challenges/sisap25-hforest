#ifndef BIT_QUANTIZE_HPP
#define BIT_QUANTIZE_HPP

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "utils.hpp"

class BitQuantizer {
public:
    // Quantize a single float value to specified bit width
    static inline uint64_t quantizeValue(float value, float rate, int64_t midpoint) {
        // Scale and offset the value
        int64_t quantized = (int64_t)(value * rate + midpoint);
        
        // Clamp to valid range [0, midpoint * 2)
        return (uint64_t)std::min(std::max((int64_t)0, quantized), midpoint + midpoint - 1);
    }
    
    // Transform points to bit-packed format (can handle any number of points)
    static void transformToBits(const float* input, BitWriter& writer, 
                               size_t numPoints, int dimensions, float rate, int bitWidth) {
        // Pre-calculate midpoint = 2^(bitWidth-1)
        int64_t midpoint = 1LL << (bitWidth - 1);
        
        // Process all values in a single loop
        size_t totalValues = numPoints * dimensions;
        for (size_t i = 0; i < totalValues; i++) {
            uint64_t quantized = quantizeValue(input[i], rate, midpoint);
            writer.writeBits(quantized, bitWidth);
        }
    }

    static void transformToBitsForGraph(const float* input, uint8_t * output, 
                               size_t numPoints, int dimensions, float rate) {
        constexpr size_t cache_line_bits = 6;  // 2^6 = 64
        constexpr size_t cache_line_size = 1 << cache_line_bits;
        constexpr size_t cache_line_mask = cache_line_size - 1;
        
        size_t totalValues = numPoints * dimensions;
        size_t num_batches = (totalValues + cache_line_mask) >> cache_line_bits;
        
        #pragma omp parallel for
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t start = batch << cache_line_bits;
            size_t end = std::min(start + cache_line_size, totalValues);
            
            for (size_t i = start; i < end; i++) {
                output[i] = quantizeValue(input[i], rate, 128);
            }
        }
    }

    // Compress existing bit-packed data by removing MSB (bitWidth -> bitWidth-1)
    // MSB information will be reconstructed from sketch during distance calculation
    static void compressBitPackedData(const uint8_t* input, BitWriter& writer,
                                     size_t numPoints, int dimensions, int bitWidth) {
        int compressedBitWidth = bitWidth - 1;
        uint64_t mask = (1ULL << compressedBitWidth) - 1;
        
        size_t totalValues = numPoints * dimensions;
        for (size_t i = 0; i < totalValues; i++) {
            size_t bitPos = i * bitWidth;
            uint64_t value = readBits(input, bitPos, bitWidth);
            uint64_t compressed = value & mask;  // Remove MSB
            writer.writeBits(compressed, compressedBitWidth);
        }
    }
    
    // Calculate bit position for a specific point and dimension
    static inline size_t calculateBitPosition(size_t pointIndex, int dimensionIndex, 
                                            int dimensions, int bitWidth) {
        return (pointIndex * dimensions + dimensionIndex) * bitWidth;
    }
    
    // Read a specific bit for a given axis (for HilbertSort compatibility)
    static inline bool readAxisBit(const uint8_t* data, size_t pointIndex, int axis, 
                                  int bitPosition, int dimensions, int bitWidth) {
        size_t valueBitPos = calculateBitPosition(pointIndex, axis, dimensions, bitWidth);
        // bitPosition 0 is LSB in little-endian
        size_t totalBitPos = valueBitPos + bitPosition;
        return readBit(data, totalBitPos);
    }
    
    // Calculate squared distance between a point and a query with optimized batch reading
    static int calculateSquaredDistance(const uint8_t* pointData, size_t pointIndex,
                                       const uint8_t* queryData, size_t queryIndex,
                                       int dimensions, int bitWidth, int laneCount) {
        int distance = 0;
        uint64_t valueMask = (1ULL << bitWidth) - 1;
        
        int d = 0;
        while (d < dimensions) {
            // Determine how many values to read in this batch
            int batchSize = std::min(laneCount, dimensions - d);
            int batchBits = batchSize * bitWidth;
            
            // Read batch from point data
            size_t pointBitPos = calculateBitPosition(pointIndex, d, dimensions, bitWidth);
            uint64_t pointBatch = readBits(pointData, pointBitPos, batchBits);
            
            // Read batch from query data
            size_t queryBitPos = calculateBitPosition(queryIndex, d, dimensions, bitWidth);
            uint64_t queryBatch = readBits(queryData, queryBitPos, batchBits);
            
            // Process each value in the batch
            for (int i = 0; i < batchSize; i++) {
                int val1 = (int)(pointBatch & valueMask);
                int val2 = (int)(queryBatch & valueMask);
                int diff = val1 - val2;
                distance += diff * diff;
                
                // Shift to next value
                pointBatch >>= bitWidth;
                queryBatch >>= bitWidth;
            }
            
            d += batchSize;
        }
        
        return distance;
    }
    
    // Calculate squared distance between compressed database point and scaled float query
    // Database point uses (bitWidth-1) bits + MSB from sketch
    static float calculateSquaredDistanceCompressed(const uint8_t* compressedPointData, size_t pointIndex,
                                                   const uint8_t* pointSketch, size_t sketchStride,
                                                   const float* scaledQueries, size_t queryIndex,
                                                   int dimensions, int bitWidth, int laneCount) {
        float distance = 0.0f;
        int compressedBitWidth = bitWidth - 1;
        uint64_t compressedMask = (1ULL << compressedBitWidth) - 1;
        uint64_t msbValue = 1ULL << compressedBitWidth;  // MSB value for reconstruction
        
        const float* queryPtr = scaledQueries + queryIndex * dimensions;
        
        int d = 0;
        while (d < dimensions) {
            // Determine how many values to read in this batch
            int batchSize = std::min(laneCount, dimensions - d);
            int compressedBatchBits = batchSize * compressedBitWidth;
            
            // Read compressed batch from point data
            size_t pointBitPos = pointIndex * dimensions * compressedBitWidth + d * compressedBitWidth;
            uint64_t pointBatch = readBits(compressedPointData, pointBitPos, compressedBatchBits);
            
            // Read MSB batch from sketch (max 57 bits at once)
            uint64_t msbBatch = readBits(pointSketch, d, batchSize);
            
            // Process each value in the batch
            for (int i = 0; i < batchSize; i++) {
                // Get compressed value from point
                int compressedVal = (int)(pointBatch & compressedMask);
                
                // Get MSB from batch
                bool msb = (msbBatch >> i) & 1;
                int val1 = msb ? (compressedVal | msbValue) : compressedVal;
                
                // Calculate squared difference
                float diff = (float)val1 - queryPtr[d + i];
                distance += diff * diff;
                
                // Shift to next value
                pointBatch >>= compressedBitWidth;
            }
            
            d += batchSize;
        }
        
        return distance;
    }
    
    // Generate sketch from bit-packed data
    static void generateSketchFromBitPacked(const uint8_t* data, uint8_t* sketch, 
                                           size_t pointIndex, int dimensions, int bitWidth, int sketchBytes) {
        // Initialize sketch with zeros
        std::memset(sketch, 0, sketchBytes);
        
        // Calculate midpoint for the bit width
        int64_t midpoint = 1LL << (bitWidth - 1);
        
        // Set sign bit for each dimension
        for (int d = 0; d < dimensions; d++) {
            size_t bitPos = calculateBitPosition(pointIndex, d, dimensions, bitWidth);
            uint64_t value = readBits(data, bitPos, bitWidth);
            if (value >= (uint64_t)midpoint) {
                SketchUtils::setBit(sketch, d);
            }
        }
    }
};

#endif // BIT_QUANTIZE_HPP