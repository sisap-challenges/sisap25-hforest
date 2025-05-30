#ifndef HSORT_GRAPH_HPP
#define HSORT_GRAPH_HPP

#include "bit_quantize.hpp"


class HilbertSortGraph {
private:
    int dim;
    int bitDepth;
    unsigned int currentBit;
    int currentBitPosition;  // Current bit position being processed (0 to bitDepth-1)
    std::vector<bool> bits;
    int baseAxis;

    std::vector<int> axes;     // Axis shuffle array (logical to physical conversion)
    std::mt19937& rng;
    uint8_t* base_ptr;         // Base pointer to quantized data array
    size_t data_dim;           // Number of dimensions (for pointer arithmetic)
    int laneCount;             // Number of values that can be read at once
    
    // Partition processing for index array
    int partition_compact(int* A, int st, int en, int currentAxis) {
        assert(0 <= currentAxis && currentAxis < dim && "Invalid axis in partition function");

        if (st >= en) return st;

        // Convert logical axis number to physical axis number
        int physicalAxis = axes[currentAxis];

        unsigned int di = bits[currentAxis] ? currentBit : 0;
        int i = st - 1;
        int j = en + 1;

        while (true) {
            do { 
                i++; 
            } while(i < j && (BitQuantizer::readAxisBit(base_ptr, A[i], physicalAxis, currentBitPosition, data_dim, bitDepth) == (di != 0)));
            
            do { 
                j--; 
            } while(i < j && (BitQuantizer::readAxisBit(base_ptr, A[j], physicalAxis, currentBitPosition, data_dim, bitDepth) != (di != 0)));

            if (i >= j) return i;

            std::swap(A[i], A[j]);
        }
    }
    

public:
    HilbertSortGraph(const std::vector<int>& axes, int bitDepth, std::mt19937& rng, uint8_t* base_ptr, size_t data_dim, int laneCount = 1)
        : dim(axes.size()), bitDepth(bitDepth), currentBit(1U << (bitDepth - 1)),
          currentBitPosition(bitDepth - 1), bits(axes.size(), false), baseAxis(0),
          axes(axes), rng(rng), base_ptr(base_ptr), data_dim(data_dim), laneCount(laneCount) {
        assert(dim > 0);
        assert(bitDepth > 0);
        assert(int(axes.size()) == dim && "Axis mapping array length must match dimensions");
        assert(base_ptr != nullptr);
        assert(data_dim > 0);
    }
    
    // Sort method for index array (without tree building)
    void sort(int* points, int num_points) {
        assert(points != nullptr && "Cannot sort null points array");
        assert(num_points > 0 && "Cannot sort empty points array");
        
        currentBit = 1U << (bitDepth - 1);
        currentBitPosition = bitDepth - 1;
        std::fill(bits.begin(), bits.end(), false);
        baseAxis = 0;
        
        // Recursive Hilbert sort without tree building
        std::function<void(int, int, int, bool, int)> hilbertSortSub_compact = 
            [&](int st, int en, int currentAxis, bool beforeBit, int sameBitCount) -> void {
                // Continue sorting to completion
                if (st >= en) {
                    return;
                }
                
                int p = partition_compact(points, st, en, currentAxis);
                
                int nextAxis = (currentAxis + 1) % dim;
                
                if (nextAxis == baseAxis) {
                    if (currentBit == 1) {
                        // At lowest bit, stop recursion
                        return;
                    }
                    
                    currentBit >>= 1;
                    currentBitPosition--;
                    
                    if (2 <= p - st) {  // Recurse only when left side has 2 or more elements
                        int oldBaseAxis = baseAxis;
                        baseAxis = (baseAxis + dim + dim - (beforeBit ? 2 : sameBitCount + 2)) % dim;
                        bits[baseAxis] = !bits[baseAxis];
                        bits[currentAxis] = !bits[currentAxis];
                        hilbertSortSub_compact(st, p - 1, baseAxis, false, 0);
                        bits[currentAxis] = !bits[currentAxis];
                        bits[baseAxis] = !bits[baseAxis];
                        baseAxis = oldBaseAxis;
                    }
                    
                    if (1 <= en - p) {  // Recurse only when right side has 2 or more elements
                        int oldBaseAxis = baseAxis;
                        baseAxis = (baseAxis + dim + dim - (beforeBit ? sameBitCount + 2 : 2)) % dim;
                        hilbertSortSub_compact(p, en, baseAxis, false, 0);
                        baseAxis = oldBaseAxis;
                    }
                    
                    currentBit <<= 1;
                    currentBitPosition++;
                } else {
                    if (2 <= p - st) {  // Recurse only when left side has 2 or more elements
                        hilbertSortSub_compact(st, p - 1, nextAxis, false, beforeBit ? 1 : sameBitCount + 1);
                    }
                    
                    if (1 <= en - p) {  // Recurse only when right side has 2 or more elements
                        bits[currentAxis] = !bits[currentAxis];
                        bits[nextAxis] = !bits[nextAxis];
                        hilbertSortSub_compact(p, en, nextAxis, true, beforeBit ? sameBitCount + 1 : 1);
                        bits[nextAxis] = !bits[nextAxis];
                        bits[currentAxis] = !bits[currentAxis];
                    }
                }
            };
        
        // Execute index array sorting directly
        hilbertSortSub_compact(0, num_points - 1, 0, false, 0);
    }
};

#endif // HSORT_GRAPH_HPP