#ifndef HTREE_HPP
#define HTREE_HPP

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include "utils.hpp"

// File-mapped HilbertTree structure
// Placed directly on mmap'ed memory, with data following the structure
struct HilbertTree {
    // Tree data size (excluding header)
    uint64_t dataSize;
    
    // Node-related information
    int nodeBits;
    int nodeBitPosBits;
    int nodeLRBits;
    int nodeLeftOffset;
    int nodeRightOffset;
    
    // Point ID information
    uint64_t pointIdOffset;
    int pointIdBits;
    
    // Initialization function (alternative to constructor)
    static void initialize(void* memory, 
                      uint64_t dataSizeValue,
                      int nodeBitsValue,
                      int nodeBitPosBitsValue,
                      int nodeLRBitsValue,
                      uint64_t pointIdOffsetValue,
                      int pointIdBitsValue) {
        HilbertTree* tree = (HilbertTree*)(memory);
        tree->dataSize = dataSizeValue;
        tree->nodeBits = nodeBitsValue;
        tree->nodeBitPosBits = nodeBitPosBitsValue;
        tree->nodeLRBits = nodeLRBitsValue;
        tree->pointIdOffset = pointIdOffsetValue;
        tree->pointIdBits = pointIdBitsValue;
        
        // Calculate offset values
        tree->nodeLeftOffset = nodeBitPosBitsValue;
        tree->nodeRightOffset = nodeBitPosBitsValue + nodeLRBitsValue;
    }
    
    // Get pointer to data part (memory following the structure)
    const uint8_t* getData() const {
        return (const uint8_t*)(void*)(this + 1);
    }
    
    // Get writable pointer to data part
    uint8_t* getDataPtr() {
        return (uint8_t*)(void*)(this + 1);
    }
    
    // Get structure size (for memory layout calculation)
    static size_t getHeaderSize() {
        return sizeof(HilbertTree);
    }
    
    // Get total size (header + data)
    size_t getTotalSize() const {
        return getHeaderSize() + dataSize;
    }
};

#endif // HTREE_HPP