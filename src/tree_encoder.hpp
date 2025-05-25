#ifndef TREE_ENCODER_HPP
#define TREE_ENCODER_HPP

#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include "htree.hpp"
#include "hsort.hpp"
#include "utils.hpp"
#include "assert.hpp"

// Class to convert HilbertSort tree structure to packed binary data
class TreeEncoder {
private:
    const std::vector<int>& treeNodes;
    const std::vector<int>& nodeInfo;
    const std::vector<int>& sortedPoints;
    int bitDepth;
    int dimensions;
    
    int nodeBits;
    int nodeBitPosBits;
    int nodeLRBits;
    int pointIdBits;
    
    int nodeLeftOffset;
    int nodeRightOffset;
    uint64_t pointIdOffset;  // Point ID offset (point ID information follows internal node information)
    
    BitWriter bitWriter;

public:
    TreeEncoder(const std::vector<int>& nodes, const std::vector<int>& info, const std::vector<int>& points, int depth, int dims)
        : treeNodes(nodes), nodeInfo(info), sortedPoints(points), bitDepth(depth), dimensions(dims) {
        
        // Calculate nodeBitPosBits: find maximum value from actual nodeInfo values
        int maxNodeInfo = 0;
        for (size_t i = 0; i < nodeInfo.size(); i++) {
            assert(nodeInfo[i] >= 0 && "nodeInfo value should be non-negative");
            assert(nodeInfo[i] < dimensions * bitDepth && "nodeInfo value should be less than dimensions*bitDepth");
            maxNodeInfo = std::max(maxNodeInfo, nodeInfo[i]);
        }
        
        // Calculate number of bits required to store the maximum value
        nodeBitPosBits = 0;
        while ((1ULL << nodeBitPosBits) <= uint64_t(maxNodeInfo)) {
            nodeBitPosBits++;
        }
        
        // Calculate required bits for node ID
        assert(treeNodes.size() >= 2 && "treeNodes should contain at least one node (2 elements)");
        int maxInternalId = treeNodes.size() - 2;  // Maximum internal node ID (even number)
        int pointsCount = sortedPoints.size();
        // Maximum leaf ID is (st+en)*2+3, which is 4n+1 when st=n, en=n-1
        int maxLeafId = pointsCount * 4 + 1;
        int maxId = std::max(maxInternalId, maxLeafId);
        
        // Minimum 3 bits required (leaf ID=5 when only 1 data point)
        int nodeBitsRequired = 3;
        while ((1ULL << nodeBitsRequired) <= uint64_t(maxId)) {
            nodeBitsRequired++;
        }
        
        
        // Calculate point ID bits - from maximum actual ID in sortedPoints
        int maxPointId = 0;
        for (size_t i = 0; i < sortedPoints.size(); i++) {
            int pointId = sortedPoints[i];
            maxPointId = std::max(maxPointId, pointId);
        }
        
        pointIdBits = 0;
        while ((1ULL << pointIdBits) <= uint64_t(maxPointId)) {
            pointIdBits++;
        }
        
        
        nodeLRBits = nodeBitsRequired;
        
        // Total bits per node
        nodeBits = nodeBitPosBits + nodeLRBits * 2;
        
        nodeLeftOffset = nodeBitPosBits;
        nodeRightOffset = nodeBitPosBits + nodeLRBits;
        
        // Point ID information start position (actual value set in encode())
        pointIdOffset = 0;
    }
    
    // Encode tree structure to binary data
    void encode() {
        bitWriter = BitWriter();
        
        // Encode internal node information
        for (size_t nodeId = 0; nodeId < nodeInfo.size(); nodeId++) {
            bitWriter.writeBits(nodeInfo[nodeId], nodeBitPosBits);
            
            // Write left child node
            int leftIdx = nodeId * 2;
            assert(leftIdx < int(treeNodes.size()));
            bitWriter.writeBits(treeNodes[leftIdx], nodeLRBits);
            
            // Write right child node
            int rightIdx = nodeId * 2 + 1;
            assert(rightIdx < int(treeNodes.size()));
            bitWriter.writeBits(treeNodes[rightIdx], nodeLRBits);
        }
        
        // Update pointIdOffset to current bit position (use exact bit offset, not byte size*8)
        pointIdOffset = bitWriter.getOffset();
        
        // Encode leaf node information
        for (size_t i = 0; i < sortedPoints.size(); i++) {
            int pointId = sortedPoints[i];
            bitWriter.writeBits(uint64_t(pointId), pointIdBits);
        }
        
        // Finalize after encoding all data
        bitWriter.finalizeBuffer();
    }
    
    // Write encoded binary data and parameters to memory for mmap
    // New version: Create memory-mapped file and construct HilbertTree in-place
    HilbertTree* createTreeFile(const std::string& filePath) {
        // Encode if not already encoded
        if (bitWriter.size() == 0) {
            encode();
        }
        
        // Create file (overwrite if exists)
        int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        assert(fd != -1 && "Failed to create file");
        
        // Set file size (header + data)
        size_t headerSize = HilbertTree::getHeaderSize();
        size_t dataSize = bitWriter.size();
        size_t totalSize = headerSize + dataSize;
        
        // Set file size
        int truncResult __attribute__((unused)) = ftruncate(fd, totalSize);
        assert(truncResult == 0 && "Failed to set file size");
        
        // Create memory map
        void* mmapAddr = mmap(NULL, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        assert(mmapAddr != MAP_FAILED && "Failed to create memory map");
        
        // File can be closed after mapping (map remains valid)
        close(fd);
        
        // Initialize HilbertTree header
        HilbertTree::initialize(mmapAddr, dataSize, nodeBits, nodeBitPosBits, nodeLRBits, pointIdOffset, pointIdBits);
        
        // Copy data part
        HilbertTree* tree = (HilbertTree*)mmapAddr;
        memcpy(tree->getDataPtr(), bitWriter.data(), dataSize);
        
        // Sync memory with disk
        msync(mmapAddr, totalSize, MS_SYNC);
        
        return tree;
    }
    
    // Load existing file (returns nullptr if file doesn't exist)
    static HilbertTree* loadTreeFile(const std::string& filePath) {
        // Check file existence
        struct stat st;
        if (stat(filePath.c_str(), &st) != 0) {
            return nullptr;  // File doesn't exist
        }
        
        // Open file
        int fd = open(filePath.c_str(), O_RDWR);
        if (fd == -1) {
            return nullptr;  // Failed to open file
        }
        
        // Memory map the file
        void* mmapAddr = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        if (mmapAddr == MAP_FAILED) {
            return nullptr;  // Failed to map file
        }
        
        // Check header size
        if (st.st_size < (long)HilbertTree::getHeaderSize()) {
            munmap(mmapAddr, st.st_size);
            return nullptr;  // Invalid file size
        }
        
        // Return as HilbertTree pointer
        return (HilbertTree*)mmapAddr;
    }
    
    // Unmap memory
    static void unmapTree(HilbertTree* tree) {
        if (tree) {
            size_t totalSize = tree->getTotalSize();
            munmap((void*)tree, totalSize);
        }
    }
    size_t getDataSize() const {
        return bitWriter.size();
    }
    
    // Get encoder parameters
    uint64_t getPointIdOffset() const { return pointIdOffset; }
    int getNodeBits() const { return nodeBits; }
    int getNodeBitPosBits() const { return nodeBitPosBits; }
    int getNodeLRBits() const { return nodeLRBits; }
    int getPointIdBits() const { return pointIdBits; }
};

#endif // TREE_ENCODER_HPP