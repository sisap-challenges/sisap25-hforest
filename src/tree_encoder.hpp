#ifndef TREE_ENCODER_HPP
#define TREE_ENCODER_HPP


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
    
    // Encoded data will be stored here
    uint8_t* encodedData = nullptr;
    size_t encodedSize = 0;

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
    
    ~TreeEncoder() {
        if (encodedData != nullptr) {
            free(encodedData);
        }
    }
    
    // Encode tree structure to binary data
    void encode() {
        // Calculate required buffer size
        size_t headerSize = HilbertTree::getHeaderSize();
        size_t internalNodeBits = nodeInfo.size() * nodeBits;
        size_t leafNodeBits = sortedPoints.size() * pointIdBits;
        size_t totalBits = internalNodeBits + leafNodeBits;
        size_t dataSize = ((totalBits + 7) >> 3) + 16;  // Add extra bytes for finalization
        encodedSize = headerSize + dataSize;
        
        // Allocate buffer for header + data
        encodedData = (uint8_t*)malloc(encodedSize);
        assert(encodedData != nullptr && "Failed to allocate memory for encoded tree data");
        
        // BitWriter writes to data portion (after header)
        BitWriter bitWriter(encodedData + headerSize, dataSize);
        
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
    
    // Create tree and optionally save to file (empty path skips file save)
    HilbertTree* createTree(const std::string& filePath) {
        // Encode if not already encoded
        if (encodedData == nullptr) {
            encode();
        }
        
        // Initialize HilbertTree header
        HilbertTree* tree = (HilbertTree*)encodedData;
        size_t dataSize = encodedSize - HilbertTree::getHeaderSize();
        HilbertTree::initialize(tree, dataSize, nodeBits, nodeBitPosBits, nodeLRBits, pointIdOffset, pointIdBits);
        
        // Write to file only if filePath is not empty
        if (!filePath.empty()) {
            FILE* fp = fopen(filePath.c_str(), "wb");
            assert(fp != nullptr && "Failed to create file");
            
            size_t written __attribute__((unused)) = fwrite(encodedData, 1, encodedSize, fp);
            assert(written == encodedSize && "Failed to write complete tree to file");
            
            fclose(fp);
        }
        
        // Transfer ownership to caller
        encodedData = nullptr;
        encodedSize = 0;
        
        return tree;
    }
    
    // Load existing file (returns nullptr if file doesn't exist)
    static HilbertTree* loadTreeFile(const std::string& filePath) {
        // Check file existence
        struct stat st;
        if (stat(filePath.c_str(), &st) != 0) {
            return nullptr;  // File doesn't exist
        }
        
        // Check header size
        if (st.st_size < (long)HilbertTree::getHeaderSize()) {
            return nullptr;  // Invalid file size
        }
        
        // Open file
        FILE* fp = fopen(filePath.c_str(), "rb");
        if (fp == nullptr) {
            return nullptr;  // Failed to open file
        }
        
        // Allocate memory for entire file
        void* memory = malloc(st.st_size);
        if (memory == nullptr) {
            fclose(fp);
            return nullptr;  // Failed to allocate memory
        }
        
        // Read file content
        size_t bytesRead = fread(memory, 1, st.st_size, fp);
        fclose(fp);
        
        if (bytesRead != (size_t)st.st_size) {
            free(memory);
            return nullptr;  // Failed to read complete file
        }
        
        // Return as HilbertTree pointer
        return (HilbertTree*)memory;
    }
    
    // Free allocated memory
    static void unmapTree(HilbertTree* tree) {
        if (tree) {
            free((void*)tree);
        }
    }
    size_t getDataSize() const {
        return encodedSize - HilbertTree::getHeaderSize();
    }
    
    // Get encoder parameters
    uint64_t getPointIdOffset() const { return pointIdOffset; }
    int getNodeBits() const { return nodeBits; }
    int getNodeBitPosBits() const { return nodeBitPosBits; }
    int getNodeLRBits() const { return nodeLRBits; }
    int getPointIdBits() const { return pointIdBits; }
};

#endif // TREE_ENCODER_HPP