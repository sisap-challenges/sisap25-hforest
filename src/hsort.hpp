#ifndef HSORT_HPP
#define HSORT_HPP


class HilbertSort {
private:
    int dim;
    int bitDepth;
    unsigned int currentBit;
    int currentBitPosition;  // Current bit position being processed (0 to bitDepth-1)
    std::vector<bool> bits;
    int baseAxis;

    int nextNodeId;
    std::vector<int> treeNodes;
    std::vector<int> nodeInfo;  // axis * bitDepth + bitPosition
    int leaf_size;             // Maximum points per leaf node (stop splitting when size <= leaf_size)
    std::vector<int> axes;     // Axis shuffle array (logical to physical conversion)
    std::mt19937& rng;
    uint8_t* base_ptr;         // Base pointer to quantized data array
    size_t data_dim;           // Number of dimensions (for pointer arithmetic)
    
    // Helper function to get actual data pointer from index
    inline uint8_t* getPointPtr(int index) const {
        return base_ptr + data_dim * index;
    }
    
    // Partition processing for index array
    int partition_compact(std::vector<int>& A, int st, int en, int currentAxis) {
        assert(0 <= currentAxis && currentAxis < dim && "Invalid axis in partition function");

        if (st >= en) return st;

        // Convert logical axis number to physical axis number
        int physicalAxis = axes[currentAxis];

        unsigned int di = bits[currentAxis] ? currentBit : 0;
        int i = st - 1;
        int j = en + 1;

        while (true) {
            do { i++; } while(i < j && ((getPointPtr(A[i])[physicalAxis] & currentBit) == di));
            do { j--; } while(i < j && ((getPointPtr(A[j])[physicalAxis] & currentBit) != di));

            if (i >= j) return i;

            std::swap(A[i], A[j]);
        }
    }
    
    // Check if all points have identical coordinates
    bool areAllPointsIdentical(const std::vector<int>& A, int start, int end) {
        if (start >= end) return true;
        
        const unsigned char* firstCoords = getPointPtr(A[start]);
        for (int i = start + 1; i <= end; i++) {
            const unsigned char* currentCoords = getPointPtr(A[i]);
            for (int d = 0; d < dim; d++) {
                if (firstCoords[d] != currentCoords[d]) {
                    return false;
                }
            }
        }
        return true;
    }
    

public:
    HilbertSort(const std::vector<int>& axes, int bitDepth, std::mt19937& rng, uint8_t* base_ptr, size_t data_dim, int leafSizeValue = 1)
        : dim(axes.size()), bitDepth(bitDepth), currentBit(1U << (bitDepth - 1)),
          currentBitPosition(bitDepth - 1), bits(axes.size(), false), baseAxis(0), nextNodeId(0),
          leaf_size(leafSizeValue), axes(axes), rng(rng), base_ptr(base_ptr), data_dim(data_dim) {
        assert(dim > 0);
        assert(bitDepth > 0);
        assert(leaf_size > 0);
        assert(int(axes.size()) == dim && "Axis mapping array length must match dimensions");
        assert(base_ptr != nullptr);
        assert(data_dim > 0);
    }
    
    // Sort method for index array
    int sort(std::vector<int>& points) {
        assert(!points.empty() && "Cannot sort empty points array");
        
        currentBit = 1U << (bitDepth - 1);
        currentBitPosition = bitDepth - 1;
        std::fill(bits.begin(), bits.end(), false);
        baseAxis = 0;
        nextNodeId = 0;
        
        treeNodes.clear();
        nodeInfo.clear();
        
        // HilbertSortSub function for index array
        std::function<int(int, int, int, bool, int)> hilbertSortSub_compact = 
            [&](int st, int en, int currentAxis, bool beforeBit, int sameBitCount) -> int {
                // Stop splitting when number of points <= leaf_size
                if (en - st + 1 <= leaf_size) {
                    // Shuffle points before creating leaf node
                    std::shuffle(points.begin() + st, points.begin() + en + 1, rng);
                    return ((st+en)<<1) + 3;  // Leaf node: (st+en)*2+3 (ensures positive value even for empty range)
                }
                
                int p = partition_compact(points, st, en, currentAxis);
                
                // After partition, if left or right side is empty (p == st or p-1 == en),
                // check if all points have identical coordinates
                if (p == st || p - 1 == en) {
                    if (areAllPointsIdentical(points, st, en)) {
                        // All points are identical, create leaf node early
                        std::shuffle(points.begin() + st, points.begin() + en + 1, rng);
                        return ((st+en)<<1) + 3;  // Leaf node
                    }
                }
                
                int nextAxis = (currentAxis + 1) % dim;
                int nodeId = nextNodeId++;
                int leftChild = -1;
                int rightChild = -1;
                
                treeNodes.push_back(-1);
                treeNodes.push_back(-1);
                // Convert logical axis number to physical axis number and save
                nodeInfo.push_back(axes[currentAxis] * bitDepth + currentBitPosition);
                
                bool di = bits[currentAxis];
                
                if (nextAxis == baseAxis) {
                    if (currentBit == 1) {
                        // Calculation at the lowest bit can be unified regardless of di
                        int firstSubarray = ((st + p) << 1) + 1;
                        int secondSubarray = ((p + en) << 1) + 3;
                        
                        // Tree structure reflection determines left/right based on di
                        if (di) {
                            leftChild = secondSubarray;
                            rightChild = firstSubarray;
                        } else {
                            leftChild = firstSubarray;
                            rightChild = secondSubarray;
                        }
                        
                        treeNodes[nodeId << 1] = leftChild;
                        treeNodes[(nodeId << 1) + 1] = rightChild;
                        return nodeId << 1;  // Internal node (even number)
                    }
                    
                    currentBit >>= 1;
                    
                    int firstSubarray = -1;
                    int secondSubarray = -1;
                    
                    if (2 <= p - st) {  // Recurse only when left side has 2 or more elements
                        int oldBaseAxis = baseAxis;
                        baseAxis = (baseAxis + dim + dim - (beforeBit ? 2 : sameBitCount + 2)) % dim;
                        bits[baseAxis] = !bits[baseAxis];
                        bits[currentAxis] = !bits[currentAxis];
                        firstSubarray = hilbertSortSub_compact(st, p - 1, baseAxis, false, 0);
                        bits[currentAxis] = !bits[currentAxis];
                        bits[baseAxis] = !bits[baseAxis];
                        baseAxis = oldBaseAxis;
                    } else {
                        // Left side is empty or has only one element, represents half-open interval [st, p)
                        firstSubarray = ((st+p)<<1) + 1;  // Leaf node
                    }
                    
                    if (1 <= en - p) {  // Recurse only when right side has 2 or more elements
                        int oldBaseAxis = baseAxis;
                        baseAxis = (baseAxis + dim + dim - (beforeBit ? sameBitCount + 2 : 2)) % dim;
                        secondSubarray = hilbertSortSub_compact(p, en, baseAxis, false, 0);
                        baseAxis = oldBaseAxis;
                    } else {
                        // Right side is empty or has only one element, represents closed interval [p, en]
                        secondSubarray = ((p+en)<<1) + 3;  // Leaf node
                    }
                    
                    if (di) {
                        leftChild = secondSubarray;
                        rightChild = firstSubarray;
                    } else {
                        leftChild = firstSubarray;
                        rightChild = secondSubarray;
                    }
                    
                    currentBit <<= 1;
                } else {
                    int firstSubarray = -1;
                    int secondSubarray = -1;
                    
                    if (2 <= p - st) {  // Recurse only when left side has 2 or more elements
                        firstSubarray = hilbertSortSub_compact(st, p - 1, nextAxis, false, beforeBit ? 1 : sameBitCount + 1);
                    } else {
                        // Left side is empty or has only one element, represents half-open interval [st, p)
                        firstSubarray = ((st+p)<<1) + 1;  // Leaf node
                    }
                    
                    if (1 <= en - p) {  // Recurse only when right side has 2 or more elements
                        bits[currentAxis] = !bits[currentAxis];
                        bits[nextAxis] = !bits[nextAxis];
                        secondSubarray = hilbertSortSub_compact(p, en, nextAxis, true, beforeBit ? sameBitCount + 1 : 1);
                        bits[nextAxis] = !bits[nextAxis];
                        bits[currentAxis] = !bits[currentAxis];
                    } else {
                        // Right side is empty or has only one element, represents closed interval [p, en]
                        secondSubarray = ((p+en)<<1) + 3;  // Leaf node
                    }
                    
                    if (di) {
                        leftChild = secondSubarray;
                        rightChild = firstSubarray;
                    } else {
                        leftChild = firstSubarray;
                        rightChild = secondSubarray;
                    }
                }
                
                treeNodes[nodeId << 1] = leftChild;
                treeNodes[(nodeId << 1) + 1] = rightChild;
                
                return nodeId << 1;  // Return internal node number (even number)
            };
        
        // Execute index array sorting directly
        int rootNodeId = hilbertSortSub_compact(0, points.size() - 1, 0, false, 0);
        
        // If rootNodeId is not 0 (all data went into one leaf), create virtual root node
        if (rootNodeId != 0) {
            assert((rootNodeId & 1) == 1 && "Leaf node ID should be odd");
            assert(treeNodes.empty() && "Expected no nodes to be created");
            assert(nodeInfo.empty() && "Expected no node info to be created");
            
            // Create virtual root node (nodeId=0)
            treeNodes.push_back(rootNodeId);  // left child = leaf node
            treeNodes.push_back(rootNodeId);  // right child = leaf node (same)
            nodeInfo.push_back(0);  // axis=0, bitPos=0
            rootNodeId = 0;
        }
        
        return rootNodeId;
    }
    
    const std::vector<int>& getTreeNodes() const {
        return treeNodes;
    }
    
    const std::vector<int>& getNodeInfo() const {
        return nodeInfo;
    }
    
    int getNodeCount() const {
        return nextNodeId;
    }
};

#endif // HSORT_HPP