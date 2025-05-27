#ifndef HSEARCH_HPP
#define HSEARCH_HPP


class HilbertTreeSearch {
public:
    HilbertTreeSearch(const HilbertTree* hilbertTree) 
        : tree(hilbertTree->getData())
        , nodeBits(hilbertTree->nodeBits)
        , nodeBitPosBits(hilbertTree->nodeBitPosBits)
        , nodeLRBits(hilbertTree->nodeLRBits)
        , nodeLeftOffset(hilbertTree->nodeLeftOffset)
        , nodeRightOffset(hilbertTree->nodeRightOffset)
        , pointIdOffset(hilbertTree->pointIdOffset)
        , pointIdBits(hilbertTree->pointIdBits) {
    }

    void search(uint8_t* queriesData, int queriesCount, int queryBitsValue, int* results, int* queryIndices) {
        if (queriesCount == 0) {
            return;
        }

        queryBits = queryBitsValue;
        queries = queriesData;
        answers = results;
        indices = queryIndices;
        
        queryBytes = (queryBits + 7) >> 3;
        visit(0, 0, queriesCount);
    }

private:
    void visit(int nodeId, int begin, int end) {
        assert(begin < end);
        
        size_t basePos = size_t(nodeBits) * nodeId;
        int bitPos = int(readBits(tree, basePos, nodeBitPosBits));
        int left = begin;
        int right = end - 1;
        
        while (left <= right) {
            while (left <= right && !readBit(queries, size_t(queryBits) * indices[left] + bitPos))
                left++;
            while (left <= right && readBit(queries, size_t(queryBits) * indices[right] + bitPos))
                right--;
            if (left < right) {
                std::swap(indices[left], indices[right]);
                left++;
                right--;
            }
        }
        
        if (begin < left) {
            uint64_t raw = readBits(tree, basePos + nodeLeftOffset, nodeLRBits);
            bool isLeaf = raw & 1;
            int link = int(raw >> 1);
            if (isLeaf) {
                for (int i = begin; i < left; i++) {
                    answers[indices[i]] = link;
                }
            } else {
                visit(link, begin, left);
            }
        }
        
        if (left < end) {
            uint64_t raw = readBits(tree, basePos + nodeRightOffset, nodeLRBits);
            bool isLeaf = raw & 1;
            int link = int(raw >> 1);
            if (isLeaf) {
                for (int i = left; i < end; i++) {
                    answers[indices[i]] = link;
                }
            } else {
                visit(link, left, end);
            }
        }
    }

    // Copied HilbertTree properties
    const uint8_t* tree;
    int nodeBits;
    int nodeBitPosBits;
    int nodeLRBits;
    int nodeLeftOffset;
    int nodeRightOffset;
    size_t pointIdOffset;
    int pointIdBits;
    
    // Search-related properties
    uint8_t* queries;
    int queryBits;
    int queryBytes;
    int* answers;
    int* indices;
};

#endif // HSEARCH_HPP