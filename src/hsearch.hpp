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

    void search(uint8_t* queriesData, int queriesCount, int queryBitsValue, int* queryIndices,
                uint8_t* querySketches, uint8_t* dataSketches, size_t sketchQwordBytes,
                uint8_t* queryRankingsBase, size_t perQueryTotalBytes, size_t acceptableScoreQwordBytes,
                int evenCandidates, int oddCandidates, int distCandidates, int numPoints, int dimensions) {
        if (queriesCount == 0) {
            return;
        }

        queryBits = queryBitsValue;
        queries = queriesData;
        indices = queryIndices;
        
        queryBytes = (queryBits + 7) >> 3;
        
        // Store additional parameters
        this->querySketches = querySketches;
        this->dataSketches = dataSketches;
        this->sketchQwordBytes = sketchQwordBytes;
        this->queryRankingsBase = queryRankingsBase;
        this->perQueryTotalBytes = perQueryTotalBytes;
        this->acceptableScoreQwordBytes = acceptableScoreQwordBytes;
        this->evenCandidates = evenCandidates;
        this->oddCandidates = oddCandidates;
        this->distCandidates = distCandidates;
        this->numPoints = numPoints;
        this->dimensions = dimensions;
        
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
                processLeaf(link, begin, left);
            } else {
                visit(link, begin, left);
            }
        }
        
        if (left < end) {
            uint64_t raw = readBits(tree, basePos + nodeRightOffset, nodeLRBits);
            bool isLeaf = raw & 1;
            int link = int(raw >> 1);
            if (isLeaf) {
                processLeaf(link, left, end);
            } else {
                visit(link, left, end);
            }
        }
    }
    
    void processLeaf(int leafValue, int begin, int end) {
        const int index = leafValue >> 1;
        const bool flag = leafValue & 1;
        
        const int candidates = flag ? oddCandidates : evenCandidates;
        assert(candidates > 0);
        
        const int half_rangeL = candidates >> 1;
        const int half_rangeR = (candidates+1) >> 1;
        const int start_idx = std::max(0, index - half_rangeL);
        const int end_idx = std::min(index + half_rangeR, numPoints);
        
        // Process each query in this range
        for (int i = begin; i < end; i++) {
            int q = indices[i];
            uint8_t* query_data = queryRankingsBase + perQueryTotalBytes * q;
            
            // Get acceptable score and ranking
            int& acceptable_score = *(int*)(void*)query_data;
            Candidates& ranking = *(Candidates*)(void*)(query_data + acceptableScoreQwordBytes);
            
            uint8_t* query_sketch = querySketches + (size_t)q * sketchQwordBytes;
            
            size_t pointIdPos = pointIdOffset + size_t(start_idx) * pointIdBits;
            
            for (int idx = start_idx; idx < end_idx; idx++, pointIdPos += pointIdBits) {
                assert(0 <= idx && idx < numPoints);
                
                size_t pre_idx = readBits(tree, pointIdPos, pointIdBits);
                assert(pre_idx < size_t(numPoints));
                
                size_t offset = pre_idx * sketchQwordBytes;
                assert(offset + sketchQwordBytes <= sketchQwordBytes * numPoints);
                uint8_t* data_sketch = dataSketches + offset;
                
                int distance = SketchUtils::hammingDistance(query_sketch, data_sketch, sketchQwordBytes, acceptable_score);
                
                if (distance <= acceptable_score) {
                    ranking.emplace_back(distance, pre_idx);
                    
                    if (ranking.full()) {
                        std::sort(ranking.begin(), ranking.end());
                        ranking.resize(std::min((int)std::distance(ranking.begin(), std::unique(ranking.begin(), ranking.end())), distCandidates));
                        
                        if(ranking.size() == distCandidates) {
                            acceptable_score = std::get<0>(ranking.back()) - 1;
                        }
                    }
                }
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
    
    // Additional parameters for sketch distance calculation
    uint8_t* querySketches;
    uint8_t* dataSketches;
    size_t sketchQwordBytes;
    uint8_t* queryRankingsBase;
    size_t perQueryTotalBytes;
    size_t acceptableScoreQwordBytes;
    int evenCandidates;
    int oddCandidates;
    int distCandidates;
    int numPoints;
    int dimensions;
};

#endif // HSEARCH_HPP