#ifndef FOREST_UTILS_HPP
#define FOREST_UTILS_HPP

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


#endif // FOREST_UTILS_HPP