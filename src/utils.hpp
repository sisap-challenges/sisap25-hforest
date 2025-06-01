#ifndef UTILS_HPP
#define UTILS_HPP

// Class for bit-wise write operations
class BitWriter {
private:
    // External buffer pointer and capacity
    uint8_t* buffer;
    size_t capacity;
    size_t currentPos;
    
    // Current bit buffer and count
    uint64_t bitBuffer;
    int bitCount;
    
public:
    BitWriter(uint8_t* buf, size_t cap) : buffer(buf), capacity(cap), currentPos(0), bitBuffer(0), bitCount(0) {}
    
    // Write bits to buffer sequentially
    void writeBits(uint64_t value, uint32_t bitCount) {
        assert(bitCount <= 57 && "writeBits: bitCount must be <= 57");
        assert(value < (1ULL << bitCount) && "writeBits: value has bits set outside of bitCount range");
        
        // Add to bit buffer
        bitBuffer |= (value << this->bitCount);
        this->bitCount += bitCount;
        
        // Output as bytes if 8 or more bits available
        while (8 <= this->bitCount) {
            assert(currentPos < capacity && "BitWriter buffer overflow");
            buffer[currentPos++] = uint8_t(bitBuffer);
            bitBuffer >>= 8;
            this->bitCount -= 8;
        }
    }
    
    // Finalization - flush remaining data
    size_t finalizeBuffer() {
        // Output remaining bits
        if (bitCount > 0) {
            assert(currentPos < capacity && "BitWriter buffer overflow");
            buffer[currentPos++] = uint8_t(bitBuffer);
            bitBuffer = 0;  // Reset bit buffer
            bitCount = 0;   // Reset bit count
        }
        
        // Add fixed 8 bytes of zeros
        for (int i = 0; i < 8; i++) {
            assert(currentPos < capacity && "BitWriter buffer overflow");
            buffer[currentPos++] = 0;
        }
        
        return currentPos;
    }
    
    // Return pointer to buffer data
    const uint8_t* data() const {
        return buffer;
    }
    
    // Buffer size written so far
    size_t size() const {
        return currentPos;
    }

    size_t getOffset() const {
        return currentPos * 8 + bitCount;
    }
};

uint64_t readBits(const uint8_t* base, uint64_t bitPos, uint32_t bitCount) {
    assert(bitCount <= 57 && "readBits: bitCount must be <= 57");

    uint64_t byteOffset = bitPos >> 3;
    uint64_t bitOffset  = bitPos & 7;

    uint64_t buffer = *(uint64_t*)(void*)(base + byteOffset);
    uint64_t shifted = buffer >> bitOffset;
    uint64_t mask = (1ULL << bitCount) - 1;

    uint64_t result = shifted & mask;
    return result;
}

bool readBit(const uint8_t* base, uint64_t bitPos) {
    uint64_t byteOffset = bitPos >> 3;
    uint64_t bitOffset  = bitPos & 7;

    uint8_t buffer = base[byteOffset];
    bool result = (buffer >> bitOffset) & 1;
    return result;
}

#endif // UTILS_HPP