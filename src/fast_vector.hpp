template<class T> class fast_vector {
private:
    int capacity_;
    int size_;
public:
    void init(int capacity) {
        assert(0 <= capacity);
        capacity_ = capacity;
        size_ = 0;
    }
    T & operator[] (int i) {
        assert(0 <= capacity_);
        assert(0 <= i);
        assert(i < size_);
        assert(0 <= size_);
        assert(size_ <= capacity_);
        return ((T*)(void*)(this+1))[i];
    }
    T & back() {
        assert(0 <= capacity_);
        assert(1 <= size_ && "back() requires container to have at least one element");
        assert(size_ <= capacity_);
        return ((T*)(void*)(this+1))[size_-1];
    }
    void clear() {
        assert(0 <= capacity_);
        size_ = 0;
    }
    void resize(int size) {
        assert(0 <= capacity_);
        assert(0 <= size_);
        assert(size_ <= capacity_);
        assert(size <= size_ && "Resize can only shrink the container");
        size_ = size;
    }
    int size() const {
        return size_;
    }
    template<class... TyArgs> inline void emplace_back(TyArgs&&... args) {
        assert(0 <= capacity_);
        assert(0 <= size_);
        assert(size_ <= capacity_);
        assert(size_ < capacity_);
        ::new((T*)(void*)(this+1) + size_)T(std::forward<TyArgs>(args)...);
        ++size_;
    }
    T * begin() {
        assert(0 <= capacity_);
        assert(0 <= size_);
        assert(size_ <= capacity_);
        return (T*)(void*)(this+1);
    }
    T * end() {
        assert(0 <= capacity_);
        assert(0 <= size_);
        assert(size_ <= capacity_);
        return (T*)(void*)(this+1) + size_;
    }
    void erase_to_end(T * p) {
        assert(0 <= capacity_);
        assert(0 <= size_);
        assert(size_ <= capacity_);
        int size = p - begin();
        assert(0 <= size);
        assert(size <= capacity_);
        size_ = size;
    }
    bool full() const {
        assert(0 <= capacity_);
        assert(0 <= size_);
        assert(size_ <= capacity_);
        return size_ == capacity_;
    }
};
