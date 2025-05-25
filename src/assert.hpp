#undef assert
#ifdef NDEBUG
#define assert(COND)
#else
#define assert(COND) if(!(COND)) my_assert_fail(#COND, __FILE__, __LINE__, __func__)
#endif
#ifndef ASSERT_HPP
#define ASSERT_HPP
#include <sstream>

void my_assert_fail(const char * cond, const char * file, unsigned int line, const char *func) {
    std::stringstream ss;
    ss << "AssertionError(" << file << ":" << line << ":" << func << " ... " << cond << ")";
    throw std::runtime_error(ss.str());
}

#endif // ASSERT_HPP