/**
 * An adapter of libscapi's pseudorandom generators to the
 * C++ UniformRandomBitGenerator concept
 */
#pragma once
#include "primitives/Prg.hpp"

template<typename T>
class prg_adapter {
private:
    PseudorandomGenerator& prg;
    std::vector<byte> buf;

public:
    prg_adapter(PseudorandomGenerator& prg): prg(prg), buf(sizeof(T)) {}
    using result_type = T;
    T min() { return std::numeric_limits<T>::min(); }
    T max() { return std::numeric_limits<T>::max(); }
    T operator()() {
        prg.getPRGBytes(buf, 0, sizeof(T));
        return *reinterpret_cast<T*>(buf.data());
    }
};
