#include <array>
#include "../op.h"
#include "cpu.h"
#include "buffer.h"

class cpu::relu : public op {

    template<
        int N,
        int M,
        typename T = float
    >
    buffer<N, M, T>forward(
        buffer<N, M, T> a
    ) {

        return buffer();
    }

    template<
        int N,
        int M,
        typename T = float
    >
    std::array<buffer<N, M, T>, 2> backward(
        buffer<N, M, T> grad 
    ) {

        return std::array<buffer, 2> { grad, grad };
    }
};
