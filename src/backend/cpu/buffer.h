#include <array>
#include "cpu.h"

template<
    int N,
    int M,
    typename T = float
>
class cpu::buffer {

    std::array<T, N * M> _buffer;

    public:

        buffer() {

        }

        T& at(
            int i
        ) {

            return this->_buffer->at(i);
        }

        int size() {

            return this->_buffer->size();
        }

};
