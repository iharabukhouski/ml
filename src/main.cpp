#include <iostream>
#include <array>
#include "utils.h"
#include "ml.h"

int main() {

    std::array<float, 2>* x = randn<float, 2, 1>();

    std::array<float, 3 * 2>* w1 = randn<float, 3, 2>();

    for (int i = 0; i < w1->size(); i++) {

        print(w1->at(i));
    }

    // // std::array<float, 3> b1 = {
    // //     1,
    // //     1,
    // //     1,
    // // };


    // // std::vector<float> x = { 0, 1 };

    // // float matrix_A[2 * 5] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Tensor a = Tensor(2.0f);
    // Tensor b = Tensor(3.0f);
    // Tensor c = Tensor(4.0f);
    // Tensor L = a * b + c;

    // // zero_grad(&L);

    // backward(&L);

    // print("\n");

    // std::vector<Tensor*>* sorted = toposort(&L);

    return 0;
};
