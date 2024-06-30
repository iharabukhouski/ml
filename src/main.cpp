#include <iostream>
#include <array>
#include <initializer_list>
// #include "utils.h"
// #include "ml.h"
#include "tensor.h"
#include "cpu.h"

// a = relu(x @ W + b)

int main() {

    Tensor<float, 2, 2> w = Tensor<float, 2, 2>({
        2, 3,
        4, 5,
    });

    // Tensor<float, 2, 2> w = randn<float, 2, 2>();

    Tensor<float, 2, 1> x = Tensor<float, 2, 1>({
        6,
        7,
    });

    // Tensor<float, 2, 1> x = randn<float, 2, 1>();

    Tensor<float, 2, 1> b = Tensor<float, 2, 1>({
        8,
        9,
    });

    Tensor<float, 2, 1> wx = matmul(w, x);

    Tensor<float, 2, 1> wxb = add(wx, b);

    Tensor<float, 2, 1> z = relu(wxb);
    // Tensor<float, 2, 1> z = wxb;

    // Tensor<float, 2, 1> z = add(x, y);

    for (int i = 0; i < z.buffer->size(); i++) {

        std::cout << z.buffer->at(i) << std::endl;
    }









    // Tensor<float, 2, 1>* x = randn<float, 2, 1>();
    // Tensor<float, 2, 1>* y = randn<float, 2, 1>();

    // dot(x, y)

    // Tensor<float, 3, 2>* w1 = randn<float, 3, 2>();

    // for (int i = 0; i < w1->buffer->size(); i++) {

    //     print(w1->at(i));
    // }

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
