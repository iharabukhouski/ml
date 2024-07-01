#include <iostream>
#include <array>

#include "dtype.h"
#include "logger.h"
// #include "ml.h"
#include "tensor.h"

// a = relu(x @ W + b)

int main() {

    Tensor<f32, 2, 1> a = Tensor<f32, 2, 1>::randn();
    Tensor<f32, 2, 1> b = Tensor<f32, 2, 1>::randn();
    Tensor<f32, 2, 1> c = a + b;

    print(b);

    // Tensor<f32, 2, 2> w = Tensor<f32, 2, 2>({
    //     2, 3,
    //     4, 5,
    // });

    // // Tensor<f32, 2, 2> w = randn<f32, 2, 2>();

    // Tensor<f32, 2, 1> b = Tensor<f32, 2, 1>({
    //     8,
    //     9,
    // });

    // // Tensor<f32, 2, 1> b = randn<f32, 2, 1>();

    // Tensor<f32, 2, 1> x = Tensor<f32, 2, 1>({
    //     6,
    //     7,
    // });

    // // Tensor<f32, 2, 1> x = randn<f32, 2, 1>();

    // Tensor<f32, 2, 1> a = relu(add(matmul(w, x), b));

    // print(a);









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
