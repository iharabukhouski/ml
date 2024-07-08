#include <iostream>
#include <array>

#include "logger.h"
#include "dtype.h"
#include "tensor.h"
// #include "adam.h"

// a = relu(x @ W + b);

// auto W = randn<2, 3>();
// auto b = randn<3, 1>();
// auto x = randn<2, 1>();

// a = relu(add(matmul(W, x), b))

// backward(a);

void foo(int a[]) {

    std::cout << sizeof(a);
}

int main() {

    int b[2];

    std::cout << sizeof(b);

    // int a = 5;
    // int b[a];

    // auto W = tensor::zeros(1, 1);
    // auto b = tensor::zeros(1, 1);

    // tensor::tensor params[] = { W, b, W, b };

    // auto state = optim::sgd::init(params);

    // state = optim::sgd::step(state);

    // auto W = randn<1, 1>();
    // auto b = randn<1, 1>();

    // auto optimizer = Adam({
    //     &W,
    //     &b
    // });

    // auto x = randn<1, 1>();

    // auto h = mul(W, x);

    // auto y = add(h, b);

    // backward(y);

    // optimizer.step();

    return 0;
};
