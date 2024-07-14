#include <array>
#include "tensor.h"
#include "sgd.h"

int main() {

    int num_steps = 5;
    float lr = 0.001;

    auto W = randn<1, 1>();
    auto b = randn<1, 1>();

    auto optimizer = sgd(
        lr,
        {
            W,
            b
        }
    );

    auto x = randn<1, 1>();
    auto y = randn<1, 1>();

    for (int i = 0; i < num_steps; i++) {

        auto y_hat = relu(add(matmul(W, x), b));

        auto l = sub(y, y_hat);

        optimizer.zero_grad();

        l.backward();

        optimizer.step();
    }

    return 0;
}
