// https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction

#include <stddef.h>

// grads = grad(W, X, y)
// V_t = beta * V_t_minus_1 + (1 - beta) * grads
// W = W - alpha * V_t



// stochastic gradsient descent with momentum

// sdg with momentum
// stores "v" over iterations
// "v" - stands for velocity
float* sgd_with_momentum(
    int N,
    const float *params,
    const float *grads,
    float lr,
    float *beta
) {

    float _beta = beta != NULL ? *beta : 0.9;

    // what is initial "v" ???
    float v_t_minus_1 = ???;

    // "beta * v_t_minus_1" - puts less significance on older "v_t"
    // "beta * v_t_minus_1 == beta * (beta * v_t_minus_2 + (1 - beta) * grads) = beta * (beta * (beta * v_t_minus_3 + (1 - beta) * grads) + (1 - beta) * grads)"
    float v_t = _beta * v_t_minus_1 + (1 - _beta) * grads;

    // new_params = params - lr * v




    float *new_params = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {

        new_params[i] = params[i] - lr * grads[i];
    }

    return new_params;
}
