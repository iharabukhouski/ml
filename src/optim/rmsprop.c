#include <stddef.h>
#include <math.h>

// RMSProps uses adaptive learning rate but does not use momentum
// https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
struct State {

    float t;
    float m;
    float v;

    float alpha;
    float beta1;
    float beta2;
    float lambda;
    float epsilon;

};

struct State init(
    float *alpha, // learning rate (default 0.001)
    float *beta1, // mean scaling (default 0.9)
    float *beta2, // variance scaling (default 0.999)
    float *lambda, // weight decay (default 0.01)
    float *epsilon // (default 1e-8)
) {

    struct State state;

    state.t = 0;
    state.m = 0;
    state.v = 0;

    state.alpha = alpha != NULL ? *alpha : 0.001;
    state.beta1 = beta1 != NULL ? *beta1 : 0.9;
    state.beta2 = beta2 != NULL ? *beta2 : 0.999;
    state.lambda = lambda != NULL ? *lambda : 0.01;
    state.epsilon = epsilon != NULL ? *epsilon : 1e-8;

    return state;
}

void step(
    struct State state,
    Tensor *params,
    Tensor *grads
) {

    state.t += 1;

    m = state.beta1 * m + (1 - state.beta1) * grads;
    v = state.beta2 * v + (1 - state.beta2) * pow(grads, 2);

    float m_hat = m / (1 - pow(state.beta1, state.t));
    float v_hat = v / (1 - pow(state.beta2, state.t));

    params = params - ((state.alpha * m_hat) / (sqrt(v_hat) + state.epsilon));
}
