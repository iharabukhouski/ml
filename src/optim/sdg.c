// sgd - stochastic gradient descent
// sgd is the same as gd; the only difference is grad is calculated using a mini-batch

float* sgd(
    int N,
    const float *params,
    const float *grads,
    float lr
) {

    float *new_params = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {

        new_params[i] = params[i] - lr * grads[i];
    }

    return new_params;
}
