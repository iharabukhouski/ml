// gd - gradient descent
// grads - calculated based on all training examples

float* gd(
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
