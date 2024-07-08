#include <vector>
#include <initializer_list>
#include "tensor.h"

template<
    typename T
>
class Adam {

    float lr = 0.001;
    std::vector<T> params;

    public:

        Adam(
            std::initializer_list<T> params
        ) {

            this->params = std::vector<T>();

            int i = 0;

            for (T param : params) {

                this->params.push_back(param);

                i++;
            }
        }

        void step() {

            for (T param : this->params) {

                print(*param);

                param->data = add(mul(param->grad, this->lr), param->data);
            }
        }
};

namespace tensor {

    struct tensor {

        float data[];

    };

    tensor zeros(
        int N,
        int M
    ) {

        tensor tensor;

        for (int i = 0; i < N * M; i++) {

            tensor.data[i] = 0;
        }

        return tensor;
    };

}

namespace optim {

    namespace sgd {

        struct state {

            size_t params_size;
            tensor::tensor params[];

        };

        struct state init(tensor::tensor params[]) {

            struct state state;

            // std::cout << sizeof(*params) << std::endl;
            // std::cout << sizeof(params[0]) << std::endl;

            state.params_size = sizeof(params) / sizeof(params[0]);

            for (int i = 0; i < state.params_size; i++) {

                state.params[i] = params[i];
            }

            return state;
        }

        state step(
            state state
        ) {

            // std::cout << state.params_size;

            return state;
        }
    }
};
