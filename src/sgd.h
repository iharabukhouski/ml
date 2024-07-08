#include <vector>
#include <initializer_list>
#include "tensor.h"

template<
    typename T
>
class sgd {

    float lr;
    std::vector<T> params;

    public:

        sgd(
            float lf,
            std::initializer_list<T> params
        ) {

            this->lr = lr;

            int i = 0;

            for (T param : params) {

                this->params.push_back(param);

                i++;
            }
        }

        void zero_grad() {

            for (int i = 0; i < this->params.size(); i++) {

                this->params->at(i) = 0;
            }
        }

        void step() {

            for (T param : this->params) {

                param->data = add(mul(param->grad, this->lr), param->data);
            }
        }
};
