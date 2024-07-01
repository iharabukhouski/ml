#pragma once

#include "../op.h"

class Backend {

    public:

        virtual Op add(Op a) = 0;

};
