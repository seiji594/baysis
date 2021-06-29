//
// models.cpp
// Baysis
//
// Created by Vladimir Sotskov on 11/06/2021.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#include "models.hpp"

namespace ssm {

    GaussianNoiseTransition::GaussianNoiseTransition(std::size_t x_size):
            Q(x_size, x_size) {};

    LinearGaussianTransition::LinearGaussianTransition(std::size_t x_size):
            GaussianNoiseTransition(x_size),
            A(x_size, x_size) {};

}

