//
// models.cpp
// Baysis
//
// Created by Vladimir Sotskov on 11/06/2021.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#include "models.hpp"

namespace ssm {

//    LinearModel::LinearModel(std::size_t input_rows, std::size_t input_cols) :
//            inputM(input_rows, input_cols), controlM(), controls() { }

//    LinearModel::LinearModel(std::size_t input_rows, std::size_t input_cols, std::size_t control_size) :
//            inputM(input_rows, input_cols), controlM(input_rows, control_size), controls(control_size) { }


//    LGTransition::LGTransition(std::size_t state_size) :
//            LinearModel(state_size, state_size), Q(state_size, state_size), LQ(state_size) { }

//    LGTransition::LGTransition(std::size_t state_size, std::size_t control_size) :
//            LinearModel(state_size, state_size, control_size), Q(state_size, state_size), LQ(state_size) { }

//    void LGTransition::init(const Matrix &A, const Matrix &Cov, const Matrix &B) {
//        setInputM(A);
//        setControlM(B);
//        Q = Cov;
//        if (!NumericalRcond::isSymmetric(Q))
//            throw LogicException("Transition model. Covariance matrix not symmetric");
//        LQ.solve(Q);
//        rclimit.checkPD(LQ.rcond(), "Transition model. Covariance matrix no PD");
//    }


//    LGObservation::LGObservation(std::size_t state_size, std::size_t obs_size) :
//            LinearModel(obs_size, state_size), R(obs_size, obs_size), LR(obs_size) {}

//    LGObservation::LGObservation(std::size_t state_size, std::size_t obs_size, std::size_t control_size) :
//            LinearModel(obs_size, state_size, control_size), R(obs_size, obs_size), LR(obs_size) {}

//    void LGObservation::init(const Matrix &C, const Matrix &Cov, const Matrix &D) {
//        setInputM(C);
//        setControlM(D);
//        R = Cov;
//        if (!NumericalRcond::isSymmetric(R))
//            throw LogicException("Observation model. Covariance matrix not symmetric");
//        LR.solve(R);
//        rclimit.checkPD(LR.rcond(), "Observation model. Covariance matrix no PD");
//    }

}

