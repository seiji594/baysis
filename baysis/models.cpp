//
// models.cpp
// Baysis
//
// Created by Vladimir Sotskov on 11/06/2021.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include "models.hpp"


namespace ssmodels {

    SSModelBase::SSModelBase(std::size_t state_dim, std::size_t seq_length) : xdim(state_dim), T(seq_length) {
        if (state_dim < 1 || seq_length < 1)
            throw LogicException("State dimension and sequence length should be at least 1.");
    }


    LinearModel::LinearModel(std::size_t input_rows,
                             std::size_t input_cols,
                             std::size_t control_size)
            : inputM(input_rows, input_cols) {
        if (control_size != 0) {
            controlM = Matrix(input_rows, control_size);
            controls = Vector(control_size);
        }
    }


    LGTransitionStationary::LGTransitionStationary(std::size_t seq_length, std::size_t state_size,
                                                   std::size_t control_size)
            : TransitionModel(state_size, seq_length),
              LinearModel(state_size, state_size, control_size), Q(state_size, state_size),
              Q_inv(state_size, state_size), Q_prior(state_size, state_size), Q_prior_inv(state_size, state_size),
              mu_prior(state_size), LQ(state_size), LQprior(state_size) { }

    void LGTransitionStationary::init(const Matrix &A, const Matrix &Cov, const Matrix &B) {
        setInputM(A);
        setControlM(B);
        Q = Cov;
        if (!NumericalRcond::isSymmetric(Q)) {
            throw LogicException("Transition model. Covariance matrix not symmetric");
        }
        LQ.compute(Q);
        rclimit.checkPD(LQ.rcond(), "Transition model. Covariance matrix not PD");
        Q_inv = Q.inverse();
    }

    void LGTransitionStationary::setPrior(const Vector &mu, const Matrix &sigma) {
        Q_prior = sigma;
        if (!NumericalRcond::isSymmetric(Q_prior)) {
            throw LogicException("Transition model. Initial covariance matrix not symmetric");
        }
        LQprior.compute(Q_prior);
        rclimit.checkPD(LQprior.rcond(), "Transition model. Covariance matrix not PD");
        Q_prior_inv = Q_prior.inverse();
        mu_prior = mu;
    }


    LGObservationStationary::LGObservationStationary(std::size_t seq_length,
                                                     std::size_t state_size,
                                                     std::size_t obs_size,
                                                     std::size_t control_size)
            : ObservationModel(obs_size, state_size, seq_length),
              LinearModel(obs_size, state_size, control_size),
              R(obs_size, obs_size), R_inv(obs_size, obs_size), LR(obs_size) {}

    void LGObservationStationary::init(const Matrix &C, const Matrix &Cov, const Matrix &D) {
        setInputM(C);
        setControlM(D);
        R = Cov;
        if (!NumericalRcond::isSymmetric(R))
            throw LogicException("Observation model. Covariance matrix not symmetric");
        LR.compute(R);
        rclimit.checkPD(LR.rcond(), "Observation model. Covariance matrix not PD");
        R_inv = R.inverse();
    }


    LPObservationStationary::LPObservationStationary(std::size_t seq_length,
                                                     std::size_t state_size,
                                                     std::size_t obs_size,
                                                     std::size_t control_size)
            : ObservationModel(obs_size, state_size, seq_length),
              LinearModel(obs_size, state_size, control_size) { }

    void LPObservationStationary::init(const Matrix &C, const Matrix &D, const Vector &ctrls) {
        setInputM(C);
        setControlM(D);
        setControls(ctrls);
    }


    GPObservationStationary::GPObservationStationary(std::size_t seq_length, std::size_t m_size, MF mf)
            : ObservationModel(m_size, m_size, seq_length),
              mean_function(mf), mean(m_size), coefficients(m_size) { }

    void GPObservationStationary::init(const Vector &mc) {
        coefficients = mc;
    }

}