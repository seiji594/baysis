//
// models.hpp
// Baysis
//
// Created by Vladimir Sotskov on 11/06/2021.
//  The code inspired in large part by Bayes++ the Bayesian Filtering Library.
//  Copyright (c) 2003,2004,2005,2006,2011,2012,2014 Michael Stevens
//  Copyright (c) 2002 Michael Stevens and Australian Centre for Field Robotics
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#ifndef BAYSIS_MODELS_HPP
#define BAYSIS_MODELS_HPP

#include "baysisexception.hpp"
#include "matsupport.hpp"
#include "probsupport.hpp"

using Eigen::Ref;


namespace ssm
{

    /**
     * Abstract transition models
     * Used to parameterise transition function and noise
     * @tparam Dist - internal distribution class with static members to compute densities and draw samples
     */
     template<typename Dist>
    class TransitionModel {
    public:
        typedef typename Dist::Sample_type State;
    };

    /**
     * Abstract Observation models
     *  Observe models are used to parameterise the observe functions
     *  Size of the observation vector potentially not fixed
     *  @tparam Dist - internal distribution class with static members to compute densities and draw samples
     */
     template<typename Dist>
    class ObservationModel {
    public:
        typedef typename Dist::Sample_type Observation;
    };

    /**
     * Class representing a linear part of a model
     */
    class LinearModel {
    public:
        LinearModel(std::size_t input_rows, std::size_t input_cols) :
                inputM(input_rows, input_cols), controlM(), controls() { }
        LinearModel(std::size_t input_rows, std::size_t input_cols, std::size_t control_size) :
                inputM(input_rows, input_cols), controlM(input_rows, control_size), controls(control_size) { }

        template<typename Derived>
        void apply(const Eigen::MatrixBase<Derived>& vec) const {
            res = inputM * vec;
            if (controlM.size() != 0)
                res.noalias() += controlM * controls;
        }

        void setInputM(const Matrix &input_m) {
            inputM = input_m;
        }

        void setControlM(const Matrix &control_m) {
            controlM = control_m;
        }

        void setControls(const Vector &cur_ctrl) {
            controls = cur_ctrl;
        }

    protected:
        NumericalRcond rclimit;
        Matrix inputM;
        Matrix controlM;
        Vector controls;
        mutable Vector res;     // result of linear transformation by input and control matrices
    };

    /**
     * Linear Gaussian transition model
     */
    class LGTransition: public TransitionModel<NormalDist>, public LinearModel {
    public:
        typedef typename TransitionModel<NormalDist>::State State;

        explicit LGTransition(std::size_t state_size) :
                LinearModel(state_size, state_size), Q(state_size, state_size), LQ(state_size) { }
        LGTransition(std::size_t state_size, std::size_t control_size) :
                LinearModel(state_size, state_size, control_size), Q(state_size, state_size), LQ(state_size) { }

        void init(const Matrix& A, const Matrix & Cov, const Matrix& B=Matrix()) {
            setInputM(A);
            setControlM(B);
            Q = Cov;
            if (!NumericalRcond::isSymmetric(Q))
                throw LogicException("Transition model. Covariance matrix not symmetric");
            LQ.compute(Q);
            rclimit.checkPD(LQ.rcond(), "Transition model. Covariance matrix no PD");
        }

        template<class Derived>
        State& stateMean(const Eigen::MatrixBase<Derived>& x) const {
            apply(x);
            return res;
        }
        const Matrix& getCov() const {
            return Q;
        }
        const Matrix& getA() const {
            return inputM;
        }
        const Eigen::LLT<Matrix>& getL() {
            return LQ;
        }

    protected:
        Matrix Q;       // State Gaussian noise covariance
    private:
        Eigen::LLT<Matrix> LQ;  // Cholesky decomposition
    };


    class LGObservation: public ObservationModel<NormalDist>, public LinearModel {
    public:
        typedef typename ObservationModel<NormalDist>::Observation Observation;

        LGObservation(std::size_t state_size, std::size_t obs_size) :
                LinearModel(obs_size, state_size), R(obs_size, obs_size), LR(obs_size) {}
        LGObservation(std::size_t state_size, std::size_t obs_size, std::size_t control_size) :
                LinearModel(obs_size, state_size, control_size), R(obs_size, obs_size), LR(obs_size) {}

        void init(const Matrix& C, const Matrix& Cov, const Matrix& D=Matrix()) {
            setInputM(C);
            setControlM(D);
            R = Cov;
            if (!NumericalRcond::isSymmetric(R))
                throw LogicException("Observation model. Covariance matrix not symmetric");
            LR.compute(R);
            rclimit.checkPD(LR.rcond(), "Observation model. Covariance matrix no PD");
        }

        template<class Derived>
        Observation& obsMean(Eigen::MatrixBase<Derived>& y) const {
            apply(y);
            return res;
        }
        const Matrix& getCov() const {
            return R;
        }
        const Matrix& getC() const {
            return inputM;
        }
        const Eigen::LLT<Matrix>& getL() {
            return LR;
        }
    protected:
        Matrix R;        // Observation Gaussian noise covariance
    private:
        Eigen::LLT<Matrix> LR;  // Cholesky decomposition
    };


}

#endif //BAYSIS_MODELS_HPP
