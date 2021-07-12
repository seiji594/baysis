//
// models.hpp
// Baysis
//
// Created by Vladimir Sotskov on 11/06/2021.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#ifndef BAYSIS_MODELS_HPP
#define BAYSIS_MODELS_HPP

#include "baysisexception.hpp"
#include "matsupport.hpp"
#include "probsupport.hpp"

using Eigen::Ref;


namespace ssmodels
{
    class SSModelBase {
    public:
        SSModelBase(std::size_t state_dim, std::size_t seq_length);

        std::size_t length() const { return T; }
        std::size_t stateDim() const { return xdim; }
    protected:
        const std::size_t xdim;
        const std::size_t T;
        NumericalRcond rclimit;
    };


    class TransitionModel: virtual public SSModelBase {
    public:
        TransitionModel(size_t state_dim, size_t seq_length) : SSModelBase(state_dim, seq_length) {}
    };


    class ObservationModel: virtual public SSModelBase {
    public:
        ObservationModel(std::size_t obs_dim, std::size_t state_dim, std::size_t seq_length)
        : SSModelBase(state_dim, seq_length), ydim(obs_dim) {}

        std::size_t obsDim() const { return ydim; }
    protected:
        const std::size_t ydim;
    };

    /**
     * Class representing a linear part of a model
     */
    class LinearModel {
    public:
        LinearModel(std::size_t input_rows, std::size_t input_cols, std::size_t control_size=0);

        template<class Derived>
        void apply(const Eigen::MatrixBase<Derived>& vec) const;

        template<class Derived>
        void setInputM(const Eigen::MatrixBase<Derived>& input_m);

        template<class Derived>
        void setControlM(const Eigen::MatrixBase<Derived>& control_m);

        template<class Derived>
        void setControls(const Eigen::MatrixBase<Derived>& cur_ctrl);

    protected:
        Matrix inputM;
        Matrix controlM;
        Vector controls;
        mutable Vector res;     // result of linear transformation by input and control matrices
    };

    /**
     * Linear Gaussian stationary transition model. The parameters not changing with time.
     */
    class LGTransitionStationary: public TransitionModel, public LinearModel {
    public:
        LGTransitionStationary(std::size_t seq_length, std::size_t state_size, std::size_t control_size = 0);

        void init(const Matrix& A, const Matrix & Cov, const Matrix& B=Matrix());
        void setPrior(const Vector& mu, const Matrix& sigma);

        template<typename DerivedA, typename DerivedB>
        double logDensity(const Eigen::MatrixBase<DerivedA>& curx, const Eigen::MatrixBase<DerivedB>& prevx) const;

        template<class Derived>
        Vector& getMean(const Eigen::MatrixBase<Derived>& x) const {
            apply(x);
            return res;
        }
        const Vector& getPriorMean() const {
            return mu_prior;
        }
        const Matrix& getA() const {
            return inputM;
        }
        const Matrix& getCov() const {
            return Q;
        }
        const Matrix& getCovInv() const {
            return Q_inv;
        }
        const Matrix& getPriorCov() const {
            return Q_prior;
        }
        const Matrix& getPriorCovInv() const {
            return Q_prior_inv;
        }
        const Eigen::LLT<Matrix> & getL() const {
            return LQ;
        }
        const Eigen::LLT<Matrix> & getLprior() const {
            return LQprior;
        }

    private:
        Matrix Q;       // State Gaussian noise covariance
        Matrix Q_inv;
        Matrix Q_prior;
        Matrix Q_prior_inv;
        Vector mu_prior;
        Eigen::LLT<Matrix> LQ;  // Cholesky decomposition
        Eigen::LLT<Matrix> LQprior;
    };


    /**
     * Linear Gaussian stationary observation model. The parameters not changing with time.
     */
    class LGObservationStationary: public ObservationModel, public LinearModel {
    public:
        LGObservationStationary(std::size_t seq_length, std::size_t state_size, std::size_t obs_size,
                                std::size_t control_size=0);

        void init(const Matrix& C, const Matrix& Cov, const Matrix& D=Matrix());

        template<typename DerivedA, typename DerivedB>
        double logDensity(const Eigen::MatrixBase<DerivedA>& y, const Eigen::MatrixBase<DerivedB>& x) const;

        template<class Derived>
        Vector& getMean(const Eigen::MatrixBase<Derived> &y) const {
            apply(y);
            return res;
        }
        const Matrix& getCov() const {
            return R;
        }
        const Matrix& getCovInv() const {
            return R_inv;
        }
        const Matrix& getC() const {
            return inputM;
        }
        const Eigen::LLT<Matrix>& getL() {
            return LR;
        }

    private:
        Matrix R;        // Observation Gaussian noise covariance
        Matrix R_inv;
        Eigen::LLT<Matrix> LR;  // Cholesky decomposition
    };

    template <typename Derived>
    void LinearModel::apply(const Eigen::MatrixBase<Derived>& vec) const {
        res = inputM * vec;
        if (controlM.size() != 0)
            res.noalias() += controlM * controls;
    }

    template <typename Derived>
    void LinearModel::setInputM(const Eigen::MatrixBase<Derived>& input_m) {
        if (input_m.size() != inputM.size())
            throw LogicException("Input matrix is the wrong size");
        inputM = input_m;
    }

    template <typename Derived>
    void LinearModel::setControlM(const Eigen::MatrixBase<Derived>& control_m) {
        if (control_m.size() != controlM.size())
            throw LogicException("Control matrix is the wrong size");
        controlM = control_m;
    }

    template <typename Derived>
    void LinearModel::setControls(const Eigen::MatrixBase<Derived>& cur_ctrl) {
        if (cur_ctrl.size() != controls.size())
            throw LogicException("Controls vector is the wrong size");
        controls = cur_ctrl;
    }


    template<typename DerivedA, typename DerivedB>
    double LGTransitionStationary::logDensity(const Eigen::MatrixBase<DerivedA> &curx,
                                              const Eigen::MatrixBase<DerivedB> &prevx) const {
        return NormalDist::logDensity(curx, getMean(prevx), LQ);
    }


    template<typename DerivedA, typename DerivedB>
    double LGObservationStationary::logDensity(const Eigen::MatrixBase<DerivedA> &y,
                                               const Eigen::MatrixBase<DerivedB> &x) const {
        return NormalDist::logDensity(y, getMean(x), LR);
    }

}

#endif //BAYSIS_MODELS_HPP
