//
// schemes.cpp
// Baysis
//
// Created by Vladimir Sotskov on 29/06/2021, 16:03.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include "filterschemes.hpp"
#include "baysisexception.hpp"


namespace schemes {

    LinearStateBase::LinearStateBase(std::size_t x_size) {
        if (x_size < 1)
            throw LogicException("zero size state");
        X = Matrix(x_size, x_size);
        x = Vector(x_size);
    }

    void LinearStateBase::init(const Vector &x_init, const Matrix &X_init) {
        x = x_init;
        X = X_init;
        if (!NumericalRcond::isPSD(X))
            throw NumericException("Initialised state covariance not PSD");
    }


    FilterBase::FilterBase(std::size_t x_size) :
            LinearStateBase(x_size), last_y_size(0) {
        temp_X = Matrix(x_size, x_size);
    }

    void FilterBase::predict(LGTransition &lgsm) {
        x = lgsm.stateMean(x);

        temp_X.noalias() = (lgsm.getA() * X).transpose();
        X = lgsm.getA() * temp_X + lgsm.getCov();
//        Check_Result(X, "In filter predict step. New covariance:");
    }


    CovarianceScheme::CovarianceScheme(std::size_t x_size, std::size_t y_initsize) :
            LinearStateBase(x_size), FilterBase(x_size) {
        observe_size(y_initsize);
    }

    void CovarianceScheme::observe(LGObservation &lgobsm, const Ref<Vector> &y) {
        Vector r = y - lgobsm.obsMean(x);        // Observation model, innovation;
        return update(lgobsm, r);
    }

    void CovarianceScheme::update(LGObservation &lgobsm, const Vector &r) {
        if (r.size() != lgobsm.getCov().rows())
            throw LogicException("Observation and model sizes inconsistent");
        observe_size(r.size()); // Dynamic sizing

        // Innovation covariance
        Matrix tempXY(X * lgobsm.getC().transpose());
        S = lgobsm.getC() * tempXY + lgobsm.getCov();
//        Check_Result(S, "In filter update step. Innovation covariance: ");
        rclimit.checkPD(S.llt().rcond(), "In filter update step. Innovation covariance not PD.");

        // Kalman gain
        K.noalias() = tempXY * S.inverse();
//        Check_Result(K, "In filter update step. Kalman gain: ");

        // State update
        x.noalias() += K * r;
        X -= K * lgobsm.getC() * X;
//        Check_Result(x, "In filter update step. New state: ");
//        Check_Result(X, "In filter update step. New state covariance: ");
    }

    void CovarianceScheme::observe_size(std::size_t y_size) {
        if (y_size != last_y_size) {
            last_y_size = y_size;
            S.resize(y_size, y_size);
            K.resize(x.size(), y_size);
        }
    }


    InformationScheme::InformationScheme(std::size_t x_size, std::size_t z_initsize) :
            LinearStateBase(x_size), FilterBase(x_size),
            e(x_size), La(x_size, x_size) {
        observe_size(z_initsize);
    }

    void InformationScheme::init(const Vector &x_init, const Matrix &X_init) {
        LinearStateBase::init(x_init, X_init);
        La = X.inverse();
        e.noalias() = La * x;
    }

    void InformationScheme::initInformation(const Vector &eta, const Matrix &Lambda) {
        e = eta;
        La = Lambda;
        if (!NumericalRcond::isPSD(La))
            throw NumericException("Initial information matrix not PSD");
        transform();
    }

    void InformationScheme::predict(LGTransition &lgsm) {
        FilterBase::predict(lgsm);
        // Information
        rclimit.checkPD(X.llt().rcond(), "In filter predict step. Covariance matrix is not PD");
        La = X.inverse();
        e.noalias() = La * x;
    }

    void InformationScheme::observe(LGObservation &lgobsm, const Ref<Vector> &y) {
        update(lgobsm, y);
    }

    void InformationScheme::update(LGObservation &lgobsm, const Vector &r) {
        if (r.size() != lgobsm.getCov().rows())
            throw LogicException("Observation and model sizes inconsistent");
        observe_size(r.size()); // Dynamic sizing

        // Observation information
        R_inv = lgobsm.getCov().inverse();
        Matrix CtRi(lgobsm.getC().transpose() * R_inv);
        e.noalias() += CtRi * r;
        La.noalias() += CtRi * lgobsm.getC();
        transform();
    }

    void InformationScheme::transform() {
        rclimit.checkPD(La.llt().rcond(),
                        "In filter transform. Information matrix is not PD");
        X = La.inverse();
        x.noalias() = X * e;
    }

    void InformationScheme::observe_size(std::size_t y_size) {
        if (y_size != last_y_size) {
            last_y_size = y_size;
            R_inv.resize(y_size, y_size);
        }
    }


    SmootherBase::SmootherBase(std::size_t x_size) :
            LinearStateBase(x_size) {}


    RtsScheme::RtsScheme(size_t x_size) :
            LinearStateBase(x_size), SmootherBase(x_size), J(x_size, x_size) {}

    void RtsScheme::updateBack(const LGTransition &lgsm,
                               const Vector &filtered_xprior, const Vector &filtered_xpost,
                               const Matrix &filtered_Xprior, const Matrix &filtered_Xpost) {
        // Assume the filtered covariances already checked for PD in the filter
//        Check_Result(J, "In update back step. The current backwards Kalman gain: ");
        J.noalias() = filtered_Xpost * lgsm.getA().transpose() * filtered_Xprior.inverse();
//        Check_Result(J, "In update back step. The new backwards Kalman gain: ");

//        Check_Result(X, "In update back step. The current state covariance:");
        X = filtered_Xpost + J * (X - filtered_Xprior) * J.transpose();
//        Check_Result(X, "In update back step. The new state covariance:");

//        Check_Result(x, "In update back step. The current state:");
        x = filtered_xpost + J * (x - filtered_xprior);
//        Check_Result(x, "In update back step. The new state:");
    }


    TwoWayScheme::TwoWayScheme(size_t x_size) :
            LinearStateBase(x_size), SmootherBase(x_size),
            last_y_size(0),
            Tau(x_size, x_size), theta(x_size),
            temp_D(x_size, x_size), I(Matrix::Identity(x_size, x_size)) {}

    void TwoWayScheme::initInformation(LGObservation &lgobsm, const Ref<Vector> &y_final,
                                       const Vector &x_final, const Matrix &X_final) {
        Matrix temp(lgobsm.getC().transpose() * lgobsm.getCov().inverse());
        Tau.noalias() = temp * lgobsm.getC();
        theta.noalias() = temp * y_final;
        observe_size(y_final.size());
        init(x_final, X_final);
    }

    void TwoWayScheme::observe_size(std::size_t y_size) {
        if (y_size != last_y_size) {
            last_y_size = y_size;
            temp_Y.resize(x.size(), y_size);
        }
    }

    void TwoWayScheme::predictBack(LGTransition &lgsm) {
        Matrix Theta(lgsm.getL().matrixL());
        temp_D = Theta * (I + Theta.transpose() * Tau * Theta).inverse() * Theta.transpose();
        theta = lgsm.getA().transpose() * (I - Tau * temp_D) * theta;
        Tau = lgsm.getA().transpose() * Tau * (I - temp_D * Tau) * lgsm.getA();
    }

    void TwoWayScheme::updateBack(LGObservation &lgobsm, const Ref<Vector> &y,
                                  const Vector &filtered_xprior, const Matrix &filtered_Xprior) {
        observe_size(y.size());
        temp_Y.noalias() = lgobsm.getC().transpose() * lgobsm.getCov().inverse();
        Tau.noalias() += temp_Y * lgobsm.getC();
        theta.noalias() += temp_Y * y;
        temp_D = filtered_Xprior.inverse();
        X = (temp_D + Tau).inverse();
        x = X * (temp_D * filtered_xprior + theta);
    }
}