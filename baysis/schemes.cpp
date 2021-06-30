//
// schemes.cpp
// Baysis
//
// Created by Vladimir Sotskov on 29/06/2021, 16:03.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include "schemes.hpp"
#include "baysisexception.hpp"


namespace schemes {

    FilterBase::FilterBase(std::size_t x_size) :
            last_y_size(0) {
        if (x_size < 1)
            throw LogicException("zero size state");
        X = Matrix(x_size, x_size);
        x = Vector(x_size);
        tempX = Matrix(x_size, x_size);
    }

    void FilterBase::init(const Vector &x_init, const Matrix &X_init) {
        x = x_init;
        X = X_init;
        if (!NumericalRcond::isPSD(X))
            throw NumericException("Initial state covariance not PSD");
    }

    void FilterBase::predict(LGtransition &lgsm) {
        x = lgsm.g(x);

        Check_Result(X, "In filter predict step. Old covariance: ");
        tempX.noalias() = (lgsm.A * X).transpose();
        X.noalias() = lgsm.A * tempX + lgsm.getQ();
        Check_Result(X, "New covariance:");
    }


    CovarianceScheme::CovarianceScheme(std::size_t x_size, std::size_t y_initsize) :
            FilterBase(x_size) {
        observe_size(y_initsize);
    }

    void CovarianceScheme::observe(LGobserve &lgobsm, const Vector &y) {
        Vector r = y - lgobsm.h(x);        // Observation model, innovation;
        return update(lgobsm, r);
    }

    void CovarianceScheme::update(LGobserve &lgobsm, const Vector &r) {
        if (r.size() != lgobsm.R.rows())
            throw LogicException("Observation and model sizes inconsistent");
        observe_size(r.size()); // Dynamic sizing

        // Innovation covariance
        Matrix tempXY(X * lgobsm.C.transpose());
        S.noalias() = lgobsm.C * tempXY + lgobsm.R;
        Check_Result(S, "In filter update step. Innovation covariance: ");
        rclimit.checkPD(S.llt().rcond(), "In filter update step. Innovation covariance not PD.");

        // Kalman gain
        K.noalias() = tempXY * S.inverse();
        Check_Result(K, "In filter update step. Kalman gain: ")

        // State update
        x.noalias() += K * r;
        X -= K * lgobsm.C * X;
        Check_Result(X, "In filter update step. New state covariance: ");
    }

    void CovarianceScheme::observe_size(std::size_t y_size) {
        if (y_size != last_y_size) {
            last_y_size = y_size;
            S.resize(y_size, y_size);
            K.resize(Eigen::NoChange_t, y_size);
        }
    }


    InformationScheme::InformationScheme(std::size_t x_size, std::size_t z_initsize) :
            FilterBase(x_size), e(x_size), La(x_size, x_size) {
        observe_size(z_initsize);
        transform_required = true;
    }

    void InformationScheme::init(const Vector &x_init, const Matrix &X_init) {
        FilterBase::init(x_init, X_init);
        La = X.inverse();
        e.noalias() = La * x;
        transform_required = false;
    }

    void InformationScheme::initInformation(const Vector &eta, const Matrix &Lambda) {
        e = eta;
        La = Lambda;
        if (!NumericalRcond::isPSD(Y))
            throw NumericException("Initial information matrix not PSD"));
        transform_required = true;
    }

    void InformationScheme::predict(LGtransition &lgsm) {
        transform();
        FilterBase::predict(lgsm);
        // Information
        rclimit.checkPD(X.llt().rcond(), "In filter predict step. Covariance matrix is not PD")
        La = X.inverse();
        e.noalias() = La * x;
    }

    void InformationScheme::observe(LGobserve &lgobsm, const Vector &y) {
        update(lgobsm, y);
    }

    void InformationScheme::update(LGobserve &lgobsm, const Vector &r) {
        if (r.size() != lgobsm.R.rows())
            throw LogicException("Observation and model sizes inconsistent");
        observe_size(r.size()); // Dynamic sizing

        // Observation information
        rclimit.checkPD(lgobsm.R.llt().rcond(), "In filter update step. Observation noise covariance not PD.");
        R_inv = lgobsm.R.inverse();
        Matrix CtRi(lgobsm.C.transpose() * R_inv);
        e.noalias() += CtRi * r;
        La.noalias() += CtRi * lgobsm.C;
        transform_required = true;
    }

    void InformationScheme::transform() {
        if (transform_required) {
            rclimit.checkPD(La.llt().rcond(), "In filter transform. Information matrix is not PD")
            X = La.inverse();
            x.noalias() = X * e;
            transform_required = false;
        }
    }

    void InformationScheme::observe_size(std::size_t y_size) {
        if (y_size != last_y_size) {
            last_y_size = y_size;
            R_inv.resize(y_size, y_size);
        }
    }

}