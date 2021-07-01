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
//FIXME: return all models attributes via getters
    FilterBase::FilterBase(std::size_t x_size) :
            last_y_size(0) {
        if (x_size < 1)
            throw LogicException("zero size state");
        X = Matrix(x_size, x_size);
        x = Vector(x_size);
        temp_X = Matrix(x_size, x_size);
    }

    void FilterBase::init(const Vector &x_init, const Matrix &X_init) {
        x = x_init;
        X = X_init;
        if (!NumericalRcond::isPSD(X))
            throw NumericException("Initialised state covariance not PSD");
    }

    void FilterBase::predict(LGtransition &lgsm) {
        x = lgsm.g(x);

        Check_Result(X, "In filter predict step. Old covariance: ");
        temp_X.noalias() = (lgsm.A * X).transpose();
        X.noalias() = lgsm.A * temp_X + lgsm.getQ();
        Check_Result(X, "New covariance:");
    }


    CovarianceScheme::CovarianceScheme(std::size_t x_size, std::size_t y_initsize) :
            FilterBase(x_size) {
        observe_size(y_initsize);
    }

    void CovarianceScheme::observe(LGobserve &lgobsm, const Ref<Vector> &y) {
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
        Check_Result(x, "In filter update step. New state: ")
        Check_Result(X, "In filter update step. New state covariance: ");
    }

    void CovarianceScheme::observe_size(std::size_t y_size) {
        if (y_size != last_y_size) {
            last_y_size = y_size;
            S.resize(y_size, y_size);
            K.resize(x.size(), y_size);
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

    void InformationScheme::observe(LGobserve &lgobsm, const Ref<Vector> &y) {
        update(lgobsm, y);
    }

    void InformationScheme::update(LGobserve &lgobsm, const Vector &r) {
        if (r.size() != lgobsm.R.rows())
            throw LogicException("Observation and model sizes inconsistent");
        observe_size(r.size()); // Dynamic sizing

        // Observation information
        rclimit.checkPD(lgobsm.R.llt().rcond(), "In filter update step. Observation noise covariance not PD.");
        R_inv = lgobsm.R.inverse(); // FIXME: return R via getter
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


    SmootherBase::SmootherBase(std::size_t x_size) :
            X(x_size, x_size), x(x_size) {}

    void SmootherBase::init(const Vector &x_final, const Matrix &X_final) {
        x = x_final;
        X = X_final;
        if (!NumericalRcond::isPSD(X))
            throw NumericException("Initialised state covariance not PSD");
    }


    RtsScheme::RtsScheme(size_t x_size) : SmootherBase(x_size), J(x_size, x_size) {}

    void
    RtsScheme::updateBack(const LGtransition &lgsm,
                          const Vector &filtered_xprior, const Vector &filtered_xpost,
                          const Matrix &filtered_Xprior, const Matrix &filtered_Xpost) {
        // Assume the filtered covariances already checked for PD in the filter
        Check_Result(J, "In update back step. The current backwards Kalman gain: ");
        J.noalias() = filtered_Xpost * lgsm.A.transpose() * filtered_Xprior.inverse();
        Check_Result(J, "In update back step. The new backwards Kalman gain: ");

        Check_Result(X, "In update back step. The current state covariance:");
        X = filtered_Xpost + J * (X - filtered_Xprior) * J.transpose();
        Check_Result(X, "In update back step. The new state covariance:");

        Check_Result(x, "In update back step. The current state:");
        x = filtered_xpost + J * (x - filtered_xprior);
        Check_Result(x, "In update back step. The new state:");
    }


    TwoWayScheme::TwoWayScheme(size_t x_size) :
            SmootherBase(x_size),
            Tau(x_size, x_size), theta(x_size),
            temp_D(x_size, x_size), I(x_size, x_size) {}

    void TwoWayScheme::initInformation(LGobserve &lgobsm, const Ref<Vector> &y_final,
                                       const Vector &x_final, const Matrix &X_final) {
        Matrix temp(lgobsm.C * lgobsm.R.inverse());
        Tau.noalias() = temp * lgobsm.C;
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

    void TwoWayScheme::predictBack(LGtransition &lgsm) {
        Eigen::LLT<Matrix> Qllt(lgsm.Q);
        rclimit.checkPD(Qllt.rcond(), "In filter update back step. State noise covariance not PD.");
        Matrix Theta(Qllt.matrixL());
        temp_D.noalias() = Theta * (I + Theta.transpose() * Tau * Theta).inverse() * Theta.transpose();
        theta = lgsm.B.transpose() * (I - Tau * temp_D) * theta;
        Tau = lgsm.B.transpose() * Tau * (I - temp_D * Tau) * lgsm.B;
    }

    void TwoWayScheme::updateBack(LGobserve &lgobsm, const Ref<Vector> &y,
                                  const Vector &filtered_xprior, const Matrix &filtered_Xprior) {
        observe_size(y.size());
        rclimit.checkPD(lgobsm.R.llt().rcond(), "In filter update back step. Observation noise covariance not PD.");
        temp_Y.noalias() = lgobsm.C.transpose() * lgobsm.R.inverse();
        Tau.noalias() += temp_Y * lgobsm.C;
        theta.noalias() += temp_Y * y;
        temp_D = filtered_Xprior.inverse();
        X = (temp_D + Tau).inverse();
        x.noalias() = X * (temp_D * filtered_xprior + theta);
    }


}