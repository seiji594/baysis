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
    }


    FilterBase::FilterBase(std::size_t x_size) :
            LinearStateBase(x_size), temp_X(x_size, x_size) { }

    void FilterBase::predict(LGTransitionStationary &lgsm) {
        x = lgsm.getMean(x);

        temp_X.noalias() = (lgsm.getA() * X).transpose();
        X = lgsm.getA() * temp_X + lgsm.getCov();
//        Check_Result(X, "In filter predict step. New covariance:");
    }


    CovarianceScheme::CovarianceScheme(std::size_t x_size, std::size_t y_size) :
            LinearStateBase(x_size), FilterBase(x_size),
            S(y_size, y_size), K(x.size(), y_size), tempXY(x_size, y_size) { }

    void CovarianceScheme::observe(LGObservationStationary &lgobsm, const Ref<Vector> &y) {
        Vector r = y - lgobsm.getMean(x);        // Observation model, innovation;
        return update(lgobsm, r);
    }

    void CovarianceScheme::update(LGObservationStationary &lgobsm, const Ref<Vector> &r) {
        // Innovation covariance
        tempXY.noalias() = X * lgobsm.getC().transpose();
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


    InformationScheme::InformationScheme(std::size_t x_size, std::size_t y_size) :
            LinearStateBase(x_size), FilterBase(x_size),
            e(x_size), La(x_size, x_size), tempCR(x_size, y_size) { }

    void InformationScheme::init(const Vector &x_init, const Matrix &X_init) {
        LinearStateBase::init(x_init, X_init);
        La = X.inverse();
        e.noalias() = La * x;
    }

    void InformationScheme::initInformation(const Ref<Vector> &eta, const Ref<Matrix> &Lambda) {
        e = eta;
        La = Lambda;
        if (!NumericalRcond::isPSD(La))
            throw NumericException("Initial information matrix not PSD");
        transform();
    }

    void InformationScheme::predict(LGTransitionStationary &lgsm) {
        FilterBase::predict(lgsm);
        // Information
        rclimit.checkPD(X.llt().rcond(), "In filter predict step. Covariance matrix is not PD");
        La = X.inverse();
        e.noalias() = La * x;
    }

    void InformationScheme::observe(LGObservationStationary &lgobsm, const Ref<Vector> &y) {
        update(lgobsm, y);
    }

    void InformationScheme::update(LGObservationStationary &lgobsm, const Ref<Vector> &r) {
        // Observation information
        tempCR.noalias() = lgobsm.getC().transpose() * lgobsm.getCovInv();
        e.noalias() += tempCR * r;
        La.noalias() += tempCR * lgobsm.getC();
        transform();
    }

    void InformationScheme::transform() {
        rclimit.checkPD(La.llt().rcond(),
                        "In filter transform. Information matrix is not PD");
        X = La.inverse();
        x.noalias() = X * e;
    }


    SmootherBase::SmootherBase(std::size_t x_size) :
            LinearStateBase(x_size) {}


    RtsScheme::RtsScheme(size_t x_size) :
            LinearStateBase(x_size), SmootherBase(x_size), J(x_size, x_size) {}

    void RtsScheme::updateBack(const LGTransitionStationary &lgsm,
                               const Ref<Vector> &filtered_xprior, const Ref<Vector> &filtered_xpost,
                               const Ref<Matrix> &filtered_Xprior, const Ref<Matrix> &filtered_Xpost) {
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
            Tau(x_size, x_size), theta(x_size),
            temp_D(x_size, x_size), I(Matrix::Identity(x_size, x_size)) {}

    void TwoWayScheme::initInformation(LGObservationStationary &lgobsm, const Ref<Vector> &y_final,
                                       const Ref<Vector> &x_final, const Ref<Matrix> &X_final) {
        Matrix temp(lgobsm.getC().transpose() * lgobsm.getCovInv());
        Tau.noalias() = temp * lgobsm.getC();
        theta.noalias() = temp * y_final;
        x = x_final;
        X = X_final;
        temp_Y = Matrix(lgobsm.stateDim(), lgobsm.obsDim());
    }

    void TwoWayScheme::predictBack(LGTransitionStationary &lgsm) {
        Matrix Theta(lgsm.getL().matrixL());
        temp_D = Theta * (I + Theta.transpose() * Tau * Theta).inverse() * Theta.transpose();
        theta = lgsm.getA().transpose() * (I - Tau * temp_D) * theta;
        Tau = lgsm.getA().transpose() * Tau * (I - temp_D * Tau) * lgsm.getA();
    }

    void TwoWayScheme::updateBack(LGObservationStationary &lgobsm, const Ref<Vector> &y,
                                  const Ref<Vector> &filtered_xprior, const Ref<Matrix> &filtered_Xprior) {
        temp_Y.noalias() = lgobsm.getC().transpose() * lgobsm.getCovInv();
        Tau.noalias() += temp_Y * lgobsm.getC();
        theta.noalias() += temp_Y * y;
        temp_D = filtered_Xprior.inverse();
        X = (temp_D + Tau).inverse();
        x = X * (temp_D * filtered_xprior + theta);
    }
}