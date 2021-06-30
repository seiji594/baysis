//
// mat_support.hpp
// Baysis
//
// Created by Vladimir Sotskov on 24/06/2021, 16:18.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#ifndef BAYSIS_MATSUPPORT_HPP
#define BAYSIS_MATSUPPORT_HPP

#include <Eigen/Dense>


typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXi Vector_int;

/** Minimum allowable reciprocal condition number for PD Matrix factorisations
 * Initialised default gives 5 decimal digits of headroom */
constexpr double LIMIT_PD_INIT = std::numeric_limits<double>::epsilon() * double(1e5);


/**
 * Numerical comparison of reciprocal condition numbers
 *  Required for all linear algebra in models, filters and samplers
 *  Implements minimum allowable reciprocal condition number for PD Matrix factorisations
 * From Bayes++ the Bayesian Filtering Library
 * Copyright (c) 2002 Michael Stevens
 */
class NumericalRcond
{
public:
    NumericalRcond() { limit_pd = LIMIT_PD_INIT; }
    void setLimitPD(double nl) { limit_pd = nl; }
    /**
     * Checks a the reciprocal condition number
     * Generates an exception if value represents a NON PSD matrix
     * Inverting condition provides a test for IEC 559 NaN values
     */
    void checkPSD(double rcond, const char* error_description) const {
        if (rcond < 0)
            throw NumericException(error_description);
    }

    /**
     * Checks a reciprocal condition number
     * Generates an exception if value represents a NON PD matrix
     * I.e. rcond is bellow given conditioning limit
     * Inverting condition provides a test for IEC 559 NaN values
     */
    void checkPD(double rcond, const char* error_description) const {
        if (rcond < limit_pd)
            throw NumericException(error_description);
    }

    /**
     * Checks if the matrix is symmetric
     */
     static bool isSymmetric(const Eigen::Ref<Matrix>& M) {
        return M.template isApprox(M.transpose());
     }

    static bool isPSD(const Eigen::Ref<Matrix>& M) {
        return isSymmetric(M) && M.ldlt().isPositive();
    }

    static bool isPD(const Eigen::Ref<Matrix>& M) {
        if (M.llt().info() == Eigen::NumericalIssue)
            return false;
        return isSymmetric(M);
    }
private:
    double limit_pd;
};

#endif //BAYSIS_MATSUPPORT_HPP
