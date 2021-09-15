//
// paramgenerators.hpp
// Baysis
//
// Created by Vladimir Sotskov on 10/09/2021, 16:20.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_PARAMGENERATORS_HPP
#define BAYSIS_PARAMGENERATORS_HPP

#include <utility>
#include <vector>
#include "matsupport.hpp"


//template<typename PriorDist>
struct IParam {
    virtual ~IParam() = default;

    virtual void setPrior(const std::vector<double> &prior_param) { prior = prior_param; }
    virtual double logDensity(double) = 0;
    virtual void update(double) = 0;

protected:
    template<typename Dist, std::size_t... Is>
    static double log_density(double x, const std::vector<double>& params, std::index_sequence<Is...>);

    std::vector<double> prior;
};

template<typename PriorDist, std::size_t Nparams>
struct DiagonalMatrixParam: public IParam {
    explicit DiagonalMatrixParam(std::size_t shape);

    double logDensity(double x) override;
    void update(double driver) override;

    Matrix param;
};


template<typename PriorDist, std::size_t Nparams>
struct SymmetricMatrixParam: public IParam {
    explicit SymmetricMatrixParam(std::size_t shape);

    void initDiagonal(double diag) { this->diag = diag; }
    double logDensity(double x) override;
    void update(double driver) override;

    Matrix param;
private:
    double diag{-1.};
};


template<typename PriorDist, std::size_t Nparams>
struct VectorParam: public IParam {
    explicit VectorParam(std::size_t shape);

    double logDensity(double x) override;
    void update(double driver) override;

    Vector param;
};


struct ConstMatrix: IParam {
    explicit ConstMatrix(std::size_t shape, double constant): param(Matrix::Constant(shape, shape, constant)) { }

    double logDensity(double x) override { return 1.; }
    void update(double driver) override { }

    Matrix param;
};


struct ConstVector: IParam {
    explicit ConstVector(std::size_t shape, double constant): param(Vector::Constant(shape, constant)) { }

    double logDensity(double x) override { return 1.; }
    void update(double driver) override { }

    Vector param;
};


template<typename Dist, size_t... Is>
double IParam::log_density(double x, const std::vector<double> &params, std::index_sequence<Is...>) {
    if (params.size() < sizeof...(Is)) {
        throw LogicException(
                "Number of provided parameters is less than required for the specified prior distribution");
    }
    return Dist::logDensity(x, params[Is]...);
}


template<typename PriorDist, std::size_t Nparams>
DiagonalMatrixParam<PriorDist, Nparams>::DiagonalMatrixParam(const size_t shape): param(shape, shape) {
    param.setZero();
}

template<typename PriorDist, std::size_t Nparams>
void DiagonalMatrixParam<PriorDist, Nparams>::update(const double driver) {
    param.diagonal().setConstant(driver);
}

template<typename PriorDist, std::size_t Nparams>
double DiagonalMatrixParam<PriorDist, Nparams>::logDensity(double x) {
    return IParam::log_density(x, prior, std::make_index_sequence<Nparams>());
}


template<typename PriorDist, std::size_t Nparams>
SymmetricMatrixParam<PriorDist, Nparams>::SymmetricMatrixParam(std::size_t shape): param(shape, shape) {
    param.setIdentity();
}

template<typename PriorDist, std::size_t Nparams>
double SymmetricMatrixParam<PriorDist, Nparams>::logDensity(double x) {
    return IParam::log_density(x, prior, std::make_index_sequence<Nparams>());
}

template<typename PriorDist, std::size_t Nparams>
void SymmetricMatrixParam<PriorDist, Nparams>::update(double driver) {
    if (diag < 0) {
        // Diagonal is constant, and we're setting only the off-diagonal values
        double diagconst = param(0, 0);
        param.setConstant(driver);
        param.diagonal().setConstant(diagconst);
    } else {
        // We're changing the diagonal only, the rest stays the same
        param.diagonal().setConstant(driver);
    }
}


template<typename PriorDist, std::size_t Nparams>
VectorParam<PriorDist, Nparams>::VectorParam(std::size_t shape): param(shape) { }

template<typename PriorDist, std::size_t Nparams>
double VectorParam<PriorDist, Nparams>::logDensity(double x) {
    return IParam::log_density(x, prior, std::make_index_sequence<Nparams>());
}

template<typename PriorDist, std::size_t Nparams>
void VectorParam<PriorDist, Nparams>::update(double driver) {
    param.setConstant(driver);
}


#endif //BAYSIS_PARAMGENERATORS_HPP
