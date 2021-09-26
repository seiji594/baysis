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


struct IParam {
    virtual ~IParam() = default;

    virtual void setPrior(const std::vector<double> &prior_param);
    virtual double logDensity(double) = 0;
    virtual void update(double) = 0;
    virtual double variance() = 0;

protected:
    template<typename Dist, typename RNG, std::size_t... Is>
    double init_draw(std::shared_ptr<RNG> rng, std::index_sequence<Is...>);

    template<typename Dist, std::size_t... Is>
    double log_density(double x, std::index_sequence<Is...>);

    template<typename Dist, std::size_t... Is>
    double get_variance(std::index_sequence<Is...>);

    std::vector<double> prior;
};


template<typename PriorDist>
struct DiagonalMatrixParam: public IParam {
    explicit DiagonalMatrixParam(std::size_t shape);

    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng);
    double logDensity(double x) override;
    void update(double driver) override;
    double variance() override;

    Matrix param;
};


template<typename PriorDist>
struct SymmetricMatrixParam: public IParam {
    explicit SymmetricMatrixParam(std::size_t shape);

    void initDiagonal(double diag) { this->diagonal = diag; }
    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng);
    double logDensity(double x) override;
    void update(double driver) override;
    double variance() override;

    Matrix param;
private:
    double diagonal{1.};
};


struct AutoregressiveStationaryCov {
    template<typename Sym, typename Diag>
    void update(const Sym& symm, const Diag& diag);

    Matrix param;
};


template<typename PriorDist>
struct VectorParam: public IParam {
    explicit VectorParam(std::size_t shape): param(shape) { }

    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng);
    double logDensity(double x) override;
    void update(double driver) override;
    double variance() override;

    Vector param;
};


struct ConstMatrix: IParam {
    ConstMatrix(std::size_t shape, double constant): param(Matrix::Constant(shape, shape, constant)) { }
    explicit ConstMatrix(std::size_t shape): ConstMatrix(shape, 1.) { }

    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng) { return 0.; }
    double logDensity(double x) override { return 0.; }
    void update(double driver) override { }
    double variance() override { return 0.; }

    Matrix param;
};


struct ConstVector: IParam {
    ConstVector(std::size_t shape, double constant): param(Vector::Constant(shape, constant)) { }
    explicit ConstVector(std::size_t shape): ConstVector(shape, 1.) { }

    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng) { return 0.; }
    double logDensity(double x) override { return 0.; }
    void update(double driver) override { }
    double variance() override { return 0.; }

    Vector param;
};


void IParam::setPrior(const std::vector<double> &prior_param) {
    prior = prior_param;
}

template<typename Dist, size_t... Is>
double IParam::log_density(double x, std::index_sequence<Is...>) {
    if (prior.size() < sizeof...(Is)) {
        throw LogicException(
                "Number of provided parameters is less than required for the specified prior distribution");
    }
    return Dist::logDensity(x, prior[Is]...);
}

template<typename Dist, typename RNG, size_t... Is>
double IParam::init_draw(std::shared_ptr<RNG> rng, std::index_sequence<Is...>) {
    RandomSample<RNG, typename Dist::Dist_type> rsg(rng);
    typename RandomSample<RNG, typename Dist::Dist_type>::Param_type params{prior[Is]...};
    return rsg.draw(params);
}

template<typename Dist, size_t... Is>
double IParam::get_variance(std::index_sequence<Is...>) {
    return Dist::variance(prior[Is]...);
}


template<typename PriorDist>
DiagonalMatrixParam<PriorDist>::DiagonalMatrixParam(const size_t shape): param(shape, shape) {
    param.setZero();
}

template<typename PriorDist>
template<typename RNG>
double DiagonalMatrixParam<PriorDist>::initDraw(std::shared_ptr<RNG> rng) {
    return init_draw<PriorDist>(rng, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
void DiagonalMatrixParam<PriorDist>::update(const double driver) {
    param.diagonal().setConstant(driver);
}

template<typename PriorDist>
double DiagonalMatrixParam<PriorDist>::logDensity(double x) {
    return log_density<PriorDist>(x, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
double DiagonalMatrixParam<PriorDist>::variance() {
    return get_variance<PriorDist>(std::make_index_sequence<PriorDist::Nparams>());
}


template<typename PriorDist>
SymmetricMatrixParam<PriorDist>::SymmetricMatrixParam(std::size_t shape): param(shape, shape) {
    param.diagonal().setConstant(diagonal);
}

template<typename PriorDist>
template<typename RNG>
double SymmetricMatrixParam<PriorDist>::initDraw(std::shared_ptr<RNG> rng) {
    return init_draw<PriorDist>(rng, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
double SymmetricMatrixParam<PriorDist>::logDensity(double x) {
    return log_density<PriorDist>(x, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
void SymmetricMatrixParam<PriorDist>::update(double driver) {
    double diagconst = param(0, 0);
    param.setConstant(driver);
    param.diagonal().setConstant(diagconst);
}

template<typename PriorDist>
double SymmetricMatrixParam<PriorDist>::variance() {
    return get_variance<PriorDist>(std::make_index_sequence<PriorDist::Nparams>());
}


template<typename PriorDist>
template<typename RNG>
double VectorParam<PriorDist>::initDraw(std::shared_ptr<RNG> rng) {
    return init_draw<PriorDist>(rng, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
double VectorParam<PriorDist>::logDensity(double x) {
    return log_density<PriorDist>(x, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
void VectorParam<PriorDist>::update(double driver) {
    param.setConstant(driver);
}

template<typename PriorDist>
double VectorParam<PriorDist>::variance() {
    return get_variance<PriorDist>(std::make_index_sequence<PriorDist::Nparams>());
}


template<typename Sym, typename Diag>
void AutoregressiveStationaryCov::update(const Sym &symm, const Diag &diag) {
    double phi = diag.param(0,0);
    phi = 1 - phi*phi;
    param = symm.param / phi;
}


#endif //BAYSIS_PARAMGENERATORS_HPP
