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
#include <limits>
#include "matsupport.hpp"


enum class ParamType { diagm=1, symm, vec, constm, constv };

struct IParam {
    virtual ~IParam() = default;

    virtual void setPrior(const std::vector<double> &prior_param) { prior = prior_param; }
    virtual void setSupport(const std::pair<double, double>& sprt) { support = sprt; }
    virtual double logDensity(double) const = 0;
    virtual void update(double) = 0;
    virtual double variance() const = 0;
    virtual void initDiagonal(double) { }  // For the symmetric matrices only

protected:
    template<typename Dist, typename RNG, std::size_t... Is>
    double init_draw(std::shared_ptr<RNG> rng, std::index_sequence<Is...>);

    template<typename Dist, std::size_t... Is>
    double log_density(double x, std::index_sequence<Is...>) const;

    template<typename Dist, std::size_t... Is>
    double get_variance(std::index_sequence<Is...>) const;

    std::vector<double> prior;
    std::pair<double, double> support{std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
};


template<typename PriorDist>
struct DiagonalMatrixParam: public IParam {
    enum { Type = static_cast<std::size_t>(ParamType::diagm) };
    static std::string name() { return "Diagonal matrix"; }

    explicit DiagonalMatrixParam(std::size_t shape);

    static std::size_t Id() { return Type * 10 + PriorDist::Id(); }
    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng);
    double logDensity(double x) const override;
    void update(double driver) override;
    double variance() const override;

    Matrix param;
};


template<typename PriorDist>
struct SymmetricMatrixParam: public IParam {
    enum { Type = static_cast<std::size_t>(ParamType::symm) };
    static std::string name() { return "Symmetric matrix"; }

    static std::size_t Id() { return Type * 10 + PriorDist::Id(); }

    explicit SymmetricMatrixParam(std::size_t shape);

    void initDiagonal(double diag) override { this->diagonal = diag; }
    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng);
    double logDensity(double x) const override;
    void update(double driver) override;
    double variance() const override;

    Matrix param;
private:
    double diagonal{1.};
};


struct AutoregressiveStationaryCov {
    void update(const Matrix& diag, const Matrix& symm);
    Matrix param;
};


template<typename PriorDist>
struct VectorParam: public IParam {
    enum { Type = static_cast<std::size_t>(ParamType::vec) };
    static std::string name() { return "Vector"; }

    static std::size_t Id() { return Type * 10 + PriorDist::Id(); }

    explicit VectorParam(std::size_t shape): param(shape) { }

    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng);
    double logDensity(double x) const override;
    void update(double driver) override;
    double variance() const override;

    Vector param;
};


struct ConstMatrix: IParam {
    enum { Type = static_cast<std::size_t>(ParamType::constm) };

    static std::size_t Id() { return Type * 10; }

    ConstMatrix(std::size_t shape, double constant): param(Matrix::Constant(shape, shape, constant)) { }
    explicit ConstMatrix(std::size_t shape): ConstMatrix(shape, 1.) { }

    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng) { return 0.; }
    double logDensity(double x) const override { return 0.; }
    void update(double driver) override { }
    double variance() const override { return 0.; }

    Matrix param;
};


struct ConstVector: IParam {
    enum { Type = static_cast<std::size_t>(ParamType::constv) };

    static std::size_t Id() { return Type * 10; }

    ConstVector(std::size_t shape, double constant): param(Vector::Constant(shape, constant)) { }
    explicit ConstVector(std::size_t shape): ConstVector(shape, 1.) { }

    template<typename RNG>
    double initDraw(std::shared_ptr<RNG> rng) { return 0.; }
    double logDensity(double x) const override { return 0.; }
    void update(double driver) override { }
    double variance() const override { return 0.; }

    Vector param;
};


template<typename Dist, size_t... Is>
double IParam::log_density(double x, std::index_sequence<Is...>) const {
    if (prior.size() < sizeof...(Is)) {
        throw LogicException(
                "Number of provided parameters is less than required for the specified prior distribution");
    }
    if (x < support.first || x > support.second)
        return std::numeric_limits<double>::lowest();
    return Dist::logDensity(x, prior[Is]...);
}

template<typename Dist, typename RNG, size_t... Is>
double IParam::init_draw(std::shared_ptr<RNG> rng, std::index_sequence<Is...>) {
    RandomSample<RNG, typename Dist::Dist_type> rsg(rng);
    typename RandomSample<RNG, typename Dist::Dist_type>::Param_type params{prior[Is]...};
    return rsg.draw(params);
}

template<typename Dist, size_t... Is>
double IParam::get_variance(std::index_sequence<Is...>) const {
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
double DiagonalMatrixParam<PriorDist>::logDensity(double x) const {
    return log_density<PriorDist>(x, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
double DiagonalMatrixParam<PriorDist>::variance() const {
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
double SymmetricMatrixParam<PriorDist>::logDensity(double x) const {
    return log_density<PriorDist>(x, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
void SymmetricMatrixParam<PriorDist>::update(double driver) {
    double diagconst = param(0, 0);
    param.setConstant(driver);
    param.diagonal().setConstant(diagconst);
}

template<typename PriorDist>
double SymmetricMatrixParam<PriorDist>::variance() const {
    return get_variance<PriorDist>(std::make_index_sequence<PriorDist::Nparams>());
}


template<typename PriorDist>
template<typename RNG>
double VectorParam<PriorDist>::initDraw(std::shared_ptr<RNG> rng) {
    return init_draw<PriorDist>(rng, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
double VectorParam<PriorDist>::logDensity(double x) const {
    return log_density<PriorDist>(x, std::make_index_sequence<PriorDist::Nparams>());
}

template<typename PriorDist>
void VectorParam<PriorDist>::update(double driver) {
    param.setConstant(driver);
}

template<typename PriorDist>
double VectorParam<PriorDist>::variance() const {
    return get_variance<PriorDist>(std::make_index_sequence<PriorDist::Nparams>());
}


inline void AutoregressiveStationaryCov::update(const Matrix& diag, const Matrix& symm) {
    double phi = diag(0,0);
    phi = 1 - phi*phi;
    param = symm / phi;
}


#endif //BAYSIS_PARAMGENERATORS_HPP
