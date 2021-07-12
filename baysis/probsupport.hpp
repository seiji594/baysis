//
// probsupport.hpp
// Baysis
//
// Created by Vladimir Sotskov on 25/06/2021, 14:26.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_PROBSUPPORT_HPP
#define BAYSIS_PROBSUPPORT_HPP

#include <random>
#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>
#include "baysisexception.hpp"
#include "matsupport.hpp"

const double PI = std::acos(-1);


/**
 * Random number traits. Used with known distributions.
 * @tparam URNG - univariate random number generator type
 * @tparam D - distribution from std::random
 */
template <class URNG>
struct GenericPseudoRandom {
    // typedefs
    typedef URNG Rng_type;
    // factories
    static Rng_type makeRnGenerator(u_long seed) {
        Rng_type gen;
        gen.seed(seed);
        return gen;
    }

    static Rng_type makeRnGenerator() {
        std::random_device r;
        std::seed_seq seeds({r(), r(), r(), r(), r(), r(), r()});
        return Rng_type(seeds);
    }
};

/**
 * Generic univariate sampler. Samples variates from the distribution type Dist
 * using random number generator type RNG
 * @tparam RNG - random number generator type passed to GenericPseudoRandom
 * @tparam Dist - STL random distribution type
 */
template<typename RNG, typename Dist>
struct RandomSample {
    typedef typename Dist::result_type Value_type;
    typedef typename Dist::param_type Param_type;
    typedef Eigen::Matrix<Value_type, Eigen::Dynamic, 1> Sample_type;

    RandomSample(const std::shared_ptr<RNG>& rgen, const Dist& d): rng(rgen), dist(d.param()) { }
    explicit RandomSample(const std::shared_ptr<RNG>& rgen): RandomSample(rgen, Dist{}) { }
    RandomSample() : RandomSample(nullptr) { }

    /**
     * Samples n i.i.d variates from the standard distribution D
     * @param n - number of variates to return
     * @return vector of variates
     */
    Sample_type draw(std::size_t n) {
        Sample_type retval(n);
        for (std::size_t i=0; i < n; ++i) {
            retval(i) = dist(*rng);
        }
        return retval;
    }

    Value_type draw() {
        return dist(*rng);
    }

    Value_type draw(const Param_type& params) {
        return dist(*rng, params);
    }

    Param_type param() const {
        return dist.param();
    }
private:
    std::shared_ptr<RNG> rng;
    Dist dist;
};

/**
 * Univariate/multivarite NormalDist distribution
 */
class NormalDist {
public:
    typedef std::normal_distribution<> Dist_type;
    typedef Eigen::Matrix<Dist_type::result_type, Eigen::Dynamic, 1> Sample_type;
    /**
     * Returns log density of normal distribution given the datapoint x; constants are ignored
     * @param x - a normal variable
     * @param mu, sigma - parameters of normal distribution
     * @return log density
     */
    static inline double logDensity(const double x, const double mu, const double sigma) {
        return -log(sigma) - 0.5 * pow(x - mu, 2) / pow(sigma, 2);
    }
    /**
     * Multivariate normal overloading of logDensity
     * @param x - the multivariate normal variable
     * @param mu - vector of marginal means
     * @param L - lower triangular matrix of Cholesky decomposition of the covariance matrix
     * @return log density
     */
    template<typename DerivedA, typename DerivedB, typename DerivedC>
    static inline double logDensity(const Eigen::MatrixBase<DerivedA>& x,
                                    const Eigen::MatrixBase<DerivedB>& mu,
                                    const Eigen::LLT<DerivedC>& L) {
        double sqstv = L.template solve(x - mu).cwiseProduct(x - mu).sum();
        double log_det = log(L.matrixL().toDenseMatrix().diagonal().array()).sum();
        return -log_det - 0.5 * sqstv;
    }
    /**
     * returns loglikelihood of the data given the distribution parameters
     * @param data - vector of i.i.d. normal variables
     * @param mu, sigma - parameters of normal distribution
     * @return value of loglikelihood
     */
    static inline double logLikelihood(const Eigen::Ref<Sample_type>& data, const double mu, const double sigma) {
        double retval(0);
        size_t n = data.size();
        for (int i = 0; i < n; ++i) {
            retval += pow(data(i) - mu, 2);
        }
        return -0.5 * n * log(2. * PI) - n * log(sigma) - 0.5 * retval / pow(sigma, 2);
    }
    /**
     * Samples n i.i.d variates from the normal distribution
     * @param n - number of variates to return
     * @param mu, sigma - parameters of normal distribution
     * @param rsg - random sequence generator using RNG random numbers generator
     * @return vector of variates
     */
    template<typename RNG>
    static inline Sample_type sample(std::size_t n, const double mu, const double sigma,
                                     RandomSample<RNG, Dist_type>& rsg, const double scale=1.) {
        return scale * sigma * rsg.draw(n).array() + mu;
    }
    /**
     * Samples one variate from a multivariate normal distribution
     * @param mu - vector of marginal means
     * @param L - lower triangular matrix of Cholesky decomposition of the covariance matrix
     * @param rsg - random sequence generator using RNG random numbers generator
     * @param scale - scaling for covariance
     * @return a multivariate normal variable
     */
    template<typename DerivedA, typename DerivedB, typename RNG>
    static inline Sample_type sample(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::LLT<DerivedB>& L,
                                     RandomSample<RNG, Dist_type>& rsg, const double scale=1.) {
        return mu + L.matrixL() * rsg.draw(mu.size()) * scale;
    }
};

/**
 * A univariate/multivariate PoissonDist distribution. Only zero covariance is implemented in case of multivariate version
 */
class PoissonDist {
public:
    typedef std::poisson_distribution<> Dist_type;
    typedef Eigen::Matrix<Dist_type::result_type, Eigen::Dynamic, 1> Sample_type;
    /**
     * Returns log density of poisson distribution given the datapoint x; constants are ignored
     * @param x - a poisson variable
     * @param lambda - parameter of poisson distribution
     * @return log density
     */
    static inline double logDensity(const double x, const double lambda) {
        return x * log(lambda) - lambda;
    }
    /**
     * Multivariate poisson overloading of logDensity
     * @param x - the multivariate normal variable
     * @param lambda - vector of parameters of multivariate poisson distribution
     * @param lambda0 - variance parameter of the multivariate poisson distribution
     * @return log density
     */
    template<typename DerivedA, typename DerivedB>
    static inline double logDensity(const Eigen::MatrixBase<DerivedA>& x,
                                    const Eigen::ArrayBase<DerivedB>& lambda,
                                    const double lambda0=0) {
        if (lambda0 != 0) {
            throw LogicException("Non-zero variance for multivariate Poisson not implemented");
        }
        return (x.template cast<double>().array() * lambda.log() - lambda).sum();
    }
    /**
     * returns loglikelihood of the data given the distribution parameters
     * @param data - vector of i.i.d. normal variables
     * @param lambda - parameter of poisson distribution
     * @return value of loglikelihood
     */
    static inline double logLikelihood(const Eigen::Ref<Sample_type>& data, const double lambda) {
        double sumlfact((data.cast<double>().array() + 1).lgamma().sum());
        double sumx(data.cast<double>().sum());
        return sumx * log(lambda) - sumlfact - data.size() * lambda;
    }
    /**
     * Samples n i.i.d variates from the poisson distribution
     * @param n - number of variates to return
     * @param rsg - random sequence generator using RNG random numbers generator; should be instantiated with
     *              the poisson distribution with correct parameter
     * @return vector of variates
     */
    template<typename RNG>
    static inline Sample_type sample(std::size_t n, RandomSample<RNG, Dist_type>& rsg) {
        return rsg.draw(n);
    }
    /**
     * Samples one variate from a multivariate poisson distribution with 0 variance parameter;
     * @param lambda - vector of parameters of poisson distribution
     * @param rng - random numbers generator
     * @return a multivariate normal variable
     */
    template<typename RNG>
    static inline Sample_type sample(const Vector& lambda, RandomSample<RNG, Dist_type>& rsg) {
        using Param_type = typename RandomSample<RNG, Dist_type>::Param_type;
        Sample_type z(lambda.size());
        for (int i = 0; i < lambda.size(); ++i) {
            Param_type params(lambda(i));
            z(i) = rsg.draw(params);
        }
        return z;
    }
};

#endif //BAYSIS_PROBSUPPORT_HPP
