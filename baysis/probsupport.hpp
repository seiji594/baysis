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

    RandomSample(): rng(GenericPseudoRandom<RNG>::makeRnGenerator()) {}
    RandomSample(u_long seed): rng(GenericPseudoRandom<RNG>::makeRnGenerator(seed)) {}
    RandomSample(const Dist& d, u_long seed=0): dist(d.param()) {
        if (seed == 0) {
            rng = GenericPseudoRandom<RNG>::makeRnGenerator();
        } else {
            rng = GenericPseudoRandom<RNG>::makeRnGenerator(seed);
        }
    }

    /**
     * Samples n i.i.d variates from the standard distribution D
     * @param n - number of variates to return
     * @return vector of variates
     */
    Sample_type draw(std::size_t n) {
        Sample_type retval(n);
        for (std::size_t i=0; i < n; i++) {
            retval(i) = dist(rng);
        }
        return retval;
    }

    Value_type draw() {
        return dist(rng);
    }

    Value_type draw(const Param_type& params) {
        return dist(rng, params);
    }

    Param_type param() const {
        return dist.param();
    }
private:
    Dist dist;
    RNG rng;
};


/**
 * Gaussian autoregressive sampler.
 * From Neal, R. M. (1998) “Regression and classification using Gaussian process priors”,
 * in J.M. Bernardo et al (editors) Bayesian Statistics 6, Oxford University Press
 * @tparam RNtraits - random number traits, defined by GenericPseudoRandom, Normal_rng specialization by default
 */
template<typename RNTraits, typename Dist>
struct GARsampler {
    // TODO: move to Schemes
    using Rng_type = typename RNTraits::Rng_type;

    // TODO: check for efficiency use of Eigen objects
    Vector sample(const Vector& mu, const Vector& cur_x, const Matrix& M, const double eps) const {
        Vector z(cur_x.size());
        Vector shift = mu + (cur_x - mu) * sqrt(1 - eps*eps);
        Rng_type rng = Rng_type::makeRnGenerator();
        Dist dist;

        for (std::size_t i=0; i < cur_x.size(); i++) {
            z[i] = dist(rng);
        }

        // FIXME: check for rcond

        return shift + eps * M * z;
    }
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
    static double logDensity(const double x, const double mu, const double sigma) {
        // FIXME: check params for correctness
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
    static double logDensity(const Eigen::MatrixBase<DerivedA>& x,
                             const Eigen::MatrixBase<DerivedB>& mu,
                             const Eigen::MatrixBase<DerivedC>& L) {
        // TODO: check size conformance
        Vector stzd;
        stzd = L.inverse() * (x - mu);  // <-- apparently this is faster; L.template triangularView<Eigen::Lower>().solve(x - mu);
        double sqstv = pow(stzd.array(), 2).sum();
        double log_det = log(L.diagonal().array()).sum();
        return -log_det - 0.5 * sqstv;
    }
    /**
     * returns loglikelihood of the data given the distribution parameters
     * @param data - vector of i.i.d. normal variables
     * @param mu, sigma - parameters of normal distribution
     * @return value of loglikelihood
     */
    static double logLikelihood(const Eigen::Ref<Sample_type>& data, const double mu, const double sigma) {
        // FIXME: check param for correctness
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
    static Sample_type sample(std::size_t n, const double mu, const double sigma, RandomSample<RNG, Dist_type>& rsg) {
        //FIXME: check params for correctness
        Sample_type z(rsg.draw(n));
        return (sigma * z).array() + mu;
    }
    /**
     * Samples one variate from a multivariate normal distribution
     * @param mu - vector of marginal means
     * @param L - lower triangular matrix of Cholesky decomposition of the covariance matrix
     * @param rsg - random sequence generator using RNG random numbers generator
     * @return a multivariate normal variable
     */
    template<typename DerivedA, typename DerivedB, typename RNG>
    static Sample_type sample(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& L,
                         RandomSample<RNG, Dist_type>& rsg) {
        Sample_type z(rsg.draw(mu.size()));
        return mu + L * z;
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
    static double logDensity(const double x, const double lambda) {
        //FIXME: check param for correctness
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
    static double logDensity(const Eigen::MatrixBase<DerivedA>& x,
                             const Eigen::ArrayBase<DerivedB>& lambda,
                             const double lambda0=0) {
        // FIXME: check params for correctness
        if (lambda0 != 0) {
            throw LogicException("Non-zero variance for multivariate PoissonDist not yet implemented");
        }
        // TODO: check size conformance
        return (x.template cast<double>().array() * lambda.log() - lambda).sum();
    }
    /**
     * returns loglikelihood of the data given the distribution parameters
     * @param data - vector of i.i.d. normal variables
     * @param lambda - parameter of poisson distribution
     * @return value of loglikelihood
     */
    static double logLikelihood(const Eigen::Ref<Sample_type>& data, const double lambda) {
        //FIXME: check param for correctness
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
    static Sample_type sample(std::size_t n, RandomSample<RNG, Dist_type>& rsg) {
        return rsg.draw(n);
    }
    /**
     * Samples one variate from a multivariate poisson distribution with 0 variance parameter;
     * @param lambda - vector of parameters of poisson distribution
     * @param rng - random numbers generator
     * @return a multivariate normal variable
     */
    template<typename RNG, typename Derived>
    static Sample_type sample(const Eigen::DenseBase<Derived>& lambda, RandomSample<RNG, Dist_type>& rsg) {
        //FIXME: check params for correctness
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
