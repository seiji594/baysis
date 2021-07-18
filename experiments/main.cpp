//
// main.cpp
// Baysis
//
// Created by Vladimir Sotskov on 25/06/2021, 17:51.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include <chrono>
#include "../baysis/probsupport.hpp"
#include "../baysis/filterschemes.cpp"
#include "../baysis/samplingschemes.hpp"
#include "../baysis/algorithms.hpp"
#include "../baysis/accumulator.hpp"
#include "../baysis/dataprovider.hpp"

using namespace std;
using namespace schemes;


int main(int argc, const char * argv[]) {

    std::shared_ptr<std::mt19937> rng = std::make_shared<std::mt19937>(GenericPseudoRandom<std::mt19937>::makeRnGenerator(1));
    RandomSample<std::mt19937, std::normal_distribution<> > rsg{rng};
/*
    RandomSample<std::mt19937, std::poisson_distribution<> > rsg2{rng};
    std::uniform_int_distribution<> randint(1,10);
    RandomSample<std::mt19937, std::uniform_int_distribution<> > rsg3{rng, randint};

    // Draw 10 random numbers from standard normal
    cout << "Vector of 10 random numbers from normal:\n" << rsg.draw(10) << endl;
    cout << "Vector of 10 random numbers from poisson:\n" << rsg2.draw(10) << endl;
    cout << "Vector of 10 random numbers from U(1,10):\n" << rsg3.draw(10) << endl;

    // Calc log density and log likelihood of Normal
    size_t n(3);
    Vector norm_sample(n);
    Vector means(n);
    means << 2.1, 2.8, 1.9;
    Matrix cov(n, n);
    cov << 0.5, 0.25, 0.34, 0.25, 0.6, 0.36, 0.34, 0.36, 0.4;
    Eigen::LLT<Matrix> llt(static_cast<Eigen::Index>(n));
    llt.compute(cov);
    NumericalRcond().checkPD(llt.rcond(), "Matrix is not positive definite");
    norm_sample = NormalDist::sample(means, llt, rsg);
    cout << "A multivariate sample from normal distribution:\n" << norm_sample << endl;
    cout << "Logdensity of the multivariate sample:\n" << NormalDist::logDensity(norm_sample, means, llt) << endl;
    cout << "Low triangular matrix of Cholesky decomposition:\n" << llt.matrixL().toDenseMatrix() << endl;

    norm_sample = NormalDist::sample(10, 3.4, 0.5, rsg);
    cout << "A sample of 10 i.i.d. normally distributed variables N(3.4, 0.5):\n" << norm_sample << endl;
    cout << "loglikelihood of the sample:\n" << NormalDist::logLikelihood(norm_sample, 3.4, 0.5) << endl;

    // Calc log density and log likelihood of Poisson
    n = 5;
    Vector_int pois_sample(n);
    Vector lambdas(n);
    lambdas << 2.1, 2.8, 1.9, 0.5, 1.7;
    pois_sample = PoissonDist::sample(lambdas, rsg2);
    cout << "A multivariate sample from poisson distribution:\n" << pois_sample << endl;
    cout << "Logdensity of the multivariate poisson:\n" << PoissonDist::logDensity(pois_sample, lambdas.array()) << endl;

    std::poisson_distribution<> pois(3.4);
    RandomSample<std::mt19937, std::poisson_distribution<> > rsg4{rng, pois};
    pois_sample = PoissonDist::sample(n, rsg4);
    cout << "A sample of 5 i.i.d. poisson distributed variables Poi(3.4):\n" << pois_sample << endl;
    cout << "loglikelihood of the sample:\n" << PoissonDist::logLikelihood(pois_sample, 3.4) << endl;
*/
/*
    std::size_t n = 15;
    Matrix A(n,n), L(n, n), eye(Matrix::Identity(n, n));
    Matrix Ainv1(n, n), Ainv2(n, n);
    L.setRandom();
    A = L * L.transpose();

    auto t1 = chrono::high_resolution_clock::now();
    Eigen::LLT<Matrix> llt(A);
    auto t2 = chrono::high_resolution_clock::now();
    Matrix ltri(llt.matrixL());
    auto t2a = chrono::high_resolution_clock::now();
    ltri = ltri.inverse();
    Ainv1.noalias() = ltri.transpose() * ltri;
    auto t3 = chrono::high_resolution_clock::now();
    Ainv2 = A.inverse();
    auto t4 = chrono::high_resolution_clock::now();
    Ainv2 = llt.solve(eye);
    auto t5 = chrono::high_resolution_clock::now();

    std::cout << "To calc llt = " << chrono::duration_cast<chrono::microseconds>(t2-t1).count() << std::endl;
    std::cout << "To get low triangula matrix = " << chrono::duration_cast<chrono::microseconds>(t2a-t2).count() << std::endl;
    std::cout << "To calc inverse with llt = " << chrono::duration_cast<chrono::microseconds>(t3-t2a).count() << std::endl;
    std::cout << "To calc inverse directly = " << chrono::duration_cast<chrono::microseconds>(t4-t3).count() << std::endl;
    std::cout << "To calc inverse via solve = " << chrono::duration_cast<chrono::microseconds>(t5-t4).count() << std::endl;
    std::cout << A.inverse().isApprox(Ainv1) << std::endl;
    std::cout << A.inverse().isApprox(Ainv2) << std::endl;
//    std::cout << "A.inverse() = \n" << A.inverse() << std::endl;
//    std::cout << "Ainv1 = \n" << Ainv1 << std::endl;
//    std::cout << "Ainv2 = \n" << Ainv2 << std::endl;
//    std::cout << "A.inverse() vs Ainv1 \n" << Ainv1 - A.inverse() << std::endl;
//    std::cout << "A.inverse() vs Ainv2 \n" << Ainv2 - A.inverse() << std::endl;
*/
    /** Some toy models for testing **/
    // Set up the model
    Matrix A(4,4), C(2, 4), Q(4, 4), R(2, 2);
    double delta = 0.1;
    A << 1., 0., delta, 0.,
         0., 1., 0., delta,
         0., 0., 1., 0.,
         0., 0., 0., 1.;
    C << 1., 0., 0., 0.,
         0., 1., 0., 0.;
    Q.setConstant(0.5);
    Q += Matrix::Identity(4, 4) * 0.5;
    R.setIdentity();

    // Prior for state
    Matrix Sinit;
    Vector minit(4);
    minit << 1., 1., 0.5, 2.;
    Sinit.setIdentity(4, 4);
    Sinit *= 0.75; //

    std::shared_ptr<LGTransitionStationary> trm = std::make_shared<LGTransitionStationary>(10, 4, 0);
    std::shared_ptr<LGObservationStationary> obsm = std::make_shared<LGObservationStationary>(10, 4, 2);
    trm->init(A,Q);
    trm->setPrior(minit, Sinit);
    obsm->init(C, R);

    DataGenerator<LGTransitionStationary, LGObservationStationary> simulator(trm, obsm, 1);
    std::cout << "Observations:\n" << simulator.getData() << std::endl;
/*
    //! Kalman filter/smoother tests
    // Initialize Kalman filter with covariance scheme
    CovarianceScheme kf(4, 2);

    kf.init((*trm).getPriorMean(), (*trm).getPriorCov());

    Matrix state_means(4, 10);
    Matrix state_mean_priors(4, 10);
    Matrix state_covs(4, 4*10);
    Matrix state_cov_priors(4, 4*10);

    for (int i = 0; i < 10; ++i) {
        if (i != 0) {
            kf.predict(*trm); // make prediction of the next state's mean and cov
            state_mean_priors.col(i) = kf.x;
            state_cov_priors.block(0, i * 4, 4, 4) = kf.X;
        }
        kf.observe(*obsm, simulator.next());  // observe data and update the state statistics
        // Saving results
        state_means.col(i) = kf.x;
        state_covs.block(0, i*4, 4, 4) = kf.X;
    }

    std::cout << "State means:\n" << state_means << std::endl;
    std::cout << "State covs:\n" << state_covs << std::endl;

    // Set up Kalman smoother using data calculated by the filter
    RtsScheme rts(4);
    rts.init(state_means.col(9), state_covs.block(0,4*9,4,4));

    Matrix smoothed_means(4, 10);
    Matrix smoothed_covs(4, 4*10);

    for (int i = 8; i >= 0; --i) {
        rts.updateBack(*trm,
                       state_mean_priors.col(i+1),
                       state_means.col(i),
                       state_cov_priors.block(0, 4 * (i+1), 4, 4),
                       state_covs.block(0, 4 * i, 4, 4));
        smoothed_means.col(i) = rts.x;
        smoothed_covs.block(0,i*4,4,4) = rts.X;
    }

    std::cout << "Smoothed state means:\n" << smoothed_means << std::endl;
    std::cout << "Smoothed state covs:\n" << smoothed_covs << std::endl;

    // Set up information filter
    InformationScheme inff(4, 2);
    inff.init((*trm).getPriorMean(), (*trm).getPriorCov());

    state_means.setZero();
    state_mean_priors.setZero();
    state_covs.setZero();
    state_cov_priors.setZero();

    state_mean_priors.col(0) = minit;
    state_cov_priors.block(0, 0, 4, 4) = Sinit;

    simulator.reset();

    for (int i = 0; i < 10; ++i) {
        if (i != 0) {
            inff.predict(*trm); // make prediction of the next state's mean and cov
            state_mean_priors.col(i) = inff.x;
            state_cov_priors.block(0, i * 4, 4, 4) = inff.X;
        }
        inff.observe(*obsm, simulator.next());  // observe data and update the state statistics
        // Saving results
        state_means.col(i) = inff.x;
        state_covs.block(0, i*4, 4, 4) = inff.X;
    }

    std::cout << "State means:\n" << state_means << std::endl;
    std::cout << "State covs:\n" << state_covs << std::endl;

    // Set up "two-way filter" smoother
    TwoWayScheme two(4);
    //FIXME: make the init API consistent between schemes (init may take obs model in all (but not necessarily use it)
    // and obs model may have a handle to observations)
    two.initInformation(*obsm, simulator.at(9),
                        state_means.col(9),
                        state_covs.block(0,4*9,4,4));

    smoothed_means.setZero();
    smoothed_covs.setZero();

    // FIXME: make the smoother API consistent : both should have either one or two functions per cycle
    for (int i = 8; i >= 0; --i) {
        two.predictBack(*trm);
        two.updateBack(*obsm, simulator.at(i),
                       state_mean_priors.col(i),
                       state_cov_priors.block(0, 4*i, 4, 4));
        smoothed_means.col(i) = two.x;
        smoothed_covs.block(0,i*4,4,4) = two.X;
    }

    std::cout << "Smoothed state means:\n" << smoothed_means << std::endl;
    std::cout << "Smoothed state covs:\n" << smoothed_covs << std::endl;
*/
/*
    //! Single state Metropolis sampler
    using Sampler_type = SingleStateScheme<LGTransitionStationary, LGObservationStationary, std::mt19937>;
    std::shared_ptr<Sampler_type> sampler(make_shared<Sampler_type>());

    algos::MCMC<Sampler_type> ssmetropolis(trm, obsm, sampler, 1000, {0.2, 0.8});
    Matrix init_x(Matrix::Constant(4, 10, 0.));  // Initial sample
    ssmetropolis.initialise(simulator.getData(), init_x, 1);
    ssmetropolis.run();

    for (const auto& s: ssmetropolis.accumulator.samples) {
        std::cout << s << std::endl;
    }

    std::cout << "Acceptances:" << std::endl;
    for (auto& acc: ssmetropolis.accumulator.accepts) {
        std::cout << acc << "\t";
    }
    std::cout << "\nDuration: " << ssmetropolis.accumulator.duration << "ms" << std::endl;
*/
    //! Embedded HMM sampler
    using Sampler_type = EmbedHmmSchemeND<LGTransitionStationary, LGObservationStationary, std::mt19937>;
    std::shared_ptr<Sampler_type> sampler(make_shared<Sampler_type>(5));  // <-- 5 pool states

    algos::MCMC<Sampler_type> ehmm(trm, obsm, sampler, 100, {0.1, 0.3});
    Matrix init_x(Matrix::Constant(4, 10, 0.));  // Initial sample
    ehmm.initialise(simulator.getData(), init_x, 1);
    ehmm.run();

    for (const auto& s: ehmm.accumulator.samples) {
        std::cout << s << std::endl;
    }

    std::cout << "Acceptances:" << std::endl;
    int i = 0;
    for (auto& acc: ehmm.accumulator.accepts) {
        if (i == 10)
            std::cout << std::endl;
        std::cout << acc << "\t";
        ++i;
    }
    std::cout << "\nDuration: " << ehmm.accumulator.duration << "ms" << std::endl;

    return 0;
}