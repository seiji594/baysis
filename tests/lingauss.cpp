//
// lingauss.cpp
// Baysis
//
// Created by Vladimir Sotskov on 14/08/2021, 21:12.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include "../baysis/matsupport.hpp"
#include "../baysis/filterschemes.cpp"
#include "../baysis/samplingschemes.hpp"
#include "../baysis/algorithms.hpp"
#include "../baysis/dataprovider.hpp"

using namespace std;


/**
 * Test run for linear gaussian models and comparing to exact solution with Kalman filter
 * Specification of models taken from 2018 paper by Shestopaloff & Neal
 */
int main(int argc, const char * argv[]) {
    // Set up the model
    int T{250}, dim{10};
    double rho{0.7}, th{0.9}, sgm{0.6}, c{-0.4};
    Matrix A(dim,dim), C(dim, dim), Q(dim, dim), R(dim, dim), Sinit(dim, dim);
    Vector muinit(Vector::Zero(dim));
    A = Matrix::Identity(dim, dim) * th;
    Q = Q.setConstant(rho);
    Q += Matrix::Identity(dim, dim) * (1 - rho);
    R.setIdentity();
    C = Matrix::Identity(dim, dim) * sgm;
    Sinit = Q * (1/(1 - th*th));
/*
    std::cout << "================= Models ================" << std::endl;
    std::cout << "Transition model:" << std::endl;
    std::cout << "\tParameters" << std::endl;
    std::cout << "Coefficients:\n" << A << std::endl;
    std::cout << "Covariance:\n" << Q << std::endl;
    std::cout << "Prior\n" << "mean:\t" << muinit.transpose() << "\ncov:\n" << Sinit << std::endl;
    std::cout << "\nObservation model:" << std::endl;
    std::cout << "Coefficients:\n" << C << std::endl;
    std::cout << "Covariance:\n" << R << std::endl;
*/

    std::shared_ptr<LGTransitionStationary> trm = std::make_shared<LGTransitionStationary>(T, dim, 0);
    std::shared_ptr<LGObservationStationary> obsm = std::make_shared<LGObservationStationary>(T, dim, dim);
    trm->init(A,Q);
    trm->setPrior(muinit, Sinit);
    obsm->init(C, R);

    DataGenerator<LGTransitionStationary, LGObservationStationary> simulator(trm, obsm, 1);
    std::cout << "Observations:\n" << simulator.getData() << std::endl;
/*
    //! Metropolis single state scheme
    int thin = 10;
    using Sampler_type = schemes::SingleStateScheme<LGTransitionStationary, LGObservationStationary, std::mt19937>;
    std::shared_ptr<Sampler_type> sampler(make_shared<Sampler_type>());
    algos::MCMC<Sampler_type> ssmetropolis(trm, obsm, sampler, 1e6, {0.2, 0.8}, thin);
    Matrix init_x(Matrix::Constant(dim, T, 0.));  // Initial sample
    ssmetropolis.initialize(simulator.getData(), init_x, 1);
    ssmetropolis.run();
    SampleAccumulator& accumulator = ssmetropolis.getStatistics();
    accumulator.setBurnin(.1);

    std::cout << "Total samples:" << accumulator.getSamples().size() << std::endl;
    std::cout << "Acceptances:" << std::endl;
    for (auto& acc: accumulator.getAcceptances()) {
        std::cout << acc << "\t";
    }
    std::cout << "\nDuration: " << accumulator.totalDuration() << "ms" << std::endl;
    Vector means = accumulator.getSmoothedMeans(T-1);
    std::cout << "Mean at T:\n" << means.transpose() << std::endl;
    std::cout << "Cov at T:\n" << accumulator.getSmoothedCov(means, T-1) << std::endl;
*/
    //! EHMM scheme
    int poolsz = 50;
    using Sampler_type = schemes::EmbedHmmSchemeND<LGTransitionStationary, LGObservationStationary, std::mt19937>;
    std::shared_ptr<Sampler_type> sampler(make_shared<Sampler_type>(poolsz));
    algos::MCMC<Sampler_type> ehmm(trm, obsm, sampler, 90, {0.1, 0.4}, 1, true);
    Matrix init_x(Matrix::Constant(dim, T, 0.));  // Initial sample
    ehmm.initialise(simulator.getData());
    ehmm.run();
    SampleAccumulator& accumulator = ehmm.getStatistics();
    accumulator.setBurnin(.1);

    std::cout << "Total samples:" << accumulator.getSamples().size() << std::endl;
    std::cout << "Acceptances:" << std::endl;
    for (auto& acc: accumulator.getAcceptances()) {
        std::cout << acc << "\t";
    }
    std::cout << "\nDuration: " << accumulator.totalDuration() << "ms" << std::endl;
    Vector means = accumulator.getSmoothedMeans(T-1);
    std::cout << "Mean at T:\n" << means.transpose() << std::endl;
    std::cout << "Cov at T:\n" << accumulator.getSmoothedCov(means, T-1) << std::endl;

    //! Kalman smoother for the same model
    algos::KalmanSmoother<schemes::InformationScheme, schemes::TwoFilterScheme> kalmsm(trm, obsm);
    kalmsm.initialise(simulator.getData());
    kalmsm.run();

    std::cout << "Smoothed state mean at T:\n" << kalmsm.smoothed_means.col(T-1).transpose() << std::endl;
    std::cout << "Smoothed state covs at T:\n" << kalmsm.smoothed_covs.block(0, T-dim, dim, dim) << std::endl;

    return 0;
}

