//
// main.cpp
// Baysis
//
// Created by Vladimir Sotskov on 25/06/2021, 17:51.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include <chrono>
#include <cxxabi.h>
//#include <unordered_map>
#include "../baysis/probsupport.hpp"
#include "../baysis/filterschemes.cpp"
#include "../baysis/utilities.hpp"
#include "../baysis/samplingschemes.hpp"
#include "../baysis/algorithms.hpp"
#include "../baysis/models.cpp"
#include "../baysis/h5bridge.cpp"
//#include "../baysis/accumulator.hpp"
#include "../baysis/dataprovider.hpp"
#include "../baysis/paramgenerators.hpp"

using namespace std;
using namespace ssmodels;
using namespace schemes;
/*
template<typename TList, std::size_t... I>
auto a2t_impl(std::index_sequence<I...>)
{
    return std::make_tuple(typeid(typename typelist::tlist_type_at<I, TList>::type).name()...);
}


template<class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple_impl(std::basic_ostream<Ch,Tr>& os,
                      const Tuple& t,
                      std::index_sequence<Is...>) {
    using swallow = int[];
    (void)swallow{0, (void(os << (Is == 0? "" : "\n") << std::get<Is>(t)), 0)...};
}

template<class Ch, class Tr, class... Args>
auto& operator<<(std::basic_ostream<Ch, Tr>& os,
                const std::tuple<Args...>& t) {
    os << "(";
    print_tuple_impl(os, t, std::index_sequence_for<Args...>{});
    return os << ")";
}
*/

int main(int argc, const char * argv[]) {

    // Check if pointer to Rng ensures each Rsg object continues the random number sequence
    std::shared_ptr<std::mt19937> rng = std::make_shared<std::mt19937>(GenericPseudoRandom<std::mt19937>::makeRnGenerator(1));
//    for (int i = 0; i < 3; ++i) {
//        RandomSample<std::mt19937, std::normal_distribution<> > rsg1{rng};
//        std::cout << rsg1.draw(5).transpose() << std::endl;
//        RandomSample<std::mt19937, std::normal_distribution<> > rsg2{rng};
//        std::cout << rsg2.draw(5).transpose() << std::endl;
//    }

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
    std::size_t n = 5;
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
/*
    //! Using HDF5 specs file to run MCMC
    File file(std::string(PATH_TO_SPEC)+"specs.h5", File::ReadOnly);

    std::regex suffix("specs?", std::regex_constants::icase);
    std::regex prefix("(\\.+|~)/(\\w+)/");
    std::regex ext("\\.h5", std::regex_constants::icase);
    auto id = std::regex_replace(file.getName(), suffix, "results");
    std::cout << id << std::endl;
    id = std::regex_replace(id, prefix, "");
    id = std::regex_replace(id, ext, "");
    std::cout << id << std::endl;

//    Matrix xinit;
//    file.getDataSet(std::string(SIMULATION_SPEC_KEY)+"/"+std::string("_provideData")).read(xinit);
//    std::cout << xinit << std::endl;
//
//    std::vector<double> sc;
//    file.getDataSet(std::string(SIMULATION_SPEC_KEY)+"/"+std::string("scaling")).read(sc);
//    std::cout << sc.size() << std::endl;
//
//    Vector mu;
//    file.getDataSet(std::string(MODEL_SPEC_KEY)+"/"+std::string(TRANSITIONM_KEY)+"/mu_prior").read(mu);
//    std:cout << mu << std::endl;
*/
    int s;
    typedef zip3<SingleStateScheme, EmbedHmmSchemeND>::withcrossprod<ParTrm_tlist, ParObsm_tlist, std::mt19937>::list List;
    std::cout << abi::__cxa_demangle(typeid(List).name(), 0, 0, &s) << std::endl;

    //! Some toy models for testing
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
    trm->init(A, Q);
    trm->setPrior(minit, Sinit);
/*    std::shared_ptr<LGObservationStationary> obsm = std::make_shared<LGObservationStationary>(10, 4, 2);
    obsm->init(C, R);

    DataGenerator<LGTransitionStationary, LGObservationStationary> simulator(trm, obsm, 1);
    std::cout << "Observations:\n" << simulator.getData() << std::endl;
*/
/*
    // Runtime Type checking
//    std::cout << "trm is LGTransitionStationary " << (typeid(*trm) == typeid(LGTransitionStationary)) << std::endl;
//    std::cout << "trm id is " << typeid(*trm).name() << std::endl;
//    std::cout << "LGtransitionStationary id is " << typeid(LGTransitionStationary).name() << std::endl;
    using Obsm = typelist::tlist<LGObservationStationary, LPObservationStationary, GPObservationStationary>;
    using Test_type = typename zip<SingleStateScheme, EmbedHmmSchemeND>::with<LGTransitionStationary, Obsm, std::mt19937>::list;
    auto sz = Test_type::size();
    typedef std::make_index_sequence<Test_type::size()> Idx;
    auto tpl = a2t_impl<Test_type>(Idx{});
    std::cout << tpl << std::endl;
*/
/*
    //! Kalman filter tests
    // Initialize Kalman filter with covariance scheme
    CovarianceScheme kf(4, 2);

    kf.initialize((*trm).getPriorMean(), (*trm).getPriorCov());

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
*/
/*
    //! Kalman smoother test
    algos::KalmanSmoother<InformationScheme, TwoFilterScheme> kalmsm(trm, obsm);
    kalmsm.initialize(simulator.getData());
    kalmsm.run();

    std::cout << "State means:\n" << kalmsm.post_means << std::endl;
    std::cout << "State covs:\n" << kalmsm.post_covs << std::endl;
    std::cout << "Smoothed state means:\n" << kalmsm.smoothed_means << std::endl;
    std::cout << "Smoothed state covs:\n" << kalmsm.smoothed_covs << std::endl;
*/

    //! Checking Poisson models
    size_t xdim = 4, ydim = 4, T=10;
    Matrix sigma = Vector::Constant(ydim, 0.6).asDiagonal();
    Matrix D = Matrix::Identity(ydim, ydim);
    Vector ctrls = Vector::Constant(ydim, -0.4);
    Vector x = Vector::Random(xdim);
    // Generalised Poisson
//    auto mf = [](const Ref<const Vector>& state, const Ref<const Vector>& coeff) -> Vector { return state.array().abs() * coeff.array(); };
//    std::shared_ptr<GPObservationStationary> gpoi = std::make_shared<GPObservationStationary>(T, ydim, mf);
//    gpoi->_provideData(sigma.diagonal());
    // Linear Poisson
    std::shared_ptr<LPObservationStationary> lpoi = std::make_shared<LPObservationStationary>(T, xdim, ydim, ydim);
    lpoi->init(sigma, D, ctrls);

//    std::cout << "State:\t" << x.transpose() << std::endl;
//    std::cout << "Mean:\t" << gpoi.getMean(x).transpose() << std::endl;
//
//    Vector_int y = gpoi.simulate(x, rng);
//
//    std::cout << "Simulated obs:\t" << y.transpose() << std::endl;
//    std::cout << "Logdensity: " << gpoi.logDensity(y, x) << std::endl;

    DataGenerator<LGTransitionStationary, LPObservationStationary> simulator_poi;
    simulator_poi.generate(trm, lpoi, 1);
//    std::cout << "Observations:\n" << simulator_poi.getData() << std::endl;

/*
    //! Single state Metropolis sampler
    using Sampler_type = SingleStateScheme<LGTransitionStationary, LGObservationStationary, std::mt19937>;
    std::shared_ptr<Sampler_type> sampler(make_shared<Sampler_type>());

    algos::MCMC<Sampler_type> ssmetropolis(trm, obsm, sampler, 1000, {0.2, 0.8});
    Matrix init_x(Matrix::Constant(4, 10, 0.));  // Initial sample
    ssmetropolis.initialize(simulator.getData());
    ssmetropolis.setData(init_x, 1);
    ssmetropolis.run();
    const SampleAccumulator& accumulator = ssmetropolis.getStatistics();

//    for (const auto& s: accumulator.getSamples()) {
//        std::cout << s << std::endl;
//    }

    std::cout << "number of samples:" << accumulator.getSamples().size() << std::endl;
    std::cout << "Acceptances:" << std::endl;
    for (auto& acc: accumulator.getAcceptances()) {
        std::cout << acc << "\t";
    }
    std::cout << "\nDuration: " << accumulator.totalDuration() << "ms" << std::endl;
    Vector means = accumulator.getSmoothedMeans(9);
    std::cout << "Mean at T\n" << means << std::endl;
    std::cout << "Cov at T\n" << accumulator.getSmoothedCov(means, 9) << std::endl;
    accumulator.save("test2", std::unordered_map<std::string, int>());
*/
/*
       //! Embedded HMM sampler
       using Sampler_type = EmbedHmmSchemeND<LGTransitionStationary, LPObservationStationary, std::mt19937>;
       std::shared_ptr<Sampler_type> sampler(std::make_shared<Sampler_type>(5, true));  // <-- 5 pool states

       algos::MCMC<Sampler_type> ehmm(trm, lpoi, sampler, 100, {0.1, 0.3}, 1, true);
       Matrix init_x(Matrix::Constant(4, 10, 0.));  // Initial sample
       ehmm.provideData(simulator_poi.getData(), int{});
       ehmm.reset(init_x, 1);
       ehmm.run();

       for (const auto& s: ehmm.getStatistics().getSamples()) {
           std::cout << s << std::endl;
       }

       std::cout << "Acceptances:" << std::endl;
       int i = 0;
       for (auto& acc: ehmm.getStatistics().getAcceptances()) {
           if (i == 10)
               std::cout << std::endl;
           std::cout << acc << "\t";
           ++i;
       }
       std::cout << "\nDuration: " << ehmm.getStatistics().totalDuration() << "ms" << std::endl;
*/
    DiagonalMatrixParam<NormalDist> parA(xdim);
    SymmetricMatrixParam<NormalDist> parQ(xdim);
    VectorParam<NormalDist> parctrl(xdim);
    DiagonalMatrixParam<NormalDist> parC(xdim);
    ConstMatrix parD(xdim);

    std::vector<double> Aprior{0., 0.25};
    std::vector<double> Qprior{1., 0.5};
    std::vector<double> Cprior{0.6, 0.7};
    std::vector<double> ctrl_prior{-0.4, 0.1};

    parA.setPrior(Aprior);
    parQ.setPrior(Qprior);
    parC.setPrior(Cprior);
    parctrl.setPrior(ctrl_prior);
/*
    double Adriver = A.initDraw(rng);
    double Cdriver = C.initDraw(rng);

    A.update(Adriver);
    C.update(Cdriver);

    std::cout << "A:\n" << A.param << std::endl;
    std::cout << "C:\n" << C.param << std::endl;

    double checkA = -0.3, checkC = 0.7;

    std::cout << "log density of driver for A = " << A.logDensity(checkA) << std::endl;
    std::cout << "log density of driver for C = " << C.logDensity(checkC) << std::endl;
*//*
    auto trm_params = std::make_tuple(parA, parQ);
    auto partrm = std::make_shared<ParametrizedModel<LGTransitionStationary,
                                                DiagonalMatrixParam<NormalDist>,
                                                SymmetricMatrixParam<NormalDist> > >(trm_params, T, xdim);
    partrm->setPrior(Vector::Zero(xdim), Matrix::Identity(xdim, xdim));

    auto obsm_params = std::make_tuple(parC, parD, parctrl);
    auto parobsm = std::make_shared<ParametrizedModel<LPObservationStationary,
                                DiagonalMatrixParam<NormalDist>,
                                ConstMatrix, VectorParam<NormalDist> > >(obsm_params, T, xdim, xdim, xdim);

    using Trm_type = ParametrizedModel<LGTransitionStationary,
                                    DiagonalMatrixParam<NormalDist>,
                                    SymmetricMatrixParam<NormalDist> >;
    using Obsm_type = ParametrizedModel<LPObservationStationary,
                                        DiagonalMatrixParam<NormalDist>,
                                        ConstMatrix, VectorParam<NormalDist> >;
    using PSampler_type = SingleStateScheme<Trm_type, Obsm_type, std::mt19937>;

    auto psampler(std::make_shared<WithParameterUpdate<PSampler_type> >(50));
    algos::MCMC<WithParameterUpdate<PSampler_type> > pehmm(partrm, parobsm, psampler, 100, {0.1, 0.3}, 1, false);
    Matrix init_x(Matrix::Constant(4, 10, 0.));  // Initial sample
    pehmm.provideData(simulator_poi.getData(), int{});
    pehmm.reset(init_x, 1);
    pehmm.run();

    for (const auto& s: pehmm.getStatistics().getParametersSamples()) {
        std::cout << s.transpose() << std::endl;
    }

    std::cout << "Parameters acceptances:" << std::endl;
    for (auto& acc: pehmm.getStatistics().getParametersAcceptances()) {
        std::cout << acc.first << ": " << acc.second << std::endl;
    }
    std::cout << "\nDuration: " << pehmm.getStatistics().totalDuration() << "ms" << std::endl;
*/
/*
    // testing Map<> object of Eigen
    int data[] = {1,2,3,4,5,6,7,8,9};
    Vector_int v = Eigen::Map<Vector_int>(data, 9);
    std::cout << typeid(v).name() << v << std::endl;
*/

    return 0;
}