//
// h5bridge.cpp
// Baysis
//
// Created by Vladimir Sotskov on 30/07/2021, 09:25.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include <regex>
#include "h5bridge.hpp"


MCMCsession::MCMCsession(const File &specs) {
    std::size_t obsmid, sid;
    Group simspecs = specs.getGroup(SIMULATION_SPEC_KEY);
    Group modelspecs = specs.getGroup(MODEL_SPEC_KEY);
    Group samplerspecs = specs.getGroup(SAMPLER_SPEC_KEY);
    simspecs.getDataSet("initialize").read(xinit);
    simspecs.getDataSet("seeds").read(seeds);
    modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("mtype").read(obsmid);
    samplerspecs.getAttribute("stype").read(sid);

    ObjectFactory<IMcmc, std::function<std::shared_ptr<IMcmc>(const Group& mspecs,
                                                              const Group& smplrspecs,
                                                              const Group& simspecs)> > factory{};
    factory.subscribe<MCMC, Sampler_tlist, McmcMaker>();
    std::size_t mcmcid = (sid - 1) * static_cast<std::size_t>(ModelType::size) + obsmid;
    mcmc = factory.create(mcmcid);
    create_id(specs.getName());
}

void MCMCsession::create_id(const std::string &fname) {
    std::regex suffix("specs?", std::regex_constants::icase);
    std::regex ext("\\.h5", std::regex_constants::icase);
    id = std::regex_replace(fname, suffix, "results");
    id = std::regex_replace(id, ext, "");
}


template<> LGTS::Model_ptr LGTS::create(const Group& modelspecs, std::size_t length) {
    Matrix A, Q, cov_prior;
    Vector mean_prior;
    std::size_t xdim = A.cols();
    modelspecs.getDataSet("A").read(A);
    modelspecs.getDataSet("Q").read(Q);
    modelspecs.getDataSet("mu_prior").read(mean_prior);
    modelspecs.getDataSet("S_prior").read(cov_prior);

    if (A.cols() != A.rows() || Q.cols() != Q.rows() || A.cols() != Q.cols() ||
    cov_prior.cols() != cov_prior.rows() || A.cols() != cov_prior.cols() ||
    mean_prior.rows() != A.cols())
        throw LogicException("Passed matrices for state model do not conform in size");

    std::shared_ptr<LGTransitionStationary> trm = std::make_shared<LGTransitionStationary>(length, xdim);
    trm->init(A,Q);
    trm->setPrior(mean_prior, cov_prior);
    return trm;
}

template<> LGOS::Model_ptr LGOS::create(const Group& modelspecs, std::size_t length) {
    Matrix C, R;
    std::size_t xdim = C.cols(), ydim = C.rows();
    modelspecs.getDataSet("C").read(C);
    modelspecs.getDataSet("R").read(R);

    if (R.cols() != R.rows() || R.rows() != C.rows())
        throw LogicException("Passed matrices for observation model do not conform in size");

    std::shared_ptr<LGObservationStationary> obsm = std::make_shared<LGObservationStationary>(length, xdim, ydim);
    obsm->init(C, R);
    return obsm;
}

template<> LPOS::Model_ptr LPOS::create(const Group& modelspecs, std::size_t length) {
    Matrix C, D;
    Vector controls;
    std::size_t xdim = C.cols(), ydim = C.rows(), cdim = D.cols();
    modelspecs.getDataSet("C").read(C);
    modelspecs.getDataSet("D").read(D);
    modelspecs.getDataSet("controls").read(controls);

    if (C.rows() != D.rows() || D.cols() != controls.rows())
        throw LogicException("Passed matrices for observation model do not conform in size");

    std::shared_ptr<LPObservationStationary> obsm = std::make_shared<LPObservationStationary>(length, xdim, ydim, cdim);
    obsm->init(C, D, controls);
    return obsm;
}

template<> GPOS::Model_ptr GPOS::create(const Group& modelspecs, std::size_t length) {
    Vector C;
    MeanFunction mftype;
    std::size_t ydim = C.rows();
    modelspecs.getDataSet("C").read(C);
    modelspecs.getAttribute("mean_function").read(mftype);

    GPObservationStationary::MF mf;
    switch (mftype) {
        case MeanFunction::bimodal:
            mf = bimodal;
            break;
    }
    std::shared_ptr<GPObservationStationary> obsm = std::make_shared<GPObservationStationary>(length, ydim, mf);
    obsm->init(C);
    return obsm;
}

/*
typedef std::unordered_map<std::string, Matrix*> Smoother_result;

template<typename F, typename S>
Smoother_result get_smoothing_dist(const std::shared_ptr<LGTransitionStationary>& trm,
                                   const std::shared_ptr<LGObservationStationary>& obsm,
                                   const Matrix& data) {
    KalmanSmoother<F, S> ks(trm, obsm);
    ks.initialize(data);
    ks.run();
    return {{"sm_means", &ks.smoothed_means}, {"sm_covs", &ks.smoothed_covs}};
}


template<typename TrM, typename ObsM>
Matrix& generate_data(const TrM& trm, const ObsM& obsm, int seed=0) {
    DataGenerator<TrM, ObsM> dg{trm, obsm, seed};
    return dg.getData();
}
*/