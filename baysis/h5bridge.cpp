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
    std::size_t trmid, obsmid, mcmc_id;
    bool is_parametrised{false};
    Group simspecs = specs.getGroup(SIMULATION_SPEC_KEY);
    Group modelspecs = specs.getGroup(MODEL_SPEC_KEY);
    Group samplerspecs = specs.getGroup(SAMPLER_SPEC_KEY);
    simspecs.getDataSet("init").read(xinit);
    simspecs.getDataSet("seeds").read(seeds);
    modelspecs.getGroup(TRANSITIONM_KEY).getAttribute("mtype").read(trmid);
    modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("mtype").read(obsmid);
    specs.getAttribute("mcmc_id").read(mcmc_id);

    if (specs.hasAttribute("parametrised"))
        specs.getAttribute("parametrised").read(is_parametrised);
    ObjectFactory<IMcmc, std::function<std::shared_ptr<IMcmc>(const Group&, const Group&, const Group&)> > factory{};
    if (is_parametrised) {
        if (!factory.subscribe<MCMC, ParSampler_tlist, ParmMcmcMaker>())
            throw LogicException("MCMC types have non-unique ids.");
    } else {
        if (!factory.subscribe<MCMC, Sampler_tlist, McmcMaker>())
            throw LogicException("MCMC types have non-unique ids.");
    }
    factory.print();  // only prints in DEBUG mode
    mcmc = factory.create(mcmc_id, modelspecs, samplerspecs, simspecs);
    create_id(specs.getName());
}

void MCMCsession::create_id(const std::string &fname) {
    std::regex suffix("_?specs?", std::regex_constants::icase);
    std::regex prefix("(\\.+|~)/(\\w+)/");
    std::regex ext("\\.h5", std::regex_constants::icase);
    id = std::regex_replace(fname, suffix, "");
    id = std::regex_replace(id, prefix, "");
    id = std::regex_replace(id, ext, "");
}


SmootherSession::SmootherSession(const File &specs) {
    ObjectFactory<ISmoother, std::function<std::shared_ptr<ISmoother>(const Group&)> > factory;
    factory.subscribe<KalmanSmoother, Filter_tlist, Smoother_tlist, SmootherMaker>();
    std::size_t fid, sid;
    specs.getGroup("smoother").getAttribute("ftype").read(fid);
    specs.getGroup("smoother").getAttribute("stype").read(sid);
    kalman = factory.create(fid*FILTER_ID_MULT+sid, specs.getGroup("model"));
}


template<> LGTS::Model_ptr LGTS::create(const Group& modelspecs, std::size_t length) {
    Matrix A, Q, cov_prior;
    Vector mean_prior;
    modelspecs.getDataSet("A").read(A);
    modelspecs.getDataSet("Q").read(Q);
    modelspecs.getDataSet("mu_prior").read(mean_prior);
    modelspecs.getDataSet("S_prior").read(cov_prior);
    std::size_t xdim = A.cols();

    if (A.cols() != A.rows() || A.cols() != Q.cols() ||
        A.cols() != cov_prior.cols() ||
        mean_prior.rows() != A.cols())
        throw LogicException("Passed matrices for state model do not conform in size");

    std::shared_ptr<LGTransitionStationary> trm = std::make_shared<LGTransitionStationary>(length, xdim);
    trm->init(A,Q);
    trm->setPrior(mean_prior, cov_prior);
    return trm;
}

template<> LGOS::Model_ptr LGOS::create(const Group& modelspecs, std::size_t length) {
    Matrix C, R;
    modelspecs.getDataSet("C").read(C);
    modelspecs.getDataSet("R").read(R);
    std::size_t xdim = C.cols(), ydim = C.rows();

    if (R.cols() != R.rows() || R.rows() != C.rows())
        throw LogicException("Passed matrices for observation model do not conform in size");

    std::shared_ptr<LGObservationStationary> obsm = std::make_shared<LGObservationStationary>(length, xdim, ydim);
    obsm->init(C, R);
    return obsm;
}

template<> LPOS::Model_ptr LPOS::create(const Group& modelspecs, std::size_t length) {
    Matrix C, D;
    Vector controls;
    modelspecs.getDataSet("C").read(C);
    modelspecs.getDataSet("D").read(D);
    modelspecs.getDataSet("controls").read(controls);
    std::size_t xdim = C.cols(), ydim = C.rows(), cdim = D.cols();

    if (C.rows() != D.rows() || D.cols() != controls.rows())
        throw LogicException("Passed matrices for observation model do not conform in size");

    std::shared_ptr<LPObservationStationary> obsm = std::make_shared<LPObservationStationary>(length, xdim, ydim, cdim);
    obsm->init(C, D, controls);
    return obsm;
}


template<> GPOS::Model_ptr GPOS::create(const Group& modelspecs, std::size_t length) {
    Vector C;
    int mftype;
    std::size_t ydim = C.rows();
    modelspecs.getDataSet("C").read(C);
    modelspecs.getAttribute("mean_function").read(mftype);

    GPObservationStationary::MF mf;
    switch (MeanFunction(mftype)) {
        case MeanFunction::bimodal:
            mf = bimodal;
            break;
    }
    std::shared_ptr<GPObservationStationary> obsm = std::make_shared<GPObservationStationary>(length, ydim, mf);
    obsm->init(C);
    return obsm;
}


std::shared_ptr<IParam> ParamMaker::createConst(const Group &spec) {
    std::size_t ydim;
    std::vector<double> const_spec;
    spec.getAttribute("ydim").read(ydim);
    spec.getDataSet("constant").read(const_spec);
    auto pt = static_cast<ParamType>(static_cast<int>(const_spec.front()));

    switch (pt) {
        case ParamType::constm:
            return std::make_shared<ConstMatrix>(ydim, const_spec.back());
        case ParamType::constv:
            return std::make_shared<ConstVector>(ydim, const_spec.back());
        default:
            throw LogicException("Constant parameter can only be a const matrix or a const vector");
    }
}


void DataInitialiser::initialise(const File &specs, const MCMCsession &session) {
    Group data = specs.getGroup(DATA_KEY);
    int mtype;
    specs.getGroup(std::string(MODEL_SPEC_KEY)+"/"+std::string(OBSERVATIONM_KEY)).getAttribute("mtype").read(mtype);

    if (data.exist("observations")) {
        std::string dtype;
        data.getAttribute("dtype").read(dtype);
        if (dtype[0] == 'i') {
            specs.getDataSet(std::string(DATA_KEY)+"/observations").read(intdata);
        } else {
            specs.getDataSet(std::string(DATA_KEY)+"/observations").read(realdata);
        }
    } else {
        // No data provided, need to generate
        u_long seed;
        data.getAttribute("seed").read(seed);

        switch (ModelType(mtype)) {
            case ModelType::lingauss:
            {
                DataGenerator<LGTransitionStationary, LGObservationStationary> dg;
                dg.generate(session.mcmc->getTransitionModel(), session.mcmc->getObservationModel(), seed);
                realdata = dg.getData();
            }
                break;
            case ModelType::linpoiss:
            {
                DataGenerator<LGTransitionStationary, LPObservationStationary> dg;
                dg.generate(session.mcmc->getTransitionModel(), session.mcmc->getObservationModel(), seed);
                intdata = dg.getData();
            }
                break;
            case ModelType::genpoiss:
            {
                DataGenerator<LGTransitionStationary, GPObservationStationary> dg;
                dg.generate(session.mcmc->getTransitionModel(), session.mcmc->getObservationModel(), seed);
                intdata = dg.getData();
            }
                break;
            default: throw LogicException("Unknown observation model type");
        }
    }
}

void DataInitialiser::provideto(const MCMCsession& session) {
    if (intdata.size() == 0 && realdata.size() == 0) {
        throw LogicException("Data for the sampler is unavailable. Generate inplace or provideto observations.");
    } else if (intdata.size() !=0 ) {
        session.mcmc->provideData(intdata, int{});
    } else {
        session.mcmc->provideData(realdata, double{});
    }
}

bool DataInitialiser::saveto(File &file) {
    if (intdata.size() != 0) {
        return saveResults<int>(file, "observations", intdata);
    } else {
        return saveResults<double>(file, "observations", realdata);
    }
}
