//
// h5bridge.hpp
// Baysis
//
// Created by Vladimir Sotskov on 21/08/2021, 18:45.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_H5BRIDGE_HPP
#define BAYSIS_H5BRIDGE_HPP

#include <functional>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5Group.hpp>
#include "algorithms.hpp"
#include "filterschemes.hpp"
//#include "samplingschemes.hpp"
#include "dataprovider.hpp"
//#include "utilities.hpp"

using namespace algos;
using namespace schemes;
using namespace ssmodels;
using namespace HighFive;

constexpr char PATH_TO_SPEC[] = "../data/";
constexpr char PATH_TO_RESULTS[] = "../data/";
constexpr char MODEL_SPEC_KEY[] = "model";
constexpr char TRANSITIONM_KEY[] = "transition";
constexpr char OBSERVATIONM_KEY[] = "observation";
constexpr char DATA_KEY[] = "data";
constexpr char SAMPLER_SPEC_KEY[] = "sampler";
constexpr char SIMULATION_SPEC_KEY[] = "simulation";

enum class MeanFunction { bimodal=4 };


using Obsm_tlist = typelist::tlist<LGObservationStationary, LPObservationStationary, GPObservationStationary>;
using Sampler_tlist = typename zip3<SingleStateScheme, EmbedHmmSchemeND>::with<LGTransitionStationary, Obsm_tlist, std::mt19937>::list;
using Param_dists = typelist::tlist<NormalDist, UniformDist, InvGammaDist>;
using DiagM_tlist = typename zip1<DiagonalMatrixParam>::with<Param_dists>::list;
using SymM_tlist = typename zip1<SymmetricMatrixParam>::with<Param_dists>::list;
using Vec_tlist = typename zip1<VectorParam>::with<Param_dists>::list;
using ParTrm_tlist = crossproduct<ParametrizedModel, LGTransitionStationary, DiagM_tlist, SymM_tlist>::list;
using ParLGOS_tlist = crossproduct<ParametrizedModel, LGObservationStationary, DiagM_tlist, DiagM_tlist>::list;
using ParLPOS_tlist = crossproductplus<ParametrizedModel, LPObservationStationary, DiagM_tlist, DiagM_tlist, ConstVector>::list;
using ParGPOS_tlist = typename zip2<ParametrizedModel>::with<GPObservationStationary, Vec_tlist>::list;
using ParObsm_tlist = typelist::tlist_concat_lists<ParLGOS_tlist, ParLPOS_tlist, ParGPOS_tlist>::type;
using ParSampler_tlist = zip1<WithParameterUpdate>::
        with<zip3<SingleStateScheme, EmbedHmmSchemeND>::
                withcrossprod<ParTrm_tlist, ParObsm_tlist, std::mt19937>::list>::list;
using Filter_tlist = typelist::tlist<schemes::CovarianceScheme, schemes::InformationScheme>;
using Smoother_tlist = typelist::tlist<schemes::RtsScheme, schemes::TwoFilterScheme>;

inline Vector bimodal(const Ref<const Vector>& state, const Ref<const Vector>& coeff) {
    return state.array().abs() * coeff.array();
}
/*
std::ostream& operator<<(std::ostream& os, ModelType mtype)
{
    switch (mtype)
    {
        case ModelType::lingauss   : os << "gauss"; break;
        case ModelType::linpoiss   : os << "pois1"; break;
        case ModelType::genpoiss   : os << "pois2";  break;
        default                    : os.setstate(std::ios_base::failbit);
    }
    return os;
}


std::ostream& operator<<(std::ostream& os, SamplerType stype)
{
    switch (stype)
    {
        case SamplerType::metropolis   : os << "met"; break;
        case SamplerType::ehmm         : os << "ehmm"; break;
        default                        : os.setstate(std::ios_base::failbit);
    }
    return os;
}
*/

template<typename ValueType, typename DataType>
bool saveResults(File& resfile, const std::string& path_to_ds,
                 const DataType& ds, const std::unordered_map<std::string, int> &attrs={}) {
    try {
        DataSet data_set = resfile.createDataSet<ValueType>(path_to_ds, DataSpace::From(ds));
        data_set.write(ds);

        for (const auto& items: attrs) {
            Attribute a = data_set.createAttribute<int>(items.first, DataSpace::From(items.second));
            a.write(items.second);
        }
    } catch (DataSetException& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
    return true;
}

struct MCMCsession {
    explicit MCMCsession(const File& specs);
    std::shared_ptr<IMcmc> mcmc;
    std::vector<u_long> seeds;
    Matrix xinit;
    std::string id;

private:
    void create_id(const std::string &fname);
};

struct DataInitialiser {
    void initialise(const File &specs);
    void provideto(const MCMCsession& session);
    bool saveto(File &file);

    Matrix states;
    Matrix realdata;
    Matrix_int intdata;
};


struct SmootherSession {
    explicit SmootherSession(const File& specs);
    std::shared_ptr<ISmoother> kalman;
    std::string id;

private:
    void create_id(const std::string &fname);
};


template<typename M>
struct ModelMaker {
    using Model_ptr = std::shared_ptr<M>;
    template<typename... P>
    using PModel = ParametrizedModel<M, P...>;

    static Model_ptr create(const Group& modelspecs, std::size_t length);

    template<typename P>
    static std::shared_ptr<PModel<P> > create(const Group& modelspecs, std::shared_ptr<P> param);

    template<typename... P>
    static std::shared_ptr<PModel<P...> > create(const Group& modelspecs, std::shared_ptr<P>... params);
};

using LGTS = ModelMaker<LGTransitionStationary>;
using LGOS = ModelMaker<LGObservationStationary>;
using LPOS = ModelMaker<LPObservationStationary>;
using GPOS = ModelMaker<GPObservationStationary>;

template<typename M>
template<typename P>
std::shared_ptr<typename ModelMaker<M>::template PModel<P> >
ModelMaker<M>::create(const Group &modelspecs, std::shared_ptr<P> param) {
    std::size_t length, ydim;
    int mtype;

    modelspecs.getAttribute("length").read(length);
    modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("mtype").read(mtype);
    modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("ydim").read(ydim);

    if (ModelType(mtype%10) == ModelType::genpoiss) {
        int mftype;
        modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("mean_function").read(mftype);
        GPObservationStationary::MF mf;
        switch (MeanFunction(mftype)) {
            case MeanFunction::bimodal:
                mf = bimodal;
                break;
        }
        return std::make_shared<PModel<P> >(std::forward<std::tuple<std::shared_ptr<P> > >
                (std::make_tuple(param)), length, ydim, mf);
    }
    return nullptr;
}

template<typename M>
template<typename... P>
std::shared_ptr<typename ModelMaker<M>::template PModel<P...> >
ModelMaker<M>::create(const Group& modelspecs, std::shared_ptr<P>... params) {
    std::size_t length;
    int mtype, xdim, ydim, cdim{0};
    modelspecs.getAttribute("length").read(length);
    modelspecs.getGroup(TRANSITIONM_KEY).getAttribute("xdim").read(xdim);
    modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("mtype").read(mtype);
    modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("ydim").read(ydim);
    if (modelspecs.getGroup(OBSERVATIONM_KEY).hasAttribute("cdim"))
        modelspecs.getGroup(OBSERVATIONM_KEY).getAttribute("cdim").read(cdim);

    return std::make_shared<PModel<P...> >(std::forward<std::tuple<std::shared_ptr<P>... > >
            (std::make_tuple(params...)), length, xdim, ydim, cdim);
}

template<>
template<typename... P>
std::shared_ptr<typename ModelMaker<LGTransitionStationary>::template PModel<P...> >
ModelMaker<LGTransitionStationary>::create(const Group& modelspecs, std::shared_ptr<P>... params) {
    std::size_t length, xdim;
    Vector mean_prior;
    Matrix cov_prior;
    modelspecs.getAttribute("length").read(length);
    modelspecs.getGroup(TRANSITIONM_KEY).getAttribute("xdim").read(xdim);
    modelspecs.getGroup(TRANSITIONM_KEY).getDataSet("mu_prior").read(mean_prior);
    modelspecs.getGroup(TRANSITIONM_KEY).getDataSet("S_prior").read(cov_prior);
    auto trm = std::make_shared<PModel<P...> >(std::forward<std::tuple<std::shared_ptr<P>...> >
            (std::make_tuple(params...)), length, xdim);
    trm->setPrior(mean_prior, cov_prior);
    return trm;
}


struct ParamMaker: CreatorWrapper<IParam, const std::vector<double>&, std::size_t> {
    template<typename Derived>
    static std::shared_ptr<IParam> create(const std::vector<double>& settings, std::size_t dim);

    static std::shared_ptr<IParam> createConst(const Group& spec);
};


struct SamplerMaker: CreatorWrapper<ISampler, const Group&> {
    template<typename Derived>
    static std::shared_ptr<ISampler> create(const Group &samplerspecs);
};

struct ParmSamplerMaker {
    template<typename Derived>
    static std::shared_ptr<ISampler> create(const Group &samplerspecs);
};

struct McmcMaker: CreatorWrapper<IMcmc, const Group&, const Group&, const Group&> {
    template<typename Derived>
    static std::shared_ptr<IMcmc> create(const Group& mspecs,
                                         const Group& smplrspecs,
                                         const Group& simspecs);
};


struct ParmMcmcMaker: CreatorWrapper<IMcmc, const Group&, const Group&, const Group&> {
    template<typename Derived>
    static std::shared_ptr<IMcmc> create(const Group& mspecs,
                                         const Group& smplrspecs,
                                         const Group& simspecs);
};


struct SmootherMaker: CreatorWrapper<ISmoother, const Group&> {
    template<typename Derived>
    static std::shared_ptr<ISmoother> create(const Group &modelspecs);
};


template<typename Derived>
std::shared_ptr<IParam> ParamMaker::create(const std::vector<double>& settings, std::size_t dim) {
    auto retval = std::make_shared<Derived>(dim);
    auto it = settings.begin();
    // First two values are always support
    auto support = std::make_pair(*it, *++it);
    if (!(support.first == 0 && support.second == 0))
        retval->setSupport(support);
    // Next one is the variance scalar
    retval->setVarscale(*++it);
    if (++it >= settings.end()) {
        auto msg = std::string("Prior parameters are not supplied for " + Derived::name());
        throw LogicException(msg.data());
    }
    retval->setPrior(std::vector<double>(it, settings.end()));
    return retval;
}


template<typename Derived>
std::shared_ptr<ISampler> SamplerMaker::create(const Group &samplerspecs) {
    int smplrid;
    samplerspecs.getAttribute("stype").read(smplrid);

    switch (SamplerType(smplrid)) {
        case SamplerType::metropolis:
            // No variables needed for initialisation
            return std::make_shared<Derived>();
        case SamplerType::ehmm:
            if (samplerspecs.hasAttribute("pool_size")) {
                std::size_t psize;
                samplerspecs.getAttribute("pool_size").read(psize);

                if (samplerspecs.hasAttribute("flip")) {
                    // We have both values for initialization
                    bool flip;
                    samplerspecs.getAttribute("flip").read(flip);
                    return std::make_shared<Derived>(psize, flip);
                }
                // Flip variable missing, will use default
                return std::make_shared<Derived>(psize);
            }
            // Not enough variables to initialise
            throw LogicException("The specification for Embdedded HMM has to provide at least pool_size variable.");
    }
}


template<typename Derived>
std::shared_ptr<ISampler> ParmSamplerMaker::create(const Group &samplerspecs) {
    int smplrid, npu;
    samplerspecs.getAttribute("stype").read(smplrid);
    samplerspecs.getAttribute("num_param_updates").read(npu);

    switch (SamplerType(smplrid)) {
        case SamplerType::metropolis:
            // No variables needed for initialisation
            return std::make_shared<Derived>(npu);
            case SamplerType::ehmm:
                if (samplerspecs.hasAttribute("pool_size")) {
                    std::size_t psize;
                    samplerspecs.getAttribute("pool_size").read(psize);

                    if (samplerspecs.hasAttribute("flip")) {
                        // We have both values for initialization
                        bool flip;
                        samplerspecs.getAttribute("flip").read(flip);
                        return std::make_shared<Derived>(npu, psize, flip);
                    }
                    // Flip variable missing, will use default
                    return std::make_shared<Derived>(npu, psize);
                }
                // Not enough variables to initialise
                throw LogicException("The specification for Embdedded HMM has to provide at least pool_size variable.");
    }
}


template<typename Derived>
std::shared_ptr<IMcmc> McmcMaker::create(const Group& mspecs,
                                         const Group& smplrspecs,
                                         const Group& simspecs) {
    Group trmspec = mspecs.getGroup(TRANSITIONM_KEY);
    Group obsmspec = mspecs.getGroup(OBSERVATIONM_KEY);
    int length, obsmid, smplrid, numiter, thin=1;
    bool rev=false;
    std::vector<double> scaling(1, 1.);

    mspecs.getAttribute("length").read(length);
    obsmspec.getAttribute("mtype").read(obsmid);
    smplrspecs.getAttribute("stype").read(smplrid);
    simspecs.getAttribute("numiter").read(numiter);
    if (simspecs.exist("scaling")) simspecs.getDataSet("scaling").read(scaling);
    if (simspecs.hasAttribute("thin")) simspecs.getAttribute("thin").read(thin);
    if (simspecs.hasAttribute("reverse")) simspecs.getAttribute("reverse").read(rev);

    auto trm = LGTS::create(trmspec, length);

    auto sampler = std::dynamic_pointer_cast<typename Derived::Scheme_type>
        (SamplerMaker::create<typename Derived::Scheme_type>(smplrspecs));

    switch (ModelType(obsmid)) {
        case ModelType::lingauss:
        {
            auto obsm = LGOS::create(obsmspec, length);
            return std::make_shared<Derived>(trm, obsm, sampler, numiter, scaling, thin, rev);
        }
        case ModelType::linpoiss:
        {
            auto obsm = LPOS::create(obsmspec, length);
            return std::make_shared<Derived>(trm, obsm, sampler, numiter, scaling, thin, rev);
        }
        case ModelType::genpoiss:
        {
            auto obsm = GPOS::create(obsmspec, length);
            return std::make_shared<Derived>(trm, obsm, sampler, numiter, scaling, thin, rev);
        }
        default: return nullptr;
    }
}

template<typename Derived>
std::shared_ptr<IMcmc> ParmMcmcMaker::create(const Group &mspecs, const Group &smplrspecs, const Group &simspecs) {
    using namespace std::placeholders;

    Group trmspec = mspecs.getGroup(TRANSITIONM_KEY);
    Group obsmspec = mspecs.getGroup(OBSERVATIONM_KEY);
    int length, obsmid, smplrid, numiter, thin=1;
    bool rev=false;
    std::vector<double> scaling(1, 1.);

    mspecs.getAttribute("length").read(length);
    obsmspec.getAttribute("mtype").read(obsmid);
    smplrspecs.getAttribute("stype").read(smplrid);
    simspecs.getAttribute("numiter").read(numiter);
    if (simspecs.exist("scaling")) simspecs.getDataSet("scaling").read(scaling);
    if (simspecs.hasAttribute("thin")) simspecs.getAttribute("thin").read(thin);
    if (simspecs.hasAttribute("reverse")) simspecs.getAttribute("reverse").read(rev);

    // Transition model creation
    std::vector<double> A, Q;
    std::size_t xdim, A_id, Q_id;
    trmspec.getDataSet("A").read(A);
    trmspec.getDataSet("Q").read(Q);
    trmspec.getAttribute("xdim").read(xdim);
    A_id = static_cast<std::size_t>(A.front());
    ObjectFactory<IParam, std::function<std::shared_ptr<IParam>(const std::vector<double>&, std::size_t)> > param_factory{};
    param_factory.template subscribe<DiagonalMatrixParam, Param_dists, ParamMaker>();
    auto param1 = param_factory.create(A_id, std::vector<double>(++A.begin(), A.end()), xdim);
    param_factory.template subscribe<SymmetricMatrixParam, Param_dists, ParamMaker>();
    Q_id = static_cast<std::size_t>(Q.front());
    double diag = Q.back();
    auto param2 = param_factory.create(Q_id, std::vector<double>(++Q.begin(), --Q.end()), xdim);
    param2->initDiagonal(diag);
    auto trm = ModelMaker<LGTransitionStationary>::create(mspecs, param1, param2);

    // Sampler creation
    auto sampler = std::dynamic_pointer_cast<typename Derived::Scheme_type>
            (ParmSamplerMaker::create<typename Derived::Scheme_type>(smplrspecs));

    // Observation model creation
    int model_id = obsmid % 10;
    std::size_t ydim;
    obsmspec.getAttribute("ydim").read(ydim);

    switch (ModelType(model_id)) {
        case ModelType::lingauss:
        {
            std::vector<double> C, R;
            std::size_t C_id, R_id;
            obsmspec.getDataSet("C").read(C);
            obsmspec.getDataSet("R").read(R);
            param_factory.template subscribe<DiagonalMatrixParam, Param_dists, ParamMaker>();
            C_id = static_cast<std::size_t>(C.front());
            R_id = static_cast<std::size_t>(R.front());
            auto obsm_param1 = param_factory.create(C_id, std::vector<double>(++C.begin(), C.end()), ydim);
            auto obsm_param2 = param_factory.create(R_id, std::vector<double>(++R.begin(), R.end()), ydim);
            auto obsm = ModelMaker<LGObservationStationary>::create(mspecs, obsm_param1, obsm_param2);
            return std::make_shared<Derived>(trm, obsm, sampler, numiter, scaling, thin, rev);
        }
        case ModelType::linpoiss:
        {
            std::vector<double> C, D;
            std::size_t C_id, D_id;
            obsmspec.getDataSet("C").read(C);
            obsmspec.getDataSet("D").read(D);
            param_factory.template subscribe<DiagonalMatrixParam, Param_dists, ParamMaker>();
            C_id = static_cast<std::size_t>(C.front());
            D_id = static_cast<std::size_t>(D.front());
            auto obsm_param1 = param_factory.create(C_id, std::vector<double>(++C.begin(), C.end()), ydim);
            auto obsm_param2 = param_factory.create(D_id, std::vector<double>(++D.begin(), D.end()), ydim);
            auto obsm_param3 = ParamMaker::createConst(obsmspec);
            auto obsm = ModelMaker<LPObservationStationary>::create(mspecs, obsm_param1, obsm_param2, obsm_param3);
            return std::make_shared<Derived>(trm, obsm, sampler, numiter, scaling, thin, rev);
        }
        case ModelType::genpoiss:
        {
            std::vector<double> C;
            std:size_t C_id;
            obsmspec.getDataSet("C").read(C);
            param_factory.template subscribe<VectorParam, Param_dists, ParamMaker>();
            C_id = static_cast<std::size_t>(C.front());
            auto obsm_param = param_factory.create(C_id, std::vector<double>(++C.begin(), C.end()), ydim);
            auto obsm = ModelMaker<GPObservationStationary>::create(mspecs, obsm_param);
            return std::make_shared<Derived>(trm, obsm, sampler, numiter, scaling, thin, rev);
        }
        default: return nullptr;
    }
}


template<typename Derived>
std::shared_ptr<ISmoother> SmootherMaker::create(const Group &modelspecs) {
    std::size_t length;
    modelspecs.getAttribute("length").read(length);
    auto trm = LGTS::create(modelspecs.getGroup("transition"), length);
    auto obsm = LGOS::create(modelspecs.getGroup("observation"), length);
    return std::make_shared<Derived>(trm, obsm);
}



#endif //BAYSIS_H5BRIDGE_HPP
