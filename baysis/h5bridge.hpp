//
// h5bridge.hpp
// Baysis
//
// Created by Vladimir Sotskov on 21/08/2021, 18:45.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_H5BRIDGE_HPP
#define BAYSIS_H5BRIDGE_HPP

#include "../extern/highfive/H5Group.hpp"
#include "algorithms.hpp"
#include "filterschemes.hpp"
#include "samplingschemes.hpp"
#include "dataprovider.hpp"
#include "ifactories.hpp"

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

enum class ModelType { lingauss=1, linpoiss, genpoiss, size=genpoiss };
enum class SamplerType { metropolis=1, ehmm };
enum class MeanFunction { bimodal=4 };

// !! The order in the observation models typelist must be the same as the cases in the ModelType enum above
using Obsm_tlist = typelist::tlist<LGObservationStationary, LPObservationStationary, GPObservationStationary>;
using Sampler_tlist = typename zip<SingleStateScheme, EmbedHmmSchemeND>::with<LGTransitionStationary, Obsm_tlist, std::mt19937>::list;
using Filter_tlist = typelist::tlist<schemes::CovarianceScheme, schemes::InformationScheme>;
using Smoother_tlist = typelist::tlist<schemes::RtsScheme, schemes::TwoFilterScheme>;

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
    void initialise(const File& specs, const MCMCsession& session);
    void provideto(const MCMCsession& session);

    Matrix realdata;
    Matrix_int intdata;
};


struct SmootherSession {
    explicit SmootherSession(const File& specs);
    std::shared_ptr<ISmoother> kalman;
};


template<typename M>
struct ModelMaker {
    typedef std::shared_ptr<M> Model_ptr;
    static Model_ptr create(const Group& modelspecs, std::size_t length);
};

using LGTS = ModelMaker<LGTransitionStationary>;
using LGOS = ModelMaker<LGObservationStationary>;
using LPOS = ModelMaker<LPObservationStationary>;
using GPOS = ModelMaker<GPObservationStationary>;


struct SamplerMaker: CreatorWrapper<ISampler, const Group&> {
    template<typename Derived>
    static std::shared_ptr<ISampler> create(const Group &samplerspecs);
};


struct McmcMaker: CreatorWrapper<IMcmc, const Group&, const Group&, const Group&> {
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
                Attribute pszattr = samplerspecs.getAttribute("pool_size");
                pszattr.read(psize);

                if (samplerspecs.hasAttribute("flip")) {
                    Attribute flipattr = samplerspecs.getAttribute("flip");
                    bool flip;
                    // We have both values for initialization
                    flipattr.read(flip);
                    return std::make_shared<Derived>(psize, flip);
                }
                // Flip variable missing, will use default
                return std::make_shared<Derived>(psize);
            }
            // Not enough variables to initialise
            throw LogicException("The specification for Embdedded HMM has to provideto at least pool_size variable.");
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

    ObjectFactory<ISampler, std::function<std::shared_ptr<ISampler>(const Group&)> > sampler_factory{};
    switch (SamplerType(smplrid)) {
        case SamplerType::metropolis:
            sampler_factory.subscribe<SingleStateScheme, LGTransitionStationary,
                                      typelist::tlist_reverse<Obsm_tlist>::type, std::mt19937, SamplerMaker>();
            break;
        case SamplerType::ehmm:
            sampler_factory.subscribe<EmbedHmmSchemeND, LGTransitionStationary,
                                      typelist::tlist_reverse<Obsm_tlist>::type, std::mt19937, SamplerMaker>();
    }

    auto sampler = std::dynamic_pointer_cast<typename Derived::Scheme_type>(sampler_factory.create(obsmid, smplrspecs));

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
std::shared_ptr<ISmoother> SmootherMaker::create(const Group &modelspecs) {
    std::size_t length;
    modelspecs.getAttribute("length").read(length);
    auto trm = LGTS::create(modelspecs.getGroup("transition"), length);
    auto obsm = LGOS::create(modelspecs.getGroup("observation"), length);
    return std::make_shared<Derived>(trm, obsm);
}


inline Vector bimodal(const Ref<const Vector>& state, const Ref<const Vector>& coeff) {
    return state.array().abs() * coeff.array();
}



#endif //BAYSIS_H5BRIDGE_HPP
