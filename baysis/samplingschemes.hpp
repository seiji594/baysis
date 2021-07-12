//
// samplingschemes.hpp
// Baysis
//
// Created by Vladimir Sotskov on 06/07/2021, 20:34.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_SAMPLINGSCHEMES_HPP
#define BAYSIS_SAMPLINGSCHEMES_HPP

#include "models.hpp"

using namespace ssmodels;


namespace schemes {

    template<typename RNG>
    struct AutoregressiveUpdate {
        template<typename DerivedA, typename DerivedB>
        void propose(const Eigen::MatrixBase<DerivedA>& x,
                     const Eigen::MatrixBase<DerivedB> &mu,
                     const Eigen::LLT<Matrix> &L);

        template<typename DerivedA, typename DerivedB>
        bool accept(const LGObservationStationary& obsm,
                    const Eigen::MatrixBase<DerivedA> &y,
                    const Eigen::MatrixBase<DerivedB> &x);

        Vector proposal;            // temp for proposals
        RandomSample<RNG, std::uniform_real_distribution<> > uniform;
        RandomSample<RNG, std::normal_distribution<> > norm;
        double eps{};
    };


    template<typename RNG>
    class SingleStateScheme {
    public:
        typedef RNG Rng_type;

        void init(const Matrix &observations, const TransitionModel &tr_model, u_long seed);
        void sample(const TransitionModel &tr_model, const ObservationModel &obs_model);
        void setScaling(const std::vector<double>& scalings, int i) { ar_update.eps = scalings[i % scalings.size()]; }

        Matrix cur_sample;
        Vector_int acceptances;
    private:
        AutoregressiveUpdate<RNG> ar_update;
        Matrix data;
        Matrix post_mean_init;
        Matrix post_mean;
        Eigen::LLT<Matrix> post_L_init;
        Eigen::LLT<Matrix> post_L;
    };

    template<typename RNG>
    template<typename DerivedA, typename DerivedB>
    void AutoregressiveUpdate<RNG>::propose(const Eigen::MatrixBase<DerivedA> &x,
                                            const Eigen::MatrixBase<DerivedB> &mu,
                                            const Eigen::LLT<Matrix> &L) {
        proposal = NormalDist::sample(mu + sqrt(1 - eps * eps) * (x - mu), L, norm, eps);
    }


    template<typename RNG>
    template<typename DerivedA, typename DerivedB>
    bool AutoregressiveUpdate<RNG>::accept(const LGObservationStationary &obsm,
                                           const Eigen::MatrixBase<DerivedA> &y,
                                           const Eigen::MatrixBase<DerivedB> &x) {
        double alpha = uniform.draw();
        double cur_ld = obsm.logDensity(y, x);
        double prop_ld = obsm.logDensity(y, proposal);
        return exp(prop_ld - cur_ld) > alpha;
    }


    template<typename RNG>
    void SingleStateScheme<RNG>::init(const Matrix &observations, const TransitionModel &tr_model, u_long seed) {
        // Resize containers
//        cur_sample.resize(tr_model.stateDim(), tr_model.length());
        acceptances.resize(tr_model.length());
//        post_mean_init.resize(tr_model.stateDim(), tr_model.stateDim());
//        post_mean.resize(tr_model.stateDim(), tr_model.stateDim());
//        data.resizeLike(observations);

        // Initialise
        const LGTransitionStationary& lg_transition = static_cast<const LGTransitionStationary&>(tr_model);
        data = observations;
//        cur_sample = lg_transition.getPriorMean();
        post_mean_init = (lg_transition.getA() * lg_transition.getA()
                           + lg_transition.getPriorCovInv() * lg_transition.getCov()).inverse() * lg_transition.getA();
        post_L_init.compute((lg_transition.getA() * lg_transition.getCovInv() * lg_transition.getA()
                             + lg_transition.getPriorCovInv()).inverse());
        post_mean = (lg_transition.getA() * lg_transition.getA()
                      + Matrix::Identity(lg_transition.stateDim(), lg_transition.stateDim())).inverse()
                     * lg_transition.getA();
        post_L.compute((lg_transition.getA() * lg_transition.getCovInv() * lg_transition.getA()
                        + lg_transition.getPriorCovInv()).inverse());

        // Setup the update scheme
        ar_update.proposal.resize(tr_model.stateDim());
        std::shared_ptr<RNG> rng;
        if (seed != 0) {
            rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator(seed));
        } else {
            rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator());
        }
        ar_update.uniform = RandomSample<RNG, std::uniform_real_distribution<> >(rng);
        ar_update.norm = RandomSample<RNG, std::normal_distribution<> >(rng);
    }

    template<typename RNG>
    void SingleStateScheme<RNG>::sample(const TransitionModel &tr_model,
                                        const ObservationModel &obs_model) {
        std::size_t T{tr_model.length()-1};

        ar_update.template propose(cur_sample.col(0),
                post_mean_init * (cur_sample.col(1) + static_cast<const LGTransitionStationary&>(tr_model).getPriorMean()),
                post_L_init);
        if (ar_update.template accept(static_cast<const LGObservationStationary&>(obs_model),
                data.col(0), cur_sample.col(0))) {
            cur_sample.col(0) = ar_update.proposal;
            ++acceptances(0);
        }

        for (int t = 1; t < T; ++t) {
            ar_update.template propose(
                    cur_sample.col(t),
                    post_mean * (cur_sample.col(t-1) + cur_sample.col(t+1)),
                    post_L);
            if (ar_update.template accept(static_cast<const LGObservationStationary&>(obs_model),
                    data.col(t), cur_sample.col(t))) {
                cur_sample.col(t) = ar_update.proposal;
                ++acceptances(t);
            }
        }

        ar_update.template propose(cur_sample.col(T),
                static_cast<const LGTransitionStationary&>(tr_model).getMean(cur_sample.col(T-1)),
                static_cast<const LGTransitionStationary&>(tr_model).getL());
        if (ar_update.template accept(static_cast<const LGObservationStationary&>(obs_model),
                data.col(T), cur_sample.col(T))) {
            cur_sample.col(T) = ar_update.proposal;
            ++acceptances(T);
        }

    }


}

#endif //BAYSIS_SAMPLINGSCHEMES_HPP
