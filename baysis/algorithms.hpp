//
// algorithms.hpp
// Baysis
//
// Created by Vladimir Sotskov on 07/07/2021, 16:43.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_ALGORITHMS_HPP
#define BAYSIS_ALGORITHMS_HPP

#include <vector>
#include "models.hpp"
#include "accumulator.hpp"

using ssmodels::TransitionModel;
using ssmodels::ObservationModel;


namespace algos {

    template<typename Scheme>
    class MCMC {
    public:
        typedef typename Scheme::Data_type Data_type;
        typedef typename Scheme::Sample_type Sample_type;

        MCMC(const std::shared_ptr<TransitionModel> &tr_model, const std::shared_ptr<ObservationModel> &obs_model,
             const std::shared_ptr<Scheme> &sampling_scheme, int N,
             const std::vector<double> &scalings = std::vector<double>(1, 1.), int thinning_factor = 1,
             bool reverse = false);

        void initialise(const Data_type& observations, const Sample_type& x_init, u_long seed = 0);
        void run();

        SampleAccumulator<typename Sample_type::Scalar> accumulator;
    private:
        const std::shared_ptr<TransitionModel> transitionM;
        const std::shared_ptr<ObservationModel> observationM;
        std::shared_ptr<Scheme> sampler;
        int numiter;
        int thin;
        bool run_reversed;
        std::vector<double> scaling;
    };


    template<typename Scheme>
    MCMC<Scheme>::MCMC(const std::shared_ptr<TransitionModel> &tr_model,
                       const std::shared_ptr<ObservationModel> &obs_model,
                       const std::shared_ptr<Scheme> &sampling_scheme,
                       int N, const std::vector<double>& scalings,
                       int thinning_factor, bool reverse)
                       : accumulator(tr_model->stateDim(), tr_model->length(), 1+N/thinning_factor),
                       transitionM(tr_model), observationM(obs_model), sampler(sampling_scheme),
                       numiter(N), thin(thinning_factor), run_reversed(reverse), scaling(scalings) {
        if (run_reversed) {
            std::size_t acc_size = 1 + numiter / thin;
            accumulator.resize(acc_size);
        }
    }


    template<typename Scheme>
    inline void MCMC<Scheme>::initialise(const Data_type& observations, const Sample_type& x_init, u_long seed) {
        sampler->cur_sample = x_init;
        sampler->init(observations, *transitionM, seed);
        sampler->setScales(scaling);
        if (run_reversed)
            sampler->reverseObservations();
    }

    template<typename Scheme>
    void MCMC<Scheme>::run() {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i <= numiter; ++i) {
            sampler->updateIter(i);
            sampler->sample(*transitionM, *observationM);
            if (i % thin == 0)
                accumulator.addSample(sampler->cur_sample, (1+run_reversed)*i/thin);

            if (run_reversed) {
                sampler->setReversed();
                sampler->cur_sample.rowwise().reverseInPlace();
                sampler->sample(*transitionM, *observationM);
                sampler->cur_sample.rowwise().reverseInPlace();
                sampler->setReversed();
                if (i % thin == 0)
                    accumulator.addSample(sampler->cur_sample, 2*(i/thin)+1);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        accumulator.setDuration(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        accumulator.setAcceptances(sampler->acceptances);

    }


}


#endif //BAYSIS_ALGORITHMS_HPP
