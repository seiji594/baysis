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
        MCMC(const std::shared_ptr<TransitionModel> &tr_model, const std::shared_ptr<ObservationModel> &obs_model,
             const std::shared_ptr<Scheme> &sampling_scheme, int N, int thinning_factor = 1,
             const std::vector<double> &scalings = std::vector<double>(1, 1.));

        void initialise(const Matrix &observations, const Matrix &x_init, u_long seed = 0);
        void run();

        SampleAccumulator accumulator;
    private:
        const std::shared_ptr<TransitionModel> transitionM;
        const std::shared_ptr<ObservationModel> observationM;
        std::shared_ptr<Scheme> sampler;
        int numiter;
        int thin;
        std::vector<double> scaling;
    };


    template<typename Scheme>
    MCMC<Scheme>::MCMC(const std::shared_ptr<TransitionModel> &tr_model,
                       const std::shared_ptr<ObservationModel> &obs_model,
                       const std::shared_ptr<Scheme> &sampling_scheme, int N, int thinning_factor,
                       const std::vector<double> &scalings)
                            : transitionM(tr_model), observationM(obs_model), sampler(sampling_scheme),
                            accumulator(SampleAccumulator(tr_model->stateDim(), tr_model->length(), 1+N/thinning_factor)),
                            numiter(N), thin(thinning_factor), scaling(scalings) { }


    template<typename Scheme>
    inline void MCMC<Scheme>::initialise(const Matrix &observations, const Matrix &x_init, u_long seed) {
        sampler->cur_sample = x_init;
        sampler->init(observations, *transitionM, seed);
    }

    template<typename Scheme>
    void MCMC<Scheme>::run() {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i <= numiter; ++i) {
            sampler->setScaling(scaling, i);
            sampler->sample(*transitionM, *observationM);
            if (i % thin == 0)
                accumulator.addSample(sampler->cur_sample, i/thin);
        }
        auto end = std::chrono::high_resolution_clock::now();
        accumulator.setDuration(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        accumulator.setAcceptances(sampler->acceptances);
    }


}


#endif //BAYSIS_ALGORITHMS_HPP
