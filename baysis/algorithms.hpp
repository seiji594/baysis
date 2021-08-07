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

    template<typename Filter, typename Smoother>
    class KalmanSmoother {
    public:
        KalmanSmoother(const std::shared_ptr<LGTransitionStationary> &tr_model,
                       const std::shared_ptr<LGObservationStationary> &obs_model);

        void initialise(const Matrix &observations);
        void run();

        Matrix smoothed_means;
        Matrix post_means;
        Matrix smoothed_covs;
        Matrix post_covs;

    private:
        const std::shared_ptr<LGTransitionStationary> transitionM;
        const std::shared_ptr<LGObservationStationary> observationM;
        Filter filter;
        Smoother smoother;
        Matrix data;
        Matrix predicted_means;
        Matrix predicted_covs;
    };


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
        SampleAccumulator& getSamples() { return accumulator; }
        // FIXME: add reset function; pass seed to it (or to run); avoid re-computing quantities each reset

    private:
        SampleAccumulator accumulator;
        const std::shared_ptr<TransitionModel> transitionM;
        const std::shared_ptr<ObservationModel> observationM;
        std::shared_ptr<Scheme> sampler;
        int numiter;
        int thin;
        bool run_reversed;
        std::vector<double> scaling;
    };


    template<typename Filter, typename Smoother>
    KalmanSmoother<Filter, Smoother>::KalmanSmoother(const std::shared_ptr<LGTransitionStationary> &tr_model,
                                                     const std::shared_ptr<LGObservationStationary> &obs_model)
                                                     : transitionM(tr_model), observationM(obs_model),
                                                     filter(tr_model->stateDim(), obs_model->obsDim()),
                                                     smoother(tr_model->stateDim()),
                                                     post_means(tr_model->stateDim(), tr_model->length()),
                                                     predicted_means(tr_model->stateDim(), tr_model->length()),
                                                     smoothed_means(tr_model->stateDim(), tr_model->length()),
                                                     smoothed_covs(tr_model->stateDim(), tr_model->stateDim()*tr_model->length()),
                                                     predicted_covs(tr_model->stateDim(), tr_model->stateDim()*tr_model->length()),
                                                     post_covs(tr_model->stateDim(), tr_model->stateDim()*tr_model->length()) { }

    template<typename Filter, typename Smoother>
    void KalmanSmoother<Filter, Smoother>::initialise(const Matrix &observations) {
        filter.init((*transitionM).getPriorMean(), (*transitionM).getPriorCov());
        data = observations;
        post_means.setZero();
        predicted_means.setZero();
        smoothed_means.setZero();
        post_covs.setZero();
        predicted_covs.setZero();
        smoothed_covs.setZero();
    }

    template<typename Filter, typename Smoother>
    void KalmanSmoother<Filter, Smoother>::run() {
        std::size_t xdim = transitionM->stateDim();
        std::size_t seql = transitionM->length();
        // Forward filter pass
        for (int i = 0; i < seql; ++i) {
            if (i != 0)
                filter.predict(*transitionM); // make prediction of the next state's mean and cov
            predicted_means.col(i) = filter.x;
            predicted_covs.block(0, i * xdim, xdim, xdim) = filter.X;
            filter.observe(*observationM, data.col(i));  // observe data and update the state statistics
            // Saving results
            post_means.col(i) = filter.x;
            post_covs.block(0, i*xdim, xdim, xdim) = filter.X;
        }

        // Backward smoothing pass
        smoother.initSmoother(*observationM, data.col(seql-1),
                              post_means.col(seql-1),
                              post_covs.block(0,xdim*(seql-1), xdim, xdim));
        smoothed_means.col(seql-1) = smoother.x;
        smoothed_covs.block(0, xdim*(seql-1), xdim, xdim) = smoother.X;

        for (int i = seql-2; i >= 0; --i) {
            smoother.predictBack(*transitionM,
                                 predicted_covs.block(0, (i+1)*xdim, xdim, xdim),
                                 post_covs.block(0, i*xdim, xdim, xdim));
            int predix = smoother.updateIndex(i);
            smoother.updateBack(*observationM, data.col(i),
                                predicted_means.col(predix), post_means.col(i),
                                predicted_covs.block(0, xdim * predix, xdim, xdim),
                                post_covs.block(0, xdim*i, xdim, xdim));
            smoothed_means.col(i) = smoother.x;
            smoothed_covs.block(0,i*xdim, xdim, xdim) = smoother.X;
        }
    }


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
