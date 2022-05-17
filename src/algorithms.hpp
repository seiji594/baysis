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

constexpr std::size_t FILTER_ID_MULT = 10;

using ssmodels::TransitionModel;
using ssmodels::ObservationModel;
using ssmodels::LGTransitionStationary;
using ssmodels::LGObservationStationary;


namespace algos {

    class IMcmc {
    public:
        virtual ~IMcmc() = default;
        virtual void provideData(const Matrix &observations, double) = 0;
        virtual void provideData(const Matrix_int &observations, int) = 0;
        virtual void reset(const Matrix &x_init, u_long seed) = 0;
        virtual void run() = 0;
        virtual std::shared_ptr<TransitionModel> getTransitionModel() const = 0;
        virtual std::shared_ptr<ObservationModel> getObservationModel() const = 0;
        virtual const SampleAccumulator& getStatistics() const = 0;
    };

    class ISmoother {
    public:
        virtual void initialise(const Matrix &observations) = 0;
        virtual void run() = 0;
        virtual const Matrix& getMeans() const = 0;
        virtual const Matrix& getCovariances() const = 0;
    };


    template<typename Filter, typename Smoother>
    class KalmanSmoother: public ISmoother {
    public:
        static std::size_t Id() { return Filter::Id()*FILTER_ID_MULT + Smoother::Id(); }

        KalmanSmoother(const std::shared_ptr<LGTransitionStationary> &tr_model,
                       const std::shared_ptr<LGObservationStationary> &obs_model);

        void initialise(const Matrix &observations) override;
        void run() override;
        const Matrix& getMeans() const override { return smoothed_means; }
        const Matrix& getCovariances() const override { return smoothed_covs; }

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
    class MCMC: public IMcmc {
    public:
        typedef Scheme Scheme_type;
        typedef typename Scheme::Data_type Data_type;

        static std::size_t Id() { return Scheme::Id(); }

        MCMC(const std::shared_ptr<TransitionModel> &tr_model, const std::shared_ptr<ObservationModel> &obs_model,
             const std::shared_ptr<Scheme> &sampling_scheme, int N,
             std::vector<double> scalings = std::vector<double>(1, 1.), int thinning_factor = 1,
             bool reverse = false);

        void provideData(const Matrix &observations, double) override;
        void provideData(const Matrix_int &observations, int) override;
        void reset(const Matrix& x_init, u_long seed) override;
        void run() override;
        std::shared_ptr<TransitionModel> getTransitionModel() const override { return transitionM; }
        std::shared_ptr<ObservationModel> getObservationModel() const override { return observationM; }
        const SampleAccumulator& getStatistics() const override { return accumulator; }

    private:
        void _provideData(const Data_type &observations);

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
                       int N, std::vector<double>  scalings,
                       int thinning_factor, bool reverse)
                       : accumulator(tr_model->stateDim(), tr_model->length(), 1+N/thinning_factor),
                       transitionM(tr_model), observationM(obs_model), sampler(sampling_scheme),
                       numiter(N), thin(thinning_factor), run_reversed(reverse), scaling(std::move(scalings)) {
        sampler->setScales(scaling);
        if (run_reversed) {
            std::size_t acc_size = 2 * (1 + numiter / thin);
            accumulator.resize(acc_size);
        }
    }

    template<typename Scheme>
    void MCMC<Scheme>::provideData(const Matrix &observations, double) {
        this->_provideData(observations.cast<typename Data_type::Scalar>());
    }

    template<typename Scheme>
    void MCMC<Scheme>::provideData(const Matrix_int &observations, int) {
        this->_provideData(observations.cast<typename Data_type::Scalar>());
    }

    template<typename Scheme>
    void MCMC<Scheme>::_provideData(const Data_type &observations) {
        sampler->setData(observations);
        if (run_reversed)
            sampler->reverseObservations();
    }

    template<typename Scheme>
    void MCMC<Scheme>::reset(const Matrix& x_init, u_long seed) {
        sampler->reset(seed);
        sampler->cur_sample = x_init;
        sampler->init(*transitionM, *observationM);
        accumulator.reset();
    }

    template<typename Scheme>
    void MCMC<Scheme>::run() {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i <= numiter; ++i) {
            sampler->updateIter(i);
            sampler->sample(*transitionM, *observationM);
            if (i % thin == 0)
                accumulator.addSample(*sampler, (1+run_reversed)*i/thin);

            if (run_reversed) {
                sampler->setReversed();
                sampler->cur_sample.rowwise().reverseInPlace();
                sampler->sample(*transitionM, *observationM);
                sampler->cur_sample.rowwise().reverseInPlace();
                sampler->setReversed();
                if (i % thin == 0)
                    accumulator.addSample(*sampler, 2*(i/thin)+1);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        accumulator.setDuration(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        accumulator.setAcceptances(*sampler);

    }


}


#endif //BAYSIS_ALGORITHMS_HPP
