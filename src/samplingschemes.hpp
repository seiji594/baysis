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

enum class SamplerType { metropolis=1, ehmm };


namespace schemes {

    class ISampler {
    public:
        virtual ~ISampler() = default;
        virtual void setData(const Matrix_int &observations) = 0;
        virtual void setData(const Matrix &observations) = 0;
        virtual void init(const TransitionModel& tr_model, const ObservationModel& obs_model) = 0;
        virtual void init(TransitionModel& tr_model, ObservationModel& obs_model) { }
        virtual void setScales(const std::vector<double>& scales) = 0;
        virtual void reset(u_long seed) = 0;
        virtual void sample(const TransitionModel &tr_model, const ObservationModel &obs_model) = 0;
        virtual void sample(TransitionModel& tr_model, ObservationModel& obs_model) { }
        virtual void updateIter(int i) { }
        virtual void reverseObservations() { }
        virtual void setReversed() { }

    protected:
        virtual void cache(const TransitionModel& tr_model) { }
        std::shared_ptr<void> rng;
    };


    template<typename ObsM,typename RNG>
    struct AutoregressiveUpdate {
        template<typename DerivedA, typename DerivedB>
        void propose(const Eigen::DenseBase<DerivedA> &x,
                     const Eigen::DenseBase<DerivedB> &mu,
                     const Eigen::LLT<Matrix> &L);

        template<typename DerivedA, typename DerivedB>
        bool accept(const ObsM& obsm,
                    const Eigen::DenseBase<DerivedA> &y,
                    const Eigen::DenseBase<DerivedB> &x);

        template<typename Derived>
        bool accept(const ObsM& obsm,
                    const Eigen::DenseBase<Derived> &y);

        RandomSample<RNG, std::uniform_real_distribution<> > uniform;
        RandomSample<RNG, std::normal_distribution<> > norm;
        Vector proposal;            // temp for proposals
        double cur_prop_ld{};       // temp for proposal logdensity
        double eps{};
    };


    template<typename ParM, typename RNG>
    struct ParameterUpdate {
        template<typename Derived>
        void init(const ParM &model,
                  const Vector &drivers,
                  const Eigen::DenseBase<Derived> &x);

        template<typename DerivedA, typename DerivedB>
        void init(const ParM &model,
                  const Vector &drivers,
                  const Eigen::DenseBase<DerivedA> &x,
                  const Eigen::DenseBase<DerivedB> &y);

        template<typename Derived>
        void propose(const Eigen::DenseBase<Derived> &mu,
                     const Eigen::LLT<Matrix> &L);

        template<typename Derived>
        bool accept(const ParM& trm,
                    const Eigen::DenseBase<Derived> &x);

        template<typename DerivedA, typename DerivedB>
        bool accept(const ParM& obsm,
                    const Eigen::DenseBase<DerivedA> &y,
                    const Eigen::DenseBase<DerivedB> &x);

        RandomSample<RNG, std::uniform_real_distribution<> > uniform;
        RandomSample<RNG, std::normal_distribution<> > norm;
        Vector proposal;
        double cur_prop_ld{};
        double eps{1.};

    private:
        template<typename Derived>
        double loglikelihood(const ParM& model, const Eigen::DenseBase<Derived>& x);

        template<typename DerivedA, typename DerivedB>
        double loglikelihood(const ParM& model,
                             const Eigen::DenseBase<DerivedA>& obs, const Eigen::DenseBase<DerivedB>& x);
    };


    template<typename TrM, typename ObsM, typename RNG>
    class SingleStateScheme: public ISampler {
    public:
        enum { Type = static_cast<std::size_t>(SamplerType::metropolis) };
        typedef Eigen::Matrix<typename TrM::Value_type, Eigen::Dynamic, Eigen::Dynamic> Sample_type;
        typedef Eigen::Matrix<typename ObsM::Value_type, Eigen::Dynamic, Eigen::Dynamic> Data_type;
        typedef RNG Rng_type;
        typedef TrM TrM_type;
        typedef ObsM ObsM_type;

        static std::size_t Id() { return Type + 100*TrM::Id() + 10*ObsM::Id(); }

        SingleStateScheme() = default;
        explicit SingleStateScheme(std::size_t): SingleStateScheme() { }; // <-- for providing conformity with other scheme
        SingleStateScheme(std::size_t, bool): SingleStateScheme() { }; // <-- for providing conformity with other scheme

        void setData(const Matrix &observations) override;
        void setData(const Matrix_int &observations) override;
        void init(const TransitionModel &tr_model, const ObservationModel &obs_model) override;
        void setScales(const std::vector<double>& scales) override { scalings = scales; }
        void sample(const TransitionModel &tr_model, const ObservationModel &obs_model) override;
        void updateIter(int i) override { ar_update.eps = scalings[i % std::min(scalings.size(), std::size_t(2))]; }
        void reset(u_long seed) override;

        Sample_type cur_sample;
        Vector_int acceptances;

    protected:
        void cache(const TransitionModel &tr_model) override;

        Data_type data;
        std::vector<double> scalings;

    private:
        AutoregressiveUpdate<ObsM, RNG> ar_update;
        // Pre-computed values
        Matrix post_mean_init;
        Matrix scaled_mean_init;
        Matrix scaled_post_mean;
        Matrix scaled_prior_mean;
        Eigen::LLT<Matrix> post_L_init;
        Eigen::LLT<Matrix> post_L;
    };


    template<typename TrM, typename ObsM, typename RNG>
    class EmbedHmmSchemeND: public ISampler{
        enum {even, odd};
    public:
        enum { Type = static_cast<std::size_t>(SamplerType::ehmm) };
        typedef std::size_t Index;
        typedef Eigen::Matrix<typename TrM::Value_type, Eigen::Dynamic, Eigen::Dynamic> Sample_type;
        typedef Eigen::Matrix<typename ObsM::Value_type, Eigen::Dynamic, Eigen::Dynamic> Data_type;
        typedef Eigen::Matrix<typename TrM::Value_type, Eigen::Dynamic, 1> State_type;
        typedef RNG Rng_type;
        typedef TrM TrM_type;
        typedef ObsM ObsM_type;

        static std::size_t Id() { return Type + 100*TrM::Id() + 10*ObsM::Id(); }

        EmbedHmmSchemeND(Index psize, bool flip) : pool_size(psize), noflip(!flip) { }
        explicit EmbedHmmSchemeND(Index psize) : EmbedHmmSchemeND(psize, false) { }
        EmbedHmmSchemeND() : EmbedHmmSchemeND(1) { } // <-- for providing conformity with other schemes

        void init(const TransitionModel &tr_model, const ObservationModel &obs_model) override;
        void setData(const Matrix &observations) override;
        void setData(const Matrix_int &observations) override;
        void setScales(const std::vector<double>& scales) override;
        void sample(const TransitionModel &tr_model, const ObservationModel &obs_model) override;
        void reset(u_long seed) override;
//        void updateIter(int i) {}
        void reverseObservations() override { data_rev = data.rowwise().reverse(); }
        void setReversed() override { reversed = !reversed; }

        Sample_type cur_sample;
        Vector_int acceptances;

    protected:
        Data_type data;
        Data_type data_rev;
        std::vector<double> scalings;

    private:
        void make_pool(const TrM &trm, const ObsM &obsm, Index t, Index a_cached=0);
        void met_update(const TrM &trm, const ObsM &obsm, Index t);
        void shift_update(const TrM &trm, const ObsM &obsm, Index t);
        Index index_update(const TrM &trm, Index t);
        bool flip(Index k, int is_odd);

        Data_type& get_data() { return reversed ? data_rev : data; }
        bool noflip;
        bool reversed=false;
        Index pool_size;
        Index cur_a{};    // auxiliary variable
        Array pool_ld;  // temp for logdensities when drawing auxiliary variable
        State_type cur_state;
        std::vector<Sample_type> pool;
        AutoregressiveUpdate<ObsM, RNG> ar_update;
        RandomSample<RNG, std::uniform_int_distribution<Index> > randint;
    };


    template<typename BaseScheme>
    class WithParameterUpdate: public BaseScheme {
    public:
        typedef typename BaseScheme::TrM_type TrM_type;
        typedef typename BaseScheme::ObsM_type ObsM_type;
        typedef typename BaseScheme::Rng_type Rng_type;

        static std::size_t Id() { return BaseScheme::Id(); }

        template<typename... Args>
        explicit WithParameterUpdate(int npupdates, Args&&... args)
                                    : BaseScheme(std::forward<Args&&>(args)...),
                                    num_param_updates(npupdates) { }

        void init(TransitionModel &tr_model, ObservationModel &obs_model) override;
        void reset(u_long seed) override;
        void sample(TransitionModel &tr_model, ObservationModel &obs_model) override;
        int numParams() const { return trm_drivers.size() + obsm_drivers.size(); }

        Vector trm_drivers;
        Vector obsm_drivers;
        int trm_par_acceptances{};
        int obsm_par_acceptances{};
    private:
        int num_param_updates;
        ParameterUpdate<typename BaseScheme::TrM_type, typename BaseScheme::Rng_type> trm_par_update;
        ParameterUpdate<typename BaseScheme::ObsM_type, typename BaseScheme::Rng_type> obsm_par_update;
    };


    template<typename ObsM, typename RNG>
    template<typename DerivedA, typename DerivedB>
    void AutoregressiveUpdate<ObsM, RNG>::propose(const Eigen::DenseBase<DerivedA> &x,
                                                  const Eigen::DenseBase<DerivedB> &mu,
                                                  const Eigen::LLT<Matrix> &L) {
        proposal = NormalDist::sample(mu.derived() + sqrt(1 - eps * eps) * (x.derived() - mu.derived()), L, norm, eps);
    }

    template<typename ObsM, typename RNG>
    template<typename DerivedA, typename DerivedB>
    bool AutoregressiveUpdate<ObsM, RNG>::accept(const ObsM &obsm,
                                                 const Eigen::DenseBase<DerivedA> &y,
                                                 const Eigen::DenseBase<DerivedB> &x) {
        double alpha = uniform.draw();
        double cur_ld = obsm.logDensity(y, x);
        double prop_ld = obsm.logDensity(y, proposal);
        return exp(prop_ld - cur_ld) > alpha;
    }

    template<typename ObsM, typename RNG>
    template<typename Derived>
    bool AutoregressiveUpdate<ObsM, RNG>::accept(const ObsM &obsm, const Eigen::DenseBase<Derived> &y) {
        double alpha = uniform.draw();
        double prop_ld = obsm.logDensity(y, proposal);
        if (exp(prop_ld - cur_prop_ld) > alpha) {
            cur_prop_ld = prop_ld;
            return true;
        }
        return false;
    }


    template<typename ParM, typename RNG>
    template<typename DerivedA, typename DerivedB>
    void ParameterUpdate<ParM, RNG>::init(const ParM &model, const Vector &drivers,
                                          const Eigen::DenseBase<DerivedA> &x,
                                          const Eigen::DenseBase<DerivedB> &y) {
        cur_prop_ld = loglikelihood(model, y, x) + model.priorLogdensity(drivers);
    }

    template<typename ParM, typename RNG>
    template<typename Derived>
    void ParameterUpdate<ParM, RNG>::init(const ParM &model, const Vector &drivers,
                                          const Eigen::DenseBase<Derived> &x) {
        cur_prop_ld = loglikelihood(model, x) + model.priorLogdensity(drivers);
    }

    template<typename ParM, typename RNG>
    template<typename Derived>
    void ParameterUpdate<ParM, RNG>::propose(const Eigen::DenseBase<Derived> &mu, const Eigen::LLT<Matrix> &L) {
        proposal = NormalDist::sample(mu.derived(), L, norm, eps);
    }

    template<typename ParM, typename RNG>
    template<typename Derived>
    double ParameterUpdate<ParM, RNG>::loglikelihood(const ParM &model, const Eigen::DenseBase<Derived> &x) {
        double ll{0};
        for (int t = 0; t < model.length(); ++t) {
            if (t == 0) {
                ll += model.logDensity(x.col(t));
            } else {
                ll += model.logDensity(x.col(t), x.col(t-1));
            }
        }
        return ll;
    }

    template<typename ParM, typename RNG>
    template<typename DerivedA, typename DerivedB>
    double ParameterUpdate<ParM, RNG>::loglikelihood(const ParM &model, const Eigen::DenseBase<DerivedA> &obs,
                                                     const Eigen::DenseBase<DerivedB> &x) {
        double ll{0};
        for (int t = 0; t < model.length(); ++t) {
            ll += model.logDensity(obs.col(t), x.col(t));
        }
        return ll;
    }

    template<typename ParM, typename RNG>
    template<typename Derived>
    bool ParameterUpdate<ParM, RNG>::accept(const ParM &trm, const Eigen::DenseBase<Derived> &x) {
        double alpha = uniform.draw();
        double prop_ld = loglikelihood(trm, x) + trm.priorLogdensity(proposal);
        if (exp(prop_ld - cur_prop_ld) > alpha) {
            cur_prop_ld = prop_ld;
            return true;
        }
        return false;
    }

    template<typename ParM, typename RNG>
    template<typename DerivedA, typename DerivedB>
    bool ParameterUpdate<ParM, RNG>::accept(const ParM &obsm, const Eigen::DenseBase<DerivedA> &y,
                                            const Eigen::DenseBase<DerivedB> &x) {
        double alpha = uniform.draw();
        double prop_ld = loglikelihood(obsm, y, x) + obsm.priorLogdensity(proposal);
        if (exp(prop_ld - cur_prop_ld) > alpha) {
            cur_prop_ld = prop_ld;
            return true;
        }
        return false;
    }


    template<typename TrM, typename ObsM, typename RNG>
    void SingleStateScheme<TrM, ObsM, RNG>::setData(const Matrix &observations) {
        data = observations.cast<typename Data_type::Scalar>();
    }

    template<typename TrM, typename ObsM, typename RNG>
    void SingleStateScheme<TrM, ObsM, RNG>::setData(const Matrix_int &observations) {
        data = observations.cast<typename Data_type::Scalar>();
    }

    template<typename TrM, typename ObsM, typename RNG>
    void
    SingleStateScheme<TrM, ObsM, RNG>::init(const TransitionModel &tr_model, const ObservationModel &obs_model) {
        // Initialise
        acceptances.setZero(tr_model.length());
        ar_update.proposal.resize(tr_model.stateDim());
        // Cache some quantities
        cache(tr_model);
    }

    template<typename TrM, typename ObsM, typename RNG>
    void SingleStateScheme<TrM, ObsM, RNG>::cache(const TransitionModel &tr_model) {
        const TrM& lg_transition = static_cast<const TrM&>(tr_model);
        Matrix post_cov_init = (lg_transition.getA().transpose() * lg_transition.getCovInv() * lg_transition.getA()
                + lg_transition.getPriorCovInv()).inverse();
        post_mean_init.noalias() = post_cov_init * lg_transition.getA().transpose() * lg_transition.getCovInv();
        scaled_mean_init.noalias() = post_cov_init * lg_transition.getPriorCovInv() * lg_transition.getPriorMean();
        post_L_init.compute(post_cov_init);

        Matrix post_cov = (lg_transition.getA().transpose() * lg_transition.getCovInv() * lg_transition.getA()
                + lg_transition.getCovInv()).inverse();
        scaled_prior_mean.noalias() = post_cov * lg_transition.getCovInv() * lg_transition.getA();
        scaled_post_mean.noalias() = post_cov * lg_transition.getA().transpose() * lg_transition.getCovInv();
        post_L.compute(post_cov);
    }

    template<typename TrM, typename ObsM, typename RNG>
    void SingleStateScheme<TrM, ObsM, RNG>::reset(u_long seed) {
//        std::shared_ptr<RNG> rng;
        if (seed != 0) {
            rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator(seed));
        } else {
            rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator());
        }
        ar_update.uniform = RandomSample<RNG, std::uniform_real_distribution<> >(std::static_pointer_cast<RNG>(rng));
        ar_update.norm = RandomSample<RNG, std::normal_distribution<> >(std::static_pointer_cast<RNG>(rng));
        acceptances.setZero();
    }

    template<typename TrM, typename ObsM, typename RNG>
    void SingleStateScheme<TrM, ObsM, RNG>::sample(const TransitionModel &tr_model,
                                                   const ObservationModel &obs_model) {
        std::size_t T{tr_model.length()-1};

        ar_update.template propose(cur_sample.col(0),
                post_mean_init * cur_sample.col(1) + scaled_mean_init,
                post_L_init);
        if (ar_update.template accept(static_cast<const ObsM&>(obs_model),
                data.col(0), cur_sample.col(0))) {
            cur_sample.col(0) = ar_update.proposal;
            ++acceptances(0);
        }

        for (int t = 1; t < T; ++t) {
            ar_update.template propose(
                    cur_sample.col(t),
                    scaled_prior_mean * cur_sample.col(t-1) + scaled_post_mean * cur_sample.col(t+1),
                    post_L);
            if (ar_update.template accept(static_cast<const ObsM&>(obs_model),
                    data.col(t), cur_sample.col(t))) {
                cur_sample.col(t) = ar_update.proposal;
                ++acceptances(t);
            }
        }

        ar_update.template propose(cur_sample.col(T),
                static_cast<const TrM&>(tr_model).getMean(cur_sample.col(T-1)),
                static_cast<const TrM&>(tr_model).getL());
        if (ar_update.template accept(static_cast<const ObsM&>(obs_model),
                data.col(T), cur_sample.col(T))) {
            cur_sample.col(T) = ar_update.proposal;
            ++acceptances(T);
        }

    }


    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::setScales(const std::vector<double> &scales) {
        // Due to the implementation of drawing the scaling factor (see method met_update())
        // the scales need to be adjusted as so:
        std::size_t backidx = std::min(scales.size()-1, std::size_t(1));
        scalings.front() = scales.front();
        scalings.at(backidx) = scales.at(backidx) - scales.front();
    }

    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::setData(const Matrix &observations) {
        data = observations.cast<typename Data_type::Scalar>();
    }

    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::setData(const Matrix_int &observations) {
        data = observations.cast<typename Data_type::Scalar>();
    }

    template<typename TrM, typename ObsM, typename RNG>
    void
    EmbedHmmSchemeND<TrM, ObsM, RNG>::init(const TransitionModel &tr_model, const ObservationModel &obs_model) {
        // Initialise
        cur_state.resize(tr_model.stateDim());
        acceptances.setZero(2*tr_model.length());
        pool_ld.resize(pool_size);
        pool = std::vector<Sample_type>(pool_size, Sample_type(tr_model.stateDim(), tr_model.length()));
        ar_update.proposal.resize(tr_model.stateDim());
    }

    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::reset(u_long seed) {
//        std::shared_ptr<RNG> rng;
        if (seed != 0) {
            rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator(seed));
        } else {
            rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator());
        }
        const std::uniform_int_distribution<Index> uintdist(0, this->pool_size - 1);
        randint = RandomSample<RNG, std::uniform_int_distribution<Index> >(std::static_pointer_cast<RNG>(rng), uintdist);
        ar_update.uniform = RandomSample<RNG, std::uniform_real_distribution<> >(std::static_pointer_cast<RNG>(rng));
        ar_update.norm = RandomSample<RNG, std::normal_distribution<> >(std::static_pointer_cast<RNG>(rng));
        acceptances.setZero();
    }

    template<typename TrM, typename ObsM, typename RNG>
    bool EmbedHmmSchemeND<TrM, ObsM, RNG>::flip(Index k, int is_odd) {
        return !noflip && (((k % 2) == 0) ^ is_odd);
    }

    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::met_update(const TrM &trm, const ObsM &obsm, Index t) {
        ar_update.eps = scalings.front() + scalings.at(std::min(scalings.size()-1, std::size_t(1))) * ar_update.uniform.draw();
        if (t == 0)
            ar_update.template propose(cur_state, trm.getPriorMean(), trm.getLprior());
        else
            ar_update.template propose(cur_state, trm.getMean(pool[cur_a].col(t-1)), trm.getL());

        if (ar_update.template accept(obsm, get_data().col(t))) {
            cur_state = ar_update.proposal;
            ++acceptances(t);
        }
    }

    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::shift_update(const TrM &trm, const ObsM &obsm, Index t) {
        Index a_prop = randint.draw();
        ar_update.proposal = cur_state + trm.getMean(pool[a_prop].col(t - 1) - pool[cur_a].col(t - 1));

        if (ar_update.template accept(obsm, get_data().col(t))) {
            cur_state = ar_update.proposal;
            cur_a = a_prop;
            ++acceptances(t+trm.length());
        }
    }

    template<typename TrM, typename ObsM, typename RNG>
    typename EmbedHmmSchemeND<TrM, ObsM, RNG>::Index EmbedHmmSchemeND<TrM, ObsM, RNG>::index_update(const TrM &trm, Index t)
    {
        if (t >= trm.length()) { // we're in the backward pass, initialising
            return static_cast<Index>(floor(ar_update.uniform.draw() * pool_size));
        }

        for (int k = 0; k < pool_size; ++k) {
            pool_ld(k) = trm.logDensity(cur_sample.col(t), pool[k].col(t-1));
        }
        pool_ld = exp(pool_ld - pool_ld.maxCoeff());
        pool_ld /= pool_ld.sum();
        signed int a_prop = -1;
        double cumsum = 0;
        double alpha = ar_update.uniform.draw();
        for (int k = 0; k < pool_size; ++k) {
            cumsum += pool_ld(k);
            ++a_prop;
            if (alpha < cumsum)
                break;
        }
        return static_cast<Index>(a_prop);
    }

    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::make_pool(const TrM &trm, const ObsM &obsm, Index t, Index a_cached) {
        double prop_ld_cached;
        int kt = randint.draw();
        cur_state = cur_sample.col(t);
        pool[kt].col(t) = cur_state;
        ar_update.cur_prop_ld = obsm.logDensity(get_data().col(t), cur_state);
        prop_ld_cached = ar_update.cur_prop_ld;
        cur_a = a_cached;

        for (int k = kt-1; k >= 0 ; --k) {
            if (flip(k + 1, odd)) {
                pool[k].col(t) = -1 * cur_state;
            } else {
                met_update(trm, obsm, t);       // Metropolis update
                if (t != 0)
                    shift_update(trm, obsm, t);     // Shift update
                pool[k].col(t) = cur_state;
            }
        }

        cur_state = cur_sample.col(t);
        ar_update.cur_prop_ld = prop_ld_cached;
        cur_a = a_cached;

        for (int k = kt+1; k < pool_size; ++k) {
            if (flip(k - 1, even)) {
                pool[k].col(t) = -1 * cur_state;
            } else {
                if (t != 0)
                    shift_update(trm, obsm, t);     // Shift update
                met_update(trm, obsm, t);       // Metropolis update
                pool[k].col(t) = cur_state;
            }
        }
    }

    template<typename TrM, typename ObsM, typename RNG>
    void EmbedHmmSchemeND<TrM, ObsM, RNG>::sample(const TransitionModel &tr_model, const ObservationModel &obs_model) {
        // Forward pass
        make_pool(static_cast<const TrM&>(tr_model), static_cast<const ObsM&>(obs_model), 0);

        for (int t = 1; t < tr_model.length(); ++t) {
            Index a_start = index_update(static_cast<const TrM&>(tr_model), t);
            make_pool(static_cast<const TrM&>(tr_model), static_cast<const ObsM&>(obs_model), t, a_start);
        }

        // Backward pass
        for (Index t = tr_model.length(); t > 0; --t) {
            Index tk = index_update(static_cast<const TrM&>(tr_model), t);
            cur_sample.col(t-1) = pool[tk].col(t-1);
        }
    }


    template<typename BaseScheme>
    void WithParameterUpdate<BaseScheme>::init(TransitionModel &tr_model, ObservationModel &obs_model) {
        trm_drivers = static_cast<TrM_type&>(tr_model).reset(std::static_pointer_cast<Rng_type>(BaseScheme::rng));
        static_cast<TrM_type &>(tr_model).update(trm_drivers, false);
        trm_par_update.init(static_cast<TrM_type&>(tr_model), trm_drivers, BaseScheme::cur_sample);

        obsm_drivers = static_cast<ObsM_type&>(obs_model).reset(std::static_pointer_cast<Rng_type>(BaseScheme::rng));
        static_cast<ObsM_type &>(obs_model).update(obsm_drivers, false);
        obsm_par_update.init(static_cast<ObsM_type&>(obs_model), obsm_drivers, BaseScheme::cur_sample, BaseScheme::data);

        std::size_t sz = BaseScheme::scalings.size();
        if (sz > 2) {
            trm_par_update.eps = BaseScheme::scalings.at(2);
            obsm_par_update.eps = BaseScheme::scalings.at(std::min(std::size_t(2), sz-1));
        }

        BaseScheme::init(tr_model, obs_model);
    }

    template<typename BaseScheme>
    void WithParameterUpdate<BaseScheme>::reset(u_long seed) {
        BaseScheme::reset(seed);
        trm_par_acceptances = 0;
        obsm_par_acceptances = 0;
        trm_par_update.uniform = RandomSample<Rng_type, std::uniform_real_distribution<> >(std::static_pointer_cast<Rng_type>(BaseScheme::rng));
        trm_par_update.norm = RandomSample<Rng_type, std::normal_distribution<> >(std::static_pointer_cast<Rng_type>(BaseScheme::rng));
        obsm_par_update.uniform = RandomSample<Rng_type, std::uniform_real_distribution<> >(std::static_pointer_cast<Rng_type>(BaseScheme::rng));
        obsm_par_update.norm = RandomSample<Rng_type, std::normal_distribution<> >(std::static_pointer_cast<Rng_type>(BaseScheme::rng));
    }

    template<typename BaseScheme>
    void WithParameterUpdate<BaseScheme>::sample(TransitionModel &tr_model, ObservationModel &obs_model) {
        BaseScheme::sample(tr_model, obs_model);

        for (int i=0; i<num_param_updates; ++i) {
            trm_par_update.template propose(trm_drivers, static_cast<TrM_type&>(tr_model).getParamsL());
            static_cast<TrM_type &>(tr_model).update(trm_par_update.proposal, false);

            if (trm_par_update.template accept(static_cast<TrM_type&>(tr_model), BaseScheme::cur_sample)) {
                trm_drivers = trm_par_update.proposal;
                ++trm_par_acceptances;
            }

            obsm_par_update.template propose(obsm_drivers, static_cast<ObsM_type&>(obs_model).getParamsL());
            static_cast<ObsM_type &>(obs_model).update(obsm_par_update.proposal, false);

            if (obsm_par_update.template accept(static_cast<ObsM_type&>(obs_model),
                    BaseScheme::data, BaseScheme::cur_sample)) {
                obsm_drivers = obsm_par_update.proposal;
                ++obsm_par_acceptances;
            }
        }
        static_cast<TrM_type &>(tr_model).update(trm_drivers);
        static_cast<ObsM_type &>(obs_model).update(obsm_drivers);

        BaseScheme::cache(tr_model);
    }

}

#endif //BAYSIS_SAMPLINGSCHEMES_HPP