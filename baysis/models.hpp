//
// models.hpp
// Baysis
//
// Created by Vladimir Sotskov on 11/06/2021.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#ifndef BAYSIS_MODELS_HPP
#define BAYSIS_MODELS_HPP

#include "baysisexception.hpp"
#include "matsupport.hpp"
#include "probsupport.hpp"
#include "paramgenerators.hpp"
#include "ifactories.hpp"


using Eigen::Ref;


namespace ssmodels
{
    class SSModelBase {
    public:
        SSModelBase(std::size_t state_dim, std::size_t seq_length);

        std::size_t length() const { return T; }
        std::size_t stateDim() const { return xdim; }
    protected:
        std::size_t xdim;
        std::size_t T;
        NumericalRcond rclimit;
    };


    class TransitionModel: public SSModelBase {
    public:
        TransitionModel(size_t state_dim, size_t seq_length) : SSModelBase(state_dim, seq_length) { }
    };


    class ObservationModel: public SSModelBase {
    public:
        ObservationModel(std::size_t obs_dim, std::size_t state_dim, std::size_t seq_length)
        : SSModelBase(state_dim, seq_length), ydim(obs_dim) {}

        std::size_t obsDim() const { return ydim; }
    protected:
        std::size_t ydim;
    };

    /**
     * Class representing a linear part of a model
     */
    class LinearModel {
    public:
        LinearModel(std::size_t input_rows, std::size_t input_cols, std::size_t control_size=0);

        template<class Derived>
        void apply(const Eigen::DenseBase<Derived> &vec) const;

        template<class Derived>
        void setInputM(const Eigen::MatrixBase<Derived> &input_m);

        template<class Derived>
        void setControlM(const Eigen::MatrixBase<Derived> &control_m);

        template<class Derived>
        void setControls(const Eigen::MatrixBase<Derived> &cur_ctrl);

    protected:
        Matrix inputM;
        Matrix controlM;
        Vector controls;
        mutable Vector res;     // result of linear transformation by input and control matrices
    };

    /**
     * Linear Gaussian stationary transition model. The parameters not changing with time.
     */
    class LGTransitionStationary: public TransitionModel, public LinearModel {
    public:
        typedef double Value_type;

        LGTransitionStationary(std::size_t seq_length, std::size_t state_size, std::size_t control_size = 0);

        void init(const Matrix& A, const Matrix & Cov, const Matrix& B=Matrix());
        void setPrior(const Vector& mu, const Matrix& sigma);

        template<typename DerivedA, typename DerivedB>
        double logDensity(const Eigen::DenseBase<DerivedA> &curx, const Eigen::DenseBase<DerivedB> &prevx) const;

        template<typename Derived>
        double logDensity(const Eigen::DenseBase<Derived> &curx) const;

        template<typename RNG>
        Vector simulate(const Vector &prev_state, std::shared_ptr<RNG> &rng, bool prior=false) const;

        template<class Derived>
        Vector& getMean(const Eigen::DenseBase<Derived> &x) const {
            apply(x);
            return res;
        }
        const Vector& getPriorMean() const {
            return mu_prior;
        }
        const Matrix& getA() const {
            return inputM;
        }
        const Matrix& getCov() const {
            return Q;
        }
        const Matrix& getCovInv() const {
            return Q_inv;
        }
        const Matrix& getPriorCov() const {
            return Q_prior;
        }
        const Matrix& getPriorCovInv() const {
            return Q_prior_inv;
        }
        const Eigen::LLT<Matrix> & getL() const {
            return LQ;
        }

        const Eigen::LLT<Matrix> & getLprior() const {
            return LQprior;
        }

    private:
        Matrix Q;       // State Gaussian noise covariance
        Matrix Q_inv;
        Matrix Q_prior;
        Matrix Q_prior_inv;
        Vector mu_prior;
        Eigen::LLT<Matrix> LQ;  // Cholesky decomposition
        Eigen::LLT<Matrix> LQprior;
    };


    /**
     * Linear Gaussian stationary observation model. The parameters not changing with time.
     */
    class LGObservationStationary: public ObservationModel, public LinearModel {
    public:
        typedef double Value_type;

        LGObservationStationary(std::size_t seq_length, std::size_t state_size, std::size_t obs_size,
                                std::size_t control_size=0);

        void init(const Matrix& C, const Matrix& Cov, const Matrix& D=Matrix());

        template<typename DerivedA, typename DerivedB>
        double logDensity(const Eigen::DenseBase<DerivedA> &y, const Eigen::DenseBase<DerivedB> &x) const;

        template<typename RNG>
        Vector simulate(const Vector &cur_state, std::shared_ptr<RNG>& rng) const;

        template<class Derived>
        Vector& getMean(const Eigen::DenseBase<Derived> &x) const {
            apply(x);
            return res;
        }
        const Matrix& getCov() const {
            return R;
        }
        const Matrix& getCovInv() const {
            return R_inv;
        }
        const Matrix& getC() const {
            return inputM;
        }
        const Eigen::LLT<Matrix>& getL() {
            return LR;
        }

    private:
        Matrix R;        // Observation Gaussian noise covariance
        Matrix R_inv;
        Eigen::LLT<Matrix> LR;  // Cholesky decomposition
    };


    /**
    * Generalised linear Poisson stationary observation model. The parameters not changing with time.
    */
    class LPObservationStationary: public ObservationModel, public LinearModel {
    public:
        typedef int Value_type;
        typedef Eigen::Matrix<Value_type, Eigen::Dynamic, 1> Sample_type;

        LPObservationStationary(std::size_t seq_length, std::size_t state_size, std::size_t obs_size,
                                std::size_t control_size=0);

        void init(const Matrix &C, const Matrix &D = Matrix(), const Vector &ctrls = Vector());

        template<class Derived>
        Vector& getMean(const Eigen::DenseBase<Derived> &x) const {
            apply(x);
            res = exp(res.array());
            return res;
        }

        template<typename DerivedA, typename DerivedB>
        double logDensity(const Eigen::DenseBase<DerivedA> &y, const Eigen::DenseBase<DerivedB> &x) const;

        template<typename RNG>
        Sample_type simulate(const Vector &cur_state, std::shared_ptr<RNG>& rng) const;
    };
    /**
     * Generalised Poisson stationary observation model. The parameters not changing with time.
     */
    class GPObservationStationary: public ObservationModel {
    public:
        typedef int Value_type;
        typedef Eigen::Matrix<Value_type, Eigen::Dynamic, 1> Sample_type;
        using MF = Vector (*)(const Ref<const Vector>&, const Ref<const Vector>&);

        GPObservationStationary(std::size_t seq_length, std::size_t m_size, MF mf);

        void init(const Vector& mc);

        template<class Derived>
        Vector& getMean(const Eigen::DenseBase<Derived> &x) const {
            mean = mean_function(x, coefficients);
            return mean;
        }

        template<typename DerivedA, typename DerivedB>
        double logDensity(const Eigen::DenseBase<DerivedA> &y, const Eigen::DenseBase<DerivedB> &x) const;

        template<typename RNG>
        Sample_type simulate(const Vector &cur_state, std::shared_ptr<RNG>& rng) const;

    private:
        MF mean_function;
        mutable Vector mean;
        Vector coefficients;
    };


    /**
     * Generic class to allow for model parametrization.
     * @tparam BaseModel - the transition or observation model to parametrise
     * @tparam Params - the parameter generator types for the model
     */
    template<typename BaseModel, typename... Params>
    class ParametrizedModel: public BaseModel {
    public:
        typedef std::tuple<Params...> Model_params;
        static constexpr auto nParams = sizeof...(Params);

        template<typename... Args>
        explicit ParametrizedModel(Model_params parms, Args&&... args);

        template<typename RNG>
        Vector reset(const std::shared_ptr<RNG>& rng);
        void update(const Vector& new_drivers);
        double priorLogdensity(const Vector& new_drivers) const;

        const Eigen::LLT<Matrix>& getParamsL() const { return covL; }

    private:
        template<std::size_t... Is>
        void update_impl(std::index_sequence<Is...>);

        Model_params params;
        Eigen::LLT<Matrix> covL;
    };


    template <typename Derived>
    void LinearModel::apply(const Eigen::DenseBase<Derived> &vec) const {
        if (inputM.isDiagonal()) {
            res = inputM.diagonal().template cwiseProduct(vec.derived());
        } else {
            res = inputM * vec.derived();
        }
        if (controlM.size() != 0) {
            if (controlM.isDiagonal())
                res += controlM.diagonal().template cwiseProduct(controls);
            else
                res += controlM * controls;
        }
    }

    template <typename Derived>
    void LinearModel::setInputM(const Eigen::MatrixBase<Derived> &input_m) {
        if (input_m.size() != inputM.size())
            throw LogicException("Input matrix is the wrong size");
        inputM = input_m;
    }

    template <typename Derived>
    void LinearModel::setControlM(const Eigen::MatrixBase<Derived> &control_m) {
        if (control_m.size() != controlM.size())
            throw LogicException("Control matrix is the wrong size");
        controlM = control_m;
    }

    template <typename Derived>
    void LinearModel::setControls(const Eigen::MatrixBase<Derived> &cur_ctrl) {
        if (cur_ctrl.size() != controls.size())
            throw LogicException("Controls vector is the wrong size");
        controls = cur_ctrl;
    }


    template<typename DerivedA, typename DerivedB>
    double LGTransitionStationary::logDensity(const Eigen::DenseBase<DerivedA> &curx,
                                              const Eigen::DenseBase<DerivedB> &prevx) const {
        return NormalDist::logDensity(curx, getMean(prevx), LQ);
    }

    template<typename Derived>
    double LGTransitionStationary::logDensity(const Eigen::DenseBase<Derived> &curx) const {
        return NormalDist::logDensity(curx, getPriorMean(), LQprior);
    }

    template<typename RNG>
    Vector LGTransitionStationary::simulate(const Vector &prev_state, std::shared_ptr<RNG> &rng, bool prior) const {
        RandomSample<RNG, std::normal_distribution<> > rsg(rng);
        return NormalDist::sample(getMean(prev_state), prior ? LQprior : LQ, rsg);
    }


    template<typename DerivedA, typename DerivedB>
    double LGObservationStationary::logDensity(const Eigen::DenseBase<DerivedA> &y,
                                               const Eigen::DenseBase<DerivedB> &x) const {
        return NormalDist::logDensity(y, getMean(x), LR);
    }

    template<typename RNG>
    Vector LGObservationStationary::simulate(const Vector &cur_state, std::shared_ptr<RNG>& rng) const {
        RandomSample<RNG, std::normal_distribution<> > rsg(rng);
        return NormalDist::sample(getMean(cur_state), LR, rsg);
    }


    template<typename DerivedA, typename DerivedB>
    double LPObservationStationary::logDensity(const Eigen::DenseBase<DerivedA> &y,
                                               const Eigen::DenseBase<DerivedB> &x) const {
        return PoissonDist::logDensity(y, getMean(x).array());
    }

    template<typename RNG>
    LPObservationStationary::Sample_type LPObservationStationary::simulate(const Vector &cur_state,
                                                                           std::shared_ptr<RNG> &rng) const {
        RandomSample<RNG, std::poisson_distribution<> > rsg(rng);
        return PoissonDist::sample(getMean(cur_state), rsg);
    }


    template<typename DerivedA, typename DerivedB>
    double GPObservationStationary::logDensity(const Eigen::DenseBase<DerivedA> &y,
                                               const Eigen::DenseBase<DerivedB> &x) const {
        return PoissonDist::logDensity(y, getMean(x).array());
    }

    template<typename RNG>
    GPObservationStationary::Sample_type GPObservationStationary::simulate(const Vector &cur_state,
                                                                           std::shared_ptr<RNG> &rng) const {
        RandomSample<RNG, std::poisson_distribution<> > rsg(rng);
        return PoissonDist::sample(getMean(cur_state), rsg);
    }


    template<typename BaseModel, typename... Params>
    template<typename... Args>
    ParametrizedModel<BaseModel, Params...>::ParametrizedModel(ParametrizedModel::Model_params parms, Args &&... args)
            : BaseModel(std::forward<Args&&>(args)...),
              params(std::move(parms)) {
        Matrix cov = Matrix::Zero(nParams, nParams);
        static_for(params, [&](auto I, auto p){
            cov(I, I) = p.variance();
        });
        covL.template compute(cov);
    }

    template<typename BaseModel, typename... Params>
    template<typename RNG>
    Vector ParametrizedModel<BaseModel, Params...>::reset(const std::shared_ptr<RNG>& rng) {
        return (Vector(nParams) << static_for(params, [&](auto I, auto p){
            return p.initDraw(rng);
        })).finished();
    }

    template<typename BaseModel, typename... Params>
    double ParametrizedModel<BaseModel, Params...>::priorLogdensity(const Vector &new_drivers) const {
        double ld{0};
        static_for(params, [&](auto I, auto p) {
            ld += p.logDensity(new_drivers(I));
        });
        return ld;
    }

    template<typename Base, typename... Params>
    void ParametrizedModel<Base, Params...>::update(const Vector &new_drivers) {
        if (new_drivers.size() != nParams) {
            std::cerr << "Number of parameter drivers must be equal to number of parameters. Nothing updated." << std::endl;
            return;
        }

        static_for(params, [&](auto I, auto p) {
            p.update(new_drivers(I));
        });
//        apply_updates<std::make_index_sequence<nParams> >(new_drivers);
        update_impl(std::index_sequence_for<Params...>());
    }

//    template<typename BaseModel, typename... Params>
//    template<std::size_t Idx, std::size_t... Ids>
//    void ParametrizedModel<BaseModel, Params...>::apply_updates(const std::vector<double> &drivers) {
//        std::get<Idx>(params).update(drivers[Idx]);
//        apply_updates<std::index_sequence<Ids>...>(drivers);
//    }

    template<typename Base, typename... Params>
    template<std::size_t... Is>
    void ParametrizedModel<Base, Params...>::update_impl(std::index_sequence<Is...>) {
        Base::init(std::get<Is>(params).param...);
        if (typeid(Base) == typeid(LGTransitionStationary)) {
            AutoregressiveStationaryCov prior_cov;
            prior_cov.template update(std::get<0>(params), std::get<1>(params));
            reinterpret_cast<LGTransitionStationary*>(this)->
                    setPrior(reinterpret_cast<LGTransitionStationary*>(this)->getPriorMean(), prior_cov.param);
        }
    }


}

#endif //BAYSIS_MODELS_HPP
