//
// schemes.hpp
// Baysis
//
// Created by Vladimir Sotskov on 29/06/2021, 16:03.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_FILTERSCHEMES_HPP
#define BAYSIS_FILTERSCHEMES_HPP

#include "matsupport.hpp"
#include "models.hpp"

using Eigen::Ref;

enum class FilterScheme: std::size_t { cov=1, info };
enum class SmootherScheme: std::size_t { rts=1, twofilter };


namespace schemes {

    class LinearStateBase {
    public:
        explicit LinearStateBase(std::size_t x_size);
        // Initialize the state of the filter
        virtual void init(const Vector &x_init, const Matrix &X_init);

        // Exposed Numerical Results
        Vector x;        // expected state
        Matrix X;        // state covariance
    protected:
        NumericalRcond rclimit;
    };

    /**
     * Base class for Kalman filter schemes
     */
    class FilterBase : virtual public LinearStateBase {
    public:
        explicit FilterBase(std::size_t x_size);

        // Make a predict step
        virtual void predict(ssmodels::LGTransitionStationary& lgsm);
        // Compute innovation based on observation of the data
        virtual void observe(ssmodels::LGObservationStationary& lgobsm, const Ref<Vector> &y) = 0;

    protected:
        // Update the expected state and covariance
        virtual void update(ssmodels::LGObservationStationary& lgobsm, const Ref<Vector> &r) = 0;
        // Transform the internal state into standard representation, if needed
        virtual void transform() = 0;

        // Permanently allocated temp
        Matrix temp_X;
    };

    /**
     * Traditional Kalman filter scheme
     *  Inspired by Bayes++ the Bayesian Filtering Library
     *  Copyright (c) 2002 Michael Stevens
     */
    class CovarianceScheme: virtual public FilterBase {
    public:
        CovarianceScheme(std::size_t x_size, std::size_t y_size);

        void observe(ssmodels::LGObservationStationary &lgobsm, const Ref<Vector> &y) override;
        static std::size_t Id() { return static_cast<size_t>(FilterScheme::cov); }

        Matrix S;		// Innovation Covariance
        Matrix K;		// Kalman Gain

    protected:
        void transform() override { /*No transformation needed*/ }
        void update(ssmodels::LGObservationStationary& lgobsm, const Ref<Vector> &r) override;

        // Permanently allocated temp
        Matrix tempXY;
    };

    /**
     * Alternative Kalman filter scheme based on canonical parameters of the Gaussian
     *  Inspired by Bayes++ the Bayesian Filtering Library
     *  Copyright (c) 2002 Michael Stevens
     */
    class InformationScheme:  virtual public FilterBase {
    public:
        InformationScheme(std::size_t x_size, std::size_t y_size);

        void init(const Vector &x_init, const Matrix &X_init) override;
        void initInformation (const Ref<Vector> &eta, const Ref<Matrix> &Lambda);  // Initialize with information directly
        void predict(ssmodels::LGTransitionStationary &lgsm) override;
        void observe(ssmodels::LGObservationStationary &lgobsm, const Ref<Vector> &y) override;
        static std::size_t Id() { return static_cast<size_t>(FilterScheme::info); }

        Vector e;				// Information state
        Matrix La;       		// Information

    protected:
        void transform() override;
        void update(ssmodels::LGObservationStationary &lgobsm, const Ref<Vector> &r) override;

        // Permanently allocated temp
        Matrix tempCR;
    };

    /**
     * Base class for all smoothing schemes
     */
     class SmootherBase: virtual public LinearStateBase {
     public:
         explicit SmootherBase(std::size_t x_size);
     };

    /**
     * Rauch, Tung and Striebel smoother scheme
     */
    class RtsScheme: virtual public SmootherBase {
    public:
        explicit RtsScheme(size_t x_size);

        void initSmoother(const ssmodels::LGObservationStationary &lgobsm, const Ref<Vector> &y_final,
                          const Ref<Vector> &x_final, const Ref<Matrix> &X_final) {
            init(x_final, X_final);
        }

        void predictBack(const ssmodels::LGTransitionStationary &lgsm,
                         const Ref<Matrix> &filtered_Xprior,
                         const Ref<Matrix> &filtered_Xpost);

        int updateIndex(int idx) { return idx + 1; }
         /**
          * Makes a backward step
          * @param lgobsm - linear gaussian transition model
          * @param filtered_xprior - predicted filtered expected state saved during filter run
          * @param filtered_xpost - posterior filtered expected state saved during filter run
          * @param filtered_Xprior - predicted filtered state covariance saved during filter run
          * @param filtered_Xpost - posterior filtered state covariance saved during filter run
          */
         void updateBack(const ssmodels::LGObservationStationary &lgobsm, const Ref<Vector> &y,
                         const Ref<Vector> &filtered_xprior, const Ref<Vector> &filtered_xpost,
                         const Ref<Matrix> &filtered_Xprior, const Ref<Matrix> &filtered_Xpost);

         static std::size_t Id() { return static_cast<size_t>(SmootherScheme::rts); }

        Matrix J;       // Backward Kalman gain
    };

    /**
     * Two-filter smoothing scheme.
     * [1] Briers M, Doucet A, Maskell S. Smoothing algorithms for state-space models. IEEE Transactions On Signal Processing, 2009
     */
     class TwoFilterScheme: virtual public SmootherBase {
     public:
         explicit TwoFilterScheme(size_t x_size);

         void initSmoother(const ssmodels::LGObservationStationary &lgobsm, const Ref<Vector> &y_final,
                           const Ref<Vector> &x_final, const Ref<Matrix> &X_final);

         void predictBack(const ssmodels::LGTransitionStationary &lgsm,
                          const Ref<Matrix> &filtered_Xprior,
                          const Ref<Matrix> &filtered_Xpost);

         int updateIndex(int idx) { return idx; }
         void updateBack(const ssmodels::LGObservationStationary &lgobsm, const Ref<Vector> &y,
                         const Ref<Vector> &filtered_xprior, const Ref<Vector> &filtered_xpost,
                         const Ref<Matrix> &filtered_Xprior, const Ref<Matrix> &filtered_Xpost);

         static std::size_t Id() { return static_cast<size_t>(SmootherScheme::twofilter); }

         Matrix Tau;            // Information matrix
         Vector theta;          // Information vector

     protected:
         // Permanently stored temps
         Matrix temp_D;
         Matrix temp_Y;
         const Matrix I;
     };
}


#endif //BAYSIS_FILTERSCHEMES_HPP
