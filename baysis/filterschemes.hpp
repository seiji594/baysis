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
using ssm::LGTransition;
using ssm::LGObservation;

//FIXME: change all functions accepting Eigen objects into templated functions or to accept Ref<>
namespace schemes {

    class LinearStateBase {
    public:
        explicit LinearStateBase(std::size_t x_size);
        // Initialize the state of the filter
        virtual void init(const Vector &x_init, const Matrix &X_init);

        // Exposed Numerical Results
        Vector x;
        Matrix X;
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
        virtual void predict(LGTransition& lgsm);
        // Compute innovation based on observation of the data
        virtual void observe(LGObservation& lgobsm, const Ref<Vector> &y) = 0;
        // expected state
        // state covariance

    protected:
        // Update the expected state and covariance
        virtual void update(LGObservation& lgobsm, const Vector &r) = 0;
        // Transform the internal state into standard representation, if needed
        virtual void transform() = 0;

        std::size_t last_y_size;
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
        CovarianceScheme(std::size_t x_size, std::size_t y_initsize);

        void observe(LGObservation &lgobsm, const Ref<Vector> &y) override;

        Matrix S;		// Innovation Covariance
        Matrix K;		// Kalman Gain

    protected:
        void transform() override { /*No transformation needed*/ }
        void update(LGObservation& lgobsm, const Vector &r) override;
        // Resize in case the observations size changes
        void observe_size(std::size_t y_size);
    };

    /**
     * Alternative Kalman filter scheme based on canonical parameters of the Gaussian
     *  Inspired by Bayes++ the Bayesian Filtering Library
     *  Copyright (c) 2002 Michael Stevens
     */
    class InformationScheme:  virtual public FilterBase {
    public:
        InformationScheme(std::size_t x_size, std::size_t z_initsize);

        void init(const Vector &x_init, const Matrix &X_init) override;
        void initInformation (const Vector &eta, const Matrix &Lambda);  // Initialize with information directly
        void predict(LGTransition &lgsm) override;
        void observe(LGObservation &lgobsm, const Ref<Vector> &y) override;

        Vector e;				// Information state
        Matrix La;       		// Information

    protected:
        void transform() override;
        void update(LGObservation &lgobsm, const Vector &r) override;
        void observe_size (std::size_t y_size);

        // Permanently allocated temps
        Matrix R_inv;
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

         /**
          * Makes a backward step
          * @param lgsm - linear gaussian transition model
          * @param filtered_xprior - predicted filtered expected state saved during filter run
          * @param filtered_xpost - posterior filtered expected state saved during filter run
          * @param filtered_Xprior - predicted filtered state covariance saved during filter run
          * @param filtered_Xpost - posterior filtered state covariance saved during filter run
          */
        void updateBack(const LGTransition &lgsm,
                        const Vector &filtered_xprior, const Vector &filtered_xpost,
                        const Matrix &filtered_Xprior, const Matrix &filtered_Xpost);

        Matrix J;       // Backward Kalman gain
    };

    /**
     * Two-way smoothing scheme.
     * [1] Briers M, Doucet A, Maskell S. Smoothing algorithms for state-space models. IEEE Transactions On Signal Processing, 2009
     */
     class TwoWayScheme: virtual public SmootherBase {
     public:
         explicit TwoWayScheme(size_t x_size);

         void initInformation(LGObservation &lgobsm, const Ref<Vector> &y_final,
                              const Vector &x_final, const Matrix &X_final);
         void predictBack(LGTransition &lgsm);
         void updateBack(LGObservation &lgobsm, const Ref<Vector> &y,
                         const Vector &filtered_xprior, const Matrix &filtered_Xprior );

         Matrix Tau;            // Information matrix
         Vector theta;          // Information vector

     protected:
         // Resize in case the observations size changes
         void observe_size(std::size_t y_size);

         size_t last_y_size;
         // Permanently stored temps
         Matrix temp_D;
         Matrix temp_Y;
         const Matrix I;
     };
}


#endif //BAYSIS_FILTERSCHEMES_HPP
