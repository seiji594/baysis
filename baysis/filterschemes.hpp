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

using Eigen::Ref;

class LGtransition;
class LGobserve;

namespace schemes {

    /**
     * Base class for Kalman filter schemes
     */
    class FilterBase {
    public:
        explicit FilterBase(std::size_t x_size);
        // Initialize the state of the filter
        virtual void init(const Vector &x_init, const Matrix &X_init);
        // Make a predict step
        virtual void predict(LGtransition& lgsm);
        // Compute innovation based on observation of the data
        virtual void observe(LGobserve& lgobsm, const Ref<Vector> &y) = 0;
        // Exposed Numerical Results
        Vector x;               // expected state
        Matrix X;               // state covariance

    protected:
        // Update the expected state and covariance
        virtual void update(LGobserve& lgobsm, const Vector &r) = 0;
        // Transform the internal state into standard representation, if needed
        virtual void transform() = 0;

        std::size_t last_y_size;
        NumericalRcond rclimit;
        // Permanently allocated temp
        Matrix temp_X;
    };

    /**
     * Traditional Kalman filter scheme
     *  Inspired by Bayes++ the Bayesian Filtering Library
     *  Copyright (c) 2002 Michael Stevens
     */
    class CovarianceScheme: FilterBase {
    public:
        CovarianceScheme(std::size_t x_size, std::size_t y_initsize);

        void observe(LGobserve &lgobsm, const Ref<Vector> &y) override;

        Matrix S;		// Innovation Covariance
        Matrix K;		// Kalman Gain

    protected:
        void transform() override { /*No transformation needed*/ }
        void update(LGobserve& lgobsm, const Vector &r) override;
        // Resize in case the observations size changes
        void observe_size(std::size_t y_size);
    };

    /**
     * Alternative Kalman filter scheme based on canonical parameters of the Gaussian
     *  Inspired by Bayes++ the Bayesian Filtering Library
     *  Copyright (c) 2002 Michael Stevens
     */
    class InformationScheme: FilterBase {
    public:
        InformationScheme(std::size_t x_size, std::size_t z_initsize);

        void init(const Vector &x_init, const Matrix &X_init) override;
        void initInformation (const Vector &eta, const Matrix &Lambda);  // Initialize with information directly
        void predict(LGtransition &lgsm) override;
        void observe(LGobserve &lgobsm, const Ref<Vector> &y) override;

        Vector e;				// Information state
        Matrix La;       		// Information

    protected:
        void transform() override;
        void update(LGobserve &lgobsm, const Vector &r) override;
        void observe_size (std::size_t y_size);

        bool transform_required;	    // Post-condition of transform is not met
        // Permanently allocated temps
        Matrix R_inv;
    };

    /**
     * Base class for all smoothing schemes
     */
     class SmootherBase {
     public:
         explicit SmootherBase(std::size_t x_size);
         // Initialize the state of the filter
         virtual void init(const Vector& x_final, const Matrix& X_final);

         // Exposed Numerical Results
         Vector x;               // expected state from smoothed distribution
         Matrix X;               // state covariance from smoothed distribution

     protected:
         NumericalRcond rclimit;
     };

    /**
     * RTS smoother scheme
     */
    class RtsScheme: SmootherBase {
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
        void updateBack(const LGtransition &lgsm,
                        const Vector &filtered_xprior, const Vector &filtered_xpost,
                        const Matrix &filtered_Xprior, const Matrix &filtered_Xpost);

        Matrix J;       // Backward Kalman gain
    };

    /**
     * Two-way smoothing scheme.
     * [1] Briers M, Doucet A, Maskell S. Smoothing algorithms for state-space models. IEEE Transactions On Signal Processing, 2009
     */
     class TwoWayScheme: SmootherBase {
     public:
         explicit TwoWayScheme(size_t x_size);

         void initInformation(LGobserve &lgobsm, const Ref<Vector> &y_final,
                              const Vector &x_final, const Matrix &X_final);
         void predictBack(LGtransition &lgsm);
         void updateBack(LGobserve &lgobsm, const Ref<Vector> &y,
                         const Vector &filtered_xprior, const Matrix &filtered_Xprior );

         Matrix Tau;            // Infomraiton matrix
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
