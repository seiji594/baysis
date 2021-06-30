//
// schemes.hpp
// Baysis
//
// Created by Vladimir Sotskov on 29/06/2021, 16:03.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#ifndef BAYSIS_SCHEMES_HPP
#define BAYSIS_SCHEMES_HPP

#include "matsupport.hpp"


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
        virtual void init(const Vector& x_init, const Matrix& X_init);
        // Make a predict step
        virtual void predict(LGtransition& lgsm);
        // Compute innovation based on observation of the data
        virtual void observe(LGobserve& lgobsm, const Vector& y) = 0;
        // Transform the internal state into standard representation, if needed
        virtual void transform() = 0;

        // Exposed Numerical Results
        Vector x;               // expected state
        Matrix X;               // state covariance
    protected:
        // Update the expected state and covariance
        virtual void update(LGobserve& lgobsm, const Vector& r) = 0;

        std::size_t last_y_size;
        NumericalRcond rclimit;
        // Permanently allocated temp
        Matrix tempX;
    };

    /**
     * Traditional Kalman filter scheme
     *  Inspired by Bayes++ the Bayesian Filtering Library
     *  Copyright (c) 2002 Michael Stevens
     */
    class CovarianceScheme: FilterBase {
    public:
        CovarianceScheme(std::size_t x_size, std::size_t y_initsize);

        void observe(LGobserve &lgobsm, const Vector &y) override;
        void transform() override { /*No transformation needed*/ }

        Matrix S;		// Innovation Covariance
        Matrix K;		// Kalman Gain

    protected:
        void update(LGobserve& lgobsm, const Vector& r) override;
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
        void initInformation (const Vector &eta, const Matrix& Lambda);  // Initialize with information directly
        void predict(LGtransition &lgsm) override;
        void observe(LGobserve &lgobsm, const Vector &y) override;
        void transform() override;

        Vector x;               // expected state
        Matrix X;               // state covariance
        Vector e;				// Information state
        Matrix La;       		// Information

    protected:
        void update(LGobserve &lgobsm, const Vector &r) override;
        void observe_size (std::size_t y_size);

        bool transform_required;	    // Post-condition of transform is not met
        // Permanently allocated temps
        Matrix R_inv;
    };

}


#endif //BAYSIS_SCHEMES_HPP
