//
// models.hpp
// Baysis
//
// Created by Vladimir Sotskov on 11/06/2021.
//  The code inspired in large part by Bayes++ the Bayesian Filtering Library.
//  Copyright (c) 2003,2004,2005,2006,2011,2012,2014 Michael Stevens
//  Copyright (c) 2002 Michael Stevens and Australian Centre for Field Robotics
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#ifndef BAYSIS_MODELS_HPP
#define BAYSIS_MODELS_HPP

#include "baysisexception.hpp"
#include "matsupport.hpp"


namespace ssm
{

    /**
     * A very abstract Polymorphic base representation
     * Interface provides: type, internal error handing, and destruction
     */
    class ModelBase {
    public:

        virtual ~ModelBase() = default;

    };


    /**
     * Abstract transition models
     * Used to parameterise transition function and nosie
     */
    class TransitionModel: public virtual ModelBase {
        // Empty
    };


    class GaussianNoiseTransition: public virtual TransitionModel {
    public:
        explicit GaussianNoiseTransition(std::size_t x_size);

        Matrix Q;        // Noise covariance

//    NumericalRcond rclimit;
        // Reciprocal condition number limit of linear components when factorised or inverted
    };


    class LinearGaussianTransition: public GaussianNoiseTransition {
    public:
        explicit LinearGaussianTransition(std::size_t x_size);

        Matrix A;   // linear matrix
    };


    /**
     * Abstract Observation models
     *  Observe models are used to parameterise the observe functions of filters
     */
    class ObservationModel: public ModelBase {
        // Empty
    };

}

#endif //BAYSIS_MODELS_HPP
