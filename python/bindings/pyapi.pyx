# distutils: language = c++
# cython: language_level=3
#
#  pyapi.pyx
#  Baysis
#
#  Created by Vladimir Sotskov on 28/07/2021, 21:25.
#  Copyright © 2021 Vladimir Sotskov. All rights reserved.
#

import numpy as np
from eigency.core cimport *

ctypedef SingleStateScheme[LGTransitionStationary, LGObservationStationary, std::mt19937] Met_lg2
ctypedef SingleStateScheme[LGTransitionStationary, LPObservationStationary, std::mt19937] Met_lgp
ctypedef SingleStateScheme[LGTransitionStationary, GPObservationStationary, std::mt19937] Met_lggp
ctypedef EmbedHmmSchemeND[LGTransitionStationary, LGObservationStationary, std::mt19937] Ehmm_lg2
ctypedef EmbedHmmSchemeND[LGTransitionStationary, LPObservationStationary, std::mt19937] Ehmm_lgp
ctypedef EmbedHmmSchemeND[LGTransitionStationary, GPObservationStationary, std::mt19937] Ehmm_lggp


cpdef enum class Filter_scheme:
    covariance
    information


cpdef enum class Smoothing_scheme:
    rts
    twofilter


cpdef enum class Model_type:
    lingaussian = 10
    linpoisson = 30
    genpoisson = 50


cpdef enum class Scheme_type:
    singlestate = 100
    ehmm = 200


cpdef enum class Sampler_type:
    met_lg2 = 120
    ehmm_lg2 = 220
    met_lgp = 140
    ehmm_lgp = 240
    met_lggp = 160
    ehmm_lggp = 260


cdef class McmcSession:
    cdef MCMC[Met_lg2]* sampler1
    cdef MCMC[Met_lgp]* sampler2
    cdef MCMC[Met_lggp]* sampler3
    cdef MCMC[Ehmm_lg2]* sampler4
    cdef MCMC[Ehmm_lgp]* sampler5
    cdef MCMC[Ehmm_lggp]* sampler6

    def __cinit__(self, specs, int numiter, double[] scalings=None, int thinning=1, bool reverse=False):
        transition_specs = specs.get['transition model']
        observation_specs = specs.get['observation model']
        sampler_specs = specs.get['sampler']
        cdef size_t length = specs['length']
        cdef vector[double] scales
        cdef bool flip
        cdef size_t pool_size

        if transition_specs is None or observation_specs is None or sampler_specs is None:
            raise ValueError("Incomplete specification of MCMC algorithm.")

        sampler1 = nullptr
        sampler2 = nullptr
        sampler3 = nullptr
        sampler4 = nullptr
        sampler5 = nullptr
        sampler6 = nullptr

        if sampler_specs['name'] == Scheme_type.ehmm:
            flip = sampler_specs['flip']
            pool_size = sampler_specs['pool size']

        if scalings is None:
            scales = [1.]
        else:
            scales = scalings

        cdef Sampler_type s = <Sampler_type>(transition_specs['name'] + observation_specs['name'] + sampler_specs['name'])

        Map[Matrix] A = Map[Matrix](specs['A'])
        Map[Matrix] Q = Map[Matrix](specs['Q'])
        Map[Matrix] Sprior = Map[Matrix](specs['Cov prior'])
        Map[Vector] muprior = Map[Vector](specs['Mean prior'])
        shared_ptr[LGTransitionModel] trm = make_model(length, A, Q, muprior, Sprior)

        if s == Sampler_type.met_lg2 or s == Sampler_type.ehmm_lg2:
            Map[Matrix] C = Map[Matrix](observation_specs['C'])
            Map[Matrix] R = Map[Matrix](observation_specs['R'])
            shared_ptr[LGObservationModel] obsm = make_model(length, C, R)

            if s == Sampler_type.met_lg2:
                shared_ptr[Met_lg2] sampler = shared_ptr[Met_lg2](new Met_lg2())
                self.sampler1 = new MCMC[Met_lg2](trm, obsm, sampler, numiter, scales, thinning, reverse)
            else:
                shared_ptr[Ehmm_lg2] sampler = shared_ptr[Ehmm_lg2](new Ehmm_lg2(pool_size, flip))
                self.sampler4 = new MCMC[Ehmm_lg2](trm, obsm, sampler, numiter, scales, thinning, reverse)

        elif s == Sampler_type.met_lgp or s == Sampler_type.ehmm_lgp:
            Map[Matrix] C = Map[Matrix](observation_specs['C'])
            Map[Matrix] D = Map[Matrix](observation_specs['D'])
            Map[Vector] ctrls = Map[Vector](observation_specs['controls'])
            shared_ptr[LPObservationModel] obsm = make_model(length, C, D, ctrls)

            if s == Sampler_type.met_lgp:
                shared_ptr[Met_lgp] sampler = shared_ptr[Met_lgp](new Met_lgp())
                self.sampler2 = new MCMC[Met_lgp](trm, obsm, sampler, numiter, scales, thinning, reverse)
            else:
                shared_ptr[Ehmm_lgp] sampler = shared_ptr[Emm_lgp](new Ehmm_lgp(pool_size, flip))
                self.sampler5 = new MCMC[Ehmm_lgp](trm, obsm, sampler, numiter, scales, thinning, reverse)

        elif s == Sampler_type.met_lggp or s == Sampler_type.ehmm_lggp:
            Map[Vector] C = Map[Vector](observation_specs['C'])
            MeanFunction mftype = observation_specs['mean function']
            shared_ptr[GPObservationModel] obsm = make_model(length, C, mftype)

            if s == Sampler_type.met_lggp:
                shared_ptr[Met_lggp] sampler = shared_ptr[Met_lggp](new Met_lggp())
                self.sampler3 = new MCMC[Met_lggp](trm, obsm, sampler, numiter, scales, thinning, reverse)
            else:
                shared_ptr[Ehmm_lggp] sampler = shared_ptr[Emm_lggp](new Ehmm_lggp(pool_size, flip))
                self.sampler6 = new MCMC[Ehmm_lggp](trm, obsm, sampler, numiter, scales, thinning, reverse)

    def __selector(self):
        if sampler1:
            return self.sampler1
        elif sampler2
            return self.sampler2
        elif sampler3
            return self.sampler3
        elif sampler4
            return self.sampler4
        elif sampler5
            return self.sampler5
        elif sampler6
            return self.sampler6



def getSmoothingDistribution(specs, np.ndarray[np.floaty64_t, ndim=2] observations, Filter_scheme filter, Smoothing_scheme smoother):
    transition_specs = specs.get['transition model']
    observation_specs = specs.get['observation model']
    cdef size_t length = specs['length']

    if transition_specs is None or observation_specs is None:
        raise ValueError("Incomplete specification of Kalman smoothing algorithm.")

    Map[Matrix] A = Map[Matrix](specs['A'])
    Map[Matrix] Q = Map[Matrix](specs['Q'])
    Map[Matrix] Sprior = Map[Matrix](specs['Cov prior'])
    Map[Vector] muprior = Map[Vector](specs['Mean prior'])
    Map[Matrix] C = Map[Matrix](observation_specs['C'])
    Map[Matrix] R = Map[Matrix](observation_specs['R'])
    Map[Matrix] data = Map[Matrix](observations)

    shared_ptr[LGTransitionModel] trm = make_model(length, A, Q, muprior, Sprior)
    shared_ptr[LGObservationModel] obsm = make_model(length, C, R)

    if filter == Filter_scheme.covariance and smoother == Smoothing_scheme.rts:
        return get_smoothing_dist[CovarianceScheme, RtsScheme](trm, obsm, data)
    elif filter == Filter_scheme.covariance and smoother == Smoothing_scheme.twofilter:
        return get_smoothing_dist[CovarianceScheme, TwoFilterScheme](trm, obsm, data)
    elif filter == Filter_scheme.information and smoother == Smoothing_scheme.rts:
        return get_smoothing_dist[InformationScheme, RtsScheme](trm, obsm, data)
    elif filter == Filter_scheme.information and smoother == Smoothing_scheme.twofilter:
        return get_smoothing_dist[InformationScheme, TwoFilterScheme](trm, obsm, data)

