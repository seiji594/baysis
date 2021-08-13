# distutils: language = c++
# cython: language_level=3
#
#  pyapi.pyx
#  Baysis
#
#  Created by Vladimir Sotskov on 28/07/2021, 21:25.
#  Copyright © 2021 Vladimir Sotskov. All rights reserved.
#

from cython.operator cimport dereference as deref
from eigency.core cimport *
from decltypes cimport *
import numpy as np

ctypedef SingleStateScheme[LGTransitionStationary, LGObservationStationary, mt19937] Met_lg2
ctypedef SingleStateScheme[LGTransitionStationary, LPObservationStationary, mt19937] Met_lgp
ctypedef SingleStateScheme[LGTransitionStationary, GPObservationStationary, mt19937] Met_lggp
ctypedef EmbedHmmSchemeND[LGTransitionStationary, LGObservationStationary, mt19937] Ehmm_lg2
ctypedef EmbedHmmSchemeND[LGTransitionStationary, LPObservationStationary, mt19937] Ehmm_lgp
ctypedef EmbedHmmSchemeND[LGTransitionStationary, GPObservationStationary, mt19937] Ehmm_lggp


cpdef enum Filter_scheme:
    covariance
    information


cpdef enum Smoothing_scheme:
    rts
    twofilter


cpdef enum Model_type:
    lingaussian = 10
    linpoisson = 30
    genpoisson = 50


cpdef enum Scheme_type:
    singlestate = 100
    ehmm = 200


cpdef enum Sampler_type:
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

    def __cinit__(self, specs, int numiter, list scalings=None, int thinning=1, bool reverse=False):
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

        cdef Model_type trm_t = transition_specs['name'] 
        cdef Model_type obsm_t = observation_specs['name']
        cdef Scheme_type smplr_t = sampler_specs['name']
        cdef Sampler_type s = <Sampler_type>(trm_t + obsm_t + smplr_t)

        cdef Map[MatrixXd] A = Map[MatrixXd](specs['A'])
        cdef Map[MatrixXd] Q = Map[MatrixXd](specs['Q'])
        cdef Map[MatrixXd] Sprior = Map[MatrixXd](specs['Cov prior'])
        cdef Map[VectorXd] muprior = Map[VectorXd](specs['Mean prior'])
        cdef shared_ptr[LGTransitionStationary] trm = make_model(length, A, Q, muprior, Sprior)
        
        # Declare all varaibles as cdefs in inner scopes are not allowed
        cdef Map[MatrixXd] C
        cdef Map[VectorXd] c
        cdef Map[MatrixXd] D
        cdef Map[MatrixXd] R 
        cdef Map[VectorXd] ctrls
        cdef MeanFunction mftype
        cdef shared_ptr[LGObservationStationary] lgobsm
        cdef shared_ptr[LPObservationStationary] lpobsm
        cdef shared_ptr[GPObservationStationary] gpobsm
        cdef shared_ptr[Met_lg2] metsampler
        cdef shared_ptr[Ehmm_lg2] ehmmsampler
        cdef shared_ptr[Met_lgp] metsampler2
        cdef shared_ptr[Ehmm_lgp] ehmmsampler2
        cdef shared_ptr[Met_lggp] metsampler3
        cdef shared_ptr[Ehmm_lggp] ehmmsampler3

        if s == Sampler_type.met_lg2 or s == Sampler_type.ehmm_lg2:
            C = Map[MatrixXd](observation_specs['C'])
            R = Map[MatrixXd](observation_specs['R'])
            lgobsm = make_model(length, C, R)

            if s == Sampler_type.met_lg2:
                metsampler = shared_ptr[Met_lg2](new Met_lg2())
                self.sampler1 = new MCMC[Met_lg2](<shared_ptr[TransitionModel]>trm, 
                                        <shared_ptr[ObservationModel]>lgobsm, 
                                        metsampler, numiter, scales, thinning, reverse)
            else:
                ehmmsampler = shared_ptr[Ehmm_lg2](new Ehmm_lg2(pool_size, flip))
                self.sampler4 = new MCMC[Ehmm_lg2](<shared_ptr[TransitionModel]>trm, 
                                        <shared_ptr[ObservationModel]>lgobsm,
                                        ehmmsampler, numiter, scales, thinning, reverse)

        elif s == Sampler_type.met_lgp or s == Sampler_type.ehmm_lgp:
            C = Map[MatrixXd](observation_specs['C'])
            D = Map[MatrixXd](observation_specs['D'])
            ctrls = Map[VectorXd](observation_specs['controls'])
            lpobsm = make_model(length, C, D, ctrls)

            if s == Sampler_type.met_lgp:
                metsampler2 = shared_ptr[Met_lgp](new Met_lgp())
                self.sampler2 = new MCMC[Met_lgp](<shared_ptr[TransitionModel]>trm, 
                                        <shared_ptr[ObservationModel]>lpobsm,
                                        metsampler2, numiter, scales, thinning, reverse)
            else:
                ehmmsampler2 = shared_ptr[Ehmm_lgp](new Ehmm_lgp(pool_size, flip))
                self.sampler5 = new MCMC[Ehmm_lgp](<shared_ptr[TransitionModel]>trm, 
                                        <shared_ptr[ObservationModel]>lpobsm, 
                                        ehmmsampler2, numiter, scales, thinning, reverse)

        elif s == Sampler_type.met_lggp or s == Sampler_type.ehmm_lggp:
            c = Map[VectorXd](observation_specs['C'])
            mftype = <MeanFunction>observation_specs['mean function']
            gpobsm = make_model(length, c, mftype)

            if s == Sampler_type.met_lggp:
                metsampler3 = shared_ptr[Met_lggp](new Met_lggp())
                self.sampler3 = new MCMC[Met_lggp](<shared_ptr[TransitionModel]>trm, 
                                        <shared_ptr[ObservationModel]>gpobsm, 
                                        metsampler3, numiter, scales, thinning, reverse)
            else:
                ehmmsampler3 = shared_ptr[Ehmm_lggp](new Ehmm_lggp(pool_size, flip))
                self.sampler6 = new MCMC[Ehmm_lggp](<shared_ptr[TransitionModel]>trm, 
                                        <shared_ptr[ObservationModel]>gpobsm, 
                                        ehmmsampler3, numiter, scales, thinning, reverse)

    def __dealloc__(self):
        if self.sampler1:
            del self.sampler1
        elif self.sampler2:
            del self.sampler2
        elif self.sampler3:
            del self.sampler3
        elif self.sampler4:
            del self.sampler4
        elif self.sampler5:
            del self.sampler5
        elif self.sampler6:
            del self.sampler6

    def initialise(self, np.ndarray observations, np.ndarray[np.float64_t, ndim=2] x_init, int seed):
        if self.sampler1:
            self.sampler1.initialise(Map[MatrixXd](observations), Map[MatrixXd](x_init), seed)
        elif self.sampler2:
            self.sampler2.initialise(Map[MatrixXi](observations), Map[MatrixXd](x_init), seed)
        elif self.sampler3:
            self.sampler3.initialise(Map[MatrixXi](observations), Map[MatrixXd](x_init), seed)
        elif self.sampler4:
            self.sampler4.initialise(Map[MatrixXd](observations), Map[MatrixXd](x_init), seed)
        elif self.sampler5:
            self.sampler5.initialise(Map[MatrixXi](observations), Map[MatrixXd](x_init), seed)
        elif self.sampler6:
            self.sampler6.initialise(Map[MatrixXi](observations), Map[MatrixXd](x_init), seed)
        
    def reset(self, np.ndarray[np.float64_t, ndim=2] x_init, int seed):
        if self.sampler1:
            self.sampler1.reset(Map[MatrixXd](x_init), seed)
        elif self.sampler2:
            self.sampler2.reset(Map[MatrixXd](x_init), seed)
        elif self.sampler3:
            self.sampler3.reset(Map[MatrixXd](x_init), seed)
        elif self.sampler4:
            self.sampler4.reset(Map[MatrixXd](x_init), seed)
        elif self.sampler5:
            self.sampler5.reset(Map[MatrixXd](x_init), seed)
        elif self.sampler6:
            self.sampler6.reset(Map[MatrixXd](x_init), seed)
        
    def run(self):
        if self.sampler1:
            self.sampler1.run()
        elif self.sampler2:
            self.sampler2.run()
        elif self.sampler3:
            self.sampler3.run()
        elif self.sampler4:
            self.sampler4.run()
        elif self.sampler5:
            self.sampler5.run()
        elif self.sampler6:
            self.sampler6.run()
        
    def getStatistics(self):
        stats = Statistics()
        if self.sampler1:
            stats.cinit(self.sampler1.getStatistics())
        elif self.sampler2:
            stats.cinit(self.sampler2.getStatistics())
        elif self.sampler3:
            stats.cinit(self.sampler3.getStatistics())
        elif self.sampler4:
            stats.cinit(self.sampler4.getStatistics())
        elif self.sampler5:
            stats.cinit(self.sampler5.getStatistics())
        elif self.sampler6:
            stats.cinit(self.sampler6.getStatistics())
        return stats
        

cdef class Statistics:
    cdef SampleAccumulator obj
    
    cdef cinit(self, SampleAccumulator& accumulator):
        self.obj = accumulator
        
    def __cinit__(self):
        pass
        
    def samples(self):
        retval = []
        for sample in self.obj.getSamples():
            retval.append(ndarray(sample))
        return retval 
    
    def acceptances(self):
        return self.obj.getAcceptances()
        
    def smoothedMeans(self, t):
        return ndarray_view(self.obj.getSmoothedMeans(t))
        
    def smoothedCov(self, t):
        return ndarray_view(self.obj.getSmoothedCov(t))
    
    def duration(self):
        return self.obj.totalDuration()
    
    

def getSmoothingDistribution(specs, np.ndarray[np.float64_t, ndim=2] observations, Filter_scheme fltr, Smoothing_scheme smoother):
    transition_specs = specs.get['transition model']
    observation_specs = specs.get['observation model']
    cdef size_t length = specs['length']

    if transition_specs is None or observation_specs is None:
        raise ValueError("Incomplete specification of Kalman smoothing algorithm.")

    cdef Map[MatrixXd] A = Map[MatrixXd](specs['A'])
    cdef Map[MatrixXd] Q = Map[MatrixXd](specs['Q'])
    cdef Map[MatrixXd] Sprior = Map[MatrixXd](specs['Cov prior'])
    cdef Map[VectorXd] muprior = Map[VectorXd](specs['Mean prior'])
    cdef Map[MatrixXd] C = Map[MatrixXd](observation_specs['C'])
    cdef Map[MatrixXd] R = Map[MatrixXd](observation_specs['R'])
    cdef Map[MatrixXd] data = Map[MatrixXd](observations)

    cdef shared_ptr[LGTransitionStationary] trm = make_model(length, A, Q, muprior, Sprior)
    cdef shared_ptr[LGObservationStationary] obsm = make_model(length, C, R)
    
    cdef Smoother_result res

    if fltr == Filter_scheme.covariance and smoother == Smoothing_scheme.rts:
        res = get_smoothing_dist[CovarianceScheme, RtsScheme](trm, obsm, data)
        return dict(sm_means=ndarray(deref(res[b'sm_means'])), sm_covs=ndarray(deref(res[b'sm_covs'])))
    elif fltr == Filter_scheme.covariance and smoother == Smoothing_scheme.twofilter:
        res = get_smoothing_dist[CovarianceScheme, TwoFilterScheme](trm, obsm, data)
        return dict(sm_means=ndarray(deref(res[b'sm_means'])), sm_covs=ndarray(deref(res[b'sm_covs'])))
    elif fltr == Filter_scheme.information and smoother == Smoothing_scheme.rts:
        res = get_smoothing_dist[InformationScheme, RtsScheme](trm, obsm, data)
        return dict(sm_means=ndarray(deref(res[b'sm_means'])), sm_covs=ndarray(deref(res[b'sm_covs'])))
    elif fltr == Filter_scheme.information and smoother == Smoothing_scheme.twofilter:
        res = get_smoothing_dist[InformationScheme, TwoFilterScheme](trm, obsm, data)
        return dict(sm_means=ndarray(deref(res[b'sm_means'])), sm_covs=ndarray(deref(res[b'sm_covs'])))


def generateData(specs, int seed):
    transition_specs = specs.get['transition model']
    observation_specs = specs.get['observation model']
    cdef size_t length = specs['length']
    
    if transition_specs is None or observation_specs is None:
        raise ValueError("Incomplete specification of the models.")
        
    cdef Map[MatrixXd] A = Map[MatrixXd](specs['A'])
    cdef Map[MatrixXd] Q = Map[MatrixXd](specs['Q'])
    cdef Map[MatrixXd] Sprior = Map[MatrixXd](specs['Cov prior'])
    cdef Map[VectorXd] muprior = Map[VectorXd](specs['Mean prior'])
    cdef shared_ptr[LGTransitionStationary] trm = make_model(length, A, Q, muprior, Sprior)
    
    cdef Map[MatrixXd] C
    cdef Map[VectorXd] c
    cdef Map[MatrixXd] D
    cdef Map[VectorXd] ctrls
    cdef Map[MatrixXd] R
    cdef MeanFunction mftype
    cdef shared_ptr[LGObservationStationary] lgobsm
    cdef shared_ptr[LPObservationStationary] lpobsm
    cdef shared_ptr[GPObservationStationary] gpobsm
    
    if observation_specs['name'] == Model_type.lingaussian:
        C = Map[MatrixXd](observation_specs['C'])
        R = Map[MatrixXd](observation_specs['R'])
        lgobsm = make_model(length, C, R)
        return ndarray(generate_data(trm, lgobsm, seed))
    elif observation_specs['name'] == Model_type.linpoisson:
        C = Map[MatrixXd](observation_specs['C'])
        D = Map[MatrixXd](observation_specs['D'])
        ctrls = Map[VectorXd](observation_specs['controls'])
        lpobsm = make_model(length, C, D, ctrls)
        return ndarray(generate_data(trm, lpobsm, seed))
    elif observation_specs['name'] == Model_type.genpoisson:
        c = Map[VectorXd](observation_specs['C'])
        mftype = <MeanFunction>observation_specs['mean function']
        gpobsm = make_model(length, c, mftype)
        return ndarray(generate_data(trm, gpobsm, seed))
    