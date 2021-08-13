#
#  decltypes.pxd
#  Baysis
#
#  Created by Vladimir Sotskov on 27/07/2021, 18:40.
#  Copyright © 2021 Vladimir Sotskov. All rights reserved.
#

from libcpp cimport bool, nullptr
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from eigency.core cimport Map, MatrixXd, MatrixXi, VectorXd, PlainObjectBase


cdef extern from '<random>' namespace 'std' nogil:
        
    cdef cppclass mt19937:
        pass
    
    
cdef extern from 'models.hpp' namespace 'ssmodels' nogil:
    
    cdef cppclass SSModelBase:
        pass
    
    cdef cppclass LinearModel:
        pass
    
    cdef cppclass TransitionModel(SSModelBase):
        pass
    
    cdef cppclass ObservationModel(SSModelBase):
        pass
    
    cdef cppclass LGTransitionStationary(TransitionModel, LinearModel):
        pass
    
    cdef cppclass LGObservationStationary(ObservationModel, LinearModel):
        pass
    
    cdef cppclass LPObservationStationary(ObservationModel, LinearModel):
        pass
    
    cdef cppclass GPObservationStationary(ObservationModel):
        pass
    
    
cdef extern from 'algorithms.hpp' namespace 'algos' nogil:

    cdef cppclass MCMC[Scheme]:
        ctypedef PlainObjectBase Data_type
        ctypedef MatrixXd Sample_type

        MCMC(const shared_ptr[TransitionModel]& tr_model,
             const shared_ptr[ObservationModel]& obs_model,
             const shared_ptr[Scheme]& sampling_scheme,
             int N, const vector[double]& scalings,
             int thinning_factor, bool reverse) except +
        void initialise(const Map[MatrixXi]& observations, 
                        const Map[MatrixXd]& x_init, 
                        int seed)
        void initialise(const Map[MatrixXd]& observations, 
                        const Map[MatrixXd]& x_init, 
                        int seed)        
        void reset(const Map[MatrixXd]& x_init, int seed)
        void run()
        const SampleAccumulator& getStatistics() const


cdef extern from 'filterschemes.hpp' namespace 'schemes' nogil:
    
    cdef cppclass LinearStateBase:
        pass
    
    cdef cppclass FilterBase(LinearStateBase):
        pass
    
    cdef cppclass CovarianceScheme(FilterBase):
        pass
    
    cdef cppclass InformationScheme(FilterBase):
        pass
    
    cdef cppclass SmootherBase(LinearStateBase):
        pass
    
    cdef cppclass RtsScheme(SmootherBase):
        pass
    
    cdef cppclass TwoFilterScheme(SmootherBase):
        pass
    

cdef extern from 'samplingschemes.hpp' namespace 'schemes' nogil:
    
    cdef cppclass SingleStateScheme[TrM, ObsM, RNG]:
        SingleStateScheme()
    
    cdef cppclass EmbedHmmSchemeND[TrM, ObsM, RNG]:
        EmbedHmmSchemeND(int psize, bool flip)
    
    
cdef extern from 'accumulator.hpp' nogil:
    
    cdef cppclass SampleAccumulator:
            VectorXd getSmoothedMeans(size_t t)
            MatrixXd getSmoothedCov(size_t t)
            vector[MatrixXd]& getSamples() 
            vector[int]& getAcceptances() 
            int totalDuration()
        
cdef extern from 'pybridge.hpp' nogil:
    cdef cppclass MeanFunction:
        pass

    
    cdef shared_ptr[LGTransitionStationary] make_model(size_t,
                                                       const Map[MatrixXd]&, 
                                                       const Map[MatrixXd]&,
                                                       const Map[VectorXd]&,
                                                       const Map[MatrixXd]&)
    
    cdef shared_ptr[LGObservationStationary] make_model(size_t,
                                                        const Map[MatrixXd]&,
                                                        const Map[MatrixXd]&)
    
    cdef shared_ptr[LPObservationStationary] make_model(size_t,
                                                        const Map[MatrixXd]&,
                                                        const Map[MatrixXd]&,
                                                        const Map[VectorXd]&)
    
    cdef shared_ptr[GPObservationStationary] make_model(size_t,
                                                        const Map[VectorXd]&, 
                                                        MeanFunction)
    
    ctypedef unordered_map[string, MatrixXd*] Smoother_result
    cdef Smoother_result get_smoothing_dist[F, S](const shared_ptr[LGTransitionStationary]&,
                                                  const shared_ptr[LGObservationStationary]&,
                                                  const Map[MatrixXd]&)
    
    cdef PlainObjectBase& generate_data[TrM, ObsM](const TrM& trm, const ObsM& obsm, int seed)
    
    
cdef extern from 'pybridge.hpp' namespace "MeanFunction" nogil:
        cdef MeanFunction bimodal