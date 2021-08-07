#
#  decltypes.pxd
#  Baysis
#
#  Created by Vladimir Sotskov on 27/07/2021, 18:40.
#  Copyright © 2021 Vladimir Sotskov. All rights reserved.
#

from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from '<memory>' namespace 'std' nogil:

    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(T*)
        T* get()
        T& operator*()
        T* operator->()
        void reset(T*)
        bool operator bool()

cdef extern from '<Eigen/Dense>' namespace 'Eigen' nogil:

    cdef cppclass Matrix[Scalar, int, int, int=*, int=*, int=*]:
        Matrix(const Scalar*)



cdef extern from 'algorithms.hpp' namesapce 'algos' nogil:

    cdef cppclass MCMC[Scheme]:
        ctypedef Scheme::Data_type Data_type
        ctypedef Scheme::Sample_type Sample_type

        MCMC[Scheme](const shared_ptr<TransitionModel>& tr_model,
                const shared_ptr<ObservationModel>& obs_model,
                const shared_ptr<Scheme>& sampling_scheme,
                int N, const vector<double>& scalings,
                int thinning_factor, bool reverse)
        void initialise(const Data_type& observations, const Sample_type& x_init, u_long seed)
        void run();

