//
// pybridge.cpp
// Baysis
//
// Created by Vladimir Sotskov on 30/07/2021, 09:25.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//


#include <unordered_map>
#include "samplingschemes.hpp"
#include "algorithms.hpp"

using namespace schemes;
using namespace algos;


enum class MeanFunction { bimodal = 4 };


inline Vector bimodal(const Ref<const Vector>& state, const Ref<const Vector>& coeff) {
    return state.array().abs() * coeff.array();
}


std::shared_ptr<LGTransitionStationary> make_model(std::size_t length,
                                                   const Eigen::Map<Matrix>& A, const Eigen::Map<Matrix>& Q,
                                                   const Eigen::Map<Vector>& mean_prior, const Eigen::Map<Matrix>& cov_prior) {
    std::size_t xdim;
    if (A.cols() != A.rows() || Q.cols() != Q.rows() || A.cols() != Q.cols() ||
    cov_prior.cols() != cov_prior.rows() || A.cols() != cov_prior.cols() ||
    mean_prior.rows() != A.cols())
        throw LogicException("Passed matrices for state model do not conform in size");

    xdim = A.cols();
    std::shared_ptr<LGTransitionStationary> trm = std::make_shared<LGTransitionStationary>(length, xdim);
    trm->init(A,Q);
    trm->setPrior(mean_prior, cov_prior);
    return trm;
}

std::shared_ptr<LGObservationStationary> make_model(std::size_t length, const Eigen::Map<Matrix>& C,
                                                    const Eigen::Map<Matrix>& R) {
    std::size_t xdim, ydim;
    if (R.cols() != R.rows() || R.rows() != C.rows())
        throw LogicException("Passed matrices for observation model do not conform in size");

    xdim = C.cols();
    ydim = C.rows();
    std::shared_ptr<LGObservationStationary> obsm = std::make_shared<LGObservationStationary>(length, xdim, ydim);
    obsm->init(C, R);
    return obsm;
}

std::shared_ptr<LPObservationStationary> make_model(std::size_t length,
                                                    const Eigen::Map<Matrix>& C,
                                                    const Eigen::Map<Matrix>& D,
                                                    const Eigen::Map<Vector>& controls) {
    std::size_t xdim, ydim, cdim;
    if (C.rows() != D.rows() || D.cols() != controls.rows())
        throw LogicException("Passed matrices for observation model do not conform in size");
    xdim = C.cols();
    ydim = C.rows();
    cdim = D.cols();
    std::shared_ptr<LPObservationStationary> obsm = std::make_shared<LPObservationStationary>(length, xdim, ydim, cdim);
    obsm->init(C, D, controls);
    return obsm;
}

std::shared_ptr<GPObservationStationary> make_model(std::size_t length, const Eigen::Map<Vector>& C, MeanFunction mftype) {
    std::size_t ydim = C.rows();
    GPObservationStationary::MF mf;
    switch (mftype) {
        case MeanFunction::bimodal:
            mf = bimodal;
            break;
    }
    std::shared_ptr<GPObservationStationary> obsm = std::make_shared<GPObservationStationary>(length, ydim, mf);
    obsm->init(C);
    return obsm;
}


typedef std::unordered_map<std::string, Matrix> Smoother_result;

template<typename F, typename S>
std::shared_ptr<Smoother_result> get_smoothing_dist(const std::shared_ptr<LGTransitionStationary>& trm,
                                                    const std::shared_ptr<LGObservationStationary>& obsm,
                                                    const Eigen::Map<Matrix>& data) {
    KalmanSmoother<F, S> ks(trm, obsm);
    ks.initialise(data);
    ks.run();
    return std::make_shared<Smoother_result>({{"sm_means", ks.smoothed_means}, {"sm_covs", ks.smoothed_covs}});
}
