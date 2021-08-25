//
// dataprovider.hpp
// Baysis
//
// Created by Vladimir Sotskov on 14/07/2021, 13:11.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_DATAPROVIDER_HPP
#define BAYSIS_DATAPROVIDER_HPP

#include <Eigen/Dense>


template<typename Scalar>
class DataProvider {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Data_type;

    virtual Ref<Data_type> next() = 0;
};


template<typename TrM, typename ObsM, typename RNG=std::mt19937>
class DataGenerator: DataProvider<typename ObsM::Value_type> {
public:
    typedef typename DataProvider<typename ObsM::Value_type>::Data_type Data_type;
    typedef Eigen::Matrix<typename ObsM::Value_type, Eigen::Dynamic, Eigen::Dynamic> Sample_type;

    DataGenerator(const std::shared_ptr<ssmodels::TransitionModel> &trm,
                  const std::shared_ptr<ssmodels::ObservationModel> &obm,
                  u_long seed=0);

    Ref<Data_type> next() override;
    Ref<Data_type> at(std::size_t t);
    Sample_type& getData() const { return observations; }
    void reset() { cur_t = 0; }

private:
    mutable Sample_type observations;
    std::size_t cur_t;
};


template<typename TrM, typename ObsM, typename RNG>
DataGenerator<TrM, ObsM, RNG>::DataGenerator(const std::shared_ptr<ssmodels::TransitionModel> &trm,
                                             const std::shared_ptr<ssmodels::ObservationModel> &obm,
                                             u_long seed) : cur_t(0) {
    std::shared_ptr<RNG> rng;
    if (seed != 0) {
        rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator(seed));
    }
    else {
        rng = std::make_shared<RNG>(GenericPseudoRandom<RNG>::makeRnGenerator());
    }
    std::size_t T{trm->length()};
    observations.resize(obm->obsDim(), T);
    Vector state(static_cast<const TrM &>(*trm).simulate(static_cast<const TrM &>(*trm).getPriorMean(), rng, true));
    observations.col(0) = static_cast<const ObsM &>(*obm).simulate(state, rng);

    for (int t = 1; t < T; ++t) {
        state = static_cast<const TrM &>(*trm).simulate(state, rng);
        observations.col(t) = static_cast<const ObsM &>(*obm).simulate(state, rng);
    }

}

template<typename TrM, typename ObsM, typename RNG>
Ref<typename DataGenerator<TrM, ObsM, RNG>::Data_type> DataGenerator<TrM, ObsM, RNG>::next() {
    if (cur_t >= observations.cols())
        throw LogicException("Iterator is beyond the last datapoint");
    return observations.col(cur_t++);
}

template<typename TrM, typename ObsM, typename RNG>
Ref<typename DataGenerator<TrM, ObsM, RNG>::Data_type> DataGenerator<TrM, ObsM, RNG>::at(std::size_t t) {
    if (t >= observations.cols())
        throw LogicException("Iterator is beyond the last datapoint");
    return observations.col(t);
}



#endif //BAYSIS_DATAPROVIDER_HPP
