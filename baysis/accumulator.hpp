//
// accumulator.hpp
// Baysis
//
// Created by Vladimir Sotskov on 09/07/2021, 17:30.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#ifndef BAYSIS_ACCUMULATOR_HPP
#define BAYSIS_ACCUMULATOR_HPP

#include <chrono>
#include <vector>
#include "Eigen/Dense"
#include "H5Cpp.h"

using namespace H5;


template<typename Scalar=double>
class SampleAccumulator {
public:
    typedef typename decltype(std::chrono::high_resolution_clock::now())::rep Timedelta;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Sample_type;

    SampleAccumulator(std::size_t nrows, std::size_t ncols, std::size_t nsamples)
    : samples(nsamples, Sample_type::Zero(nrows, ncols)), accepts() {}

    void resize(std::size_t new_sz);
    void addSample(const Sample_type &sample, std::size_t iter);
    void setAcceptances(const Vector_int& acc);
    void setDuration(Timedelta dur);
    Vector getSmoothedMeans(std::size_t t);
    Matrix getSmoothedCov(std::size_t t);
    herr_t save(const std::string& fname);

//private:
    std::vector<Sample_type> samples;
    std::vector<int> accepts;
    Timedelta duration{};

};

template<typename Scalar>
void SampleAccumulator<Scalar>::addSample(const Sample_type &sample, std::size_t iter) {
    samples[iter] = sample;
}

template<typename Scalar>
void SampleAccumulator<Scalar>::setAcceptances(const Vector_int &acc) {
    for (int i = 0; i < acc.size(); ++i) {
        accepts.push_back(acc(i));
    }
}

template<typename Scalar>
void SampleAccumulator<Scalar>::setDuration(Timedelta dur) {
    duration = dur;
}

template<typename Scalar>
void SampleAccumulator<Scalar>::resize(std::size_t new_sz) {
    std::size_t ncols = samples.front().cols();
    std::size_t nrows = samples.front().rows();
    samples = std::vector<Sample_type>(new_sz, Sample_type::Zero(nrows, ncols));
}

template<typename Scalar>
Vector SampleAccumulator<Scalar>::getSmoothedMeans(std::size_t t) {
    Vector retval = Vector::Zero(samples.front().rows());

    for (auto& sample: samples) {
        retval += sample.col(t);
    }

    return retval / samples.size();
}

template<typename Scalar>
Matrix SampleAccumulator<Scalar>::getSmoothedCov(std::size_t t) {
    std::size_t nrows = samples.front().rows();
    Matrix retval(nrows, samples.size());
    Vector means = getSmoothedMeans(t);

    for (int i = 0; i < samples.size(); ++i) {
        retval.col(i) = samples[i].col(t) - means;
    }

    return retval * retval.transpose() / samples.size();
}
/*
template<typename Scalar>
herr_t SampleAccumulator<Scalar>::save(const std::string& fname) {
    const H5std_string file_name("../data/"+fname+".h5");
    const H5std_string dset_name("samples");
    std::size_t dim1 = samples.size();
    std::size_t dim2 = samples.back().rows();
    std::size_t dim3 = samples.back().cols();
    hsize_t dims[3] = {dim1, dim2, dim3};
    double data[dim1][dim2][dim3];

    try {
        Exception::dontPrint();
        H5File file(file_name, H5F_ACC_TRUNC);
        DataSpace dspace(3, dims);
        DataSet dset(file.createDataSet(dset_name, H5T_NATIVE_DOUBLE, dspace));
        dset.write(&samples, H5T_NATIVE_DOUBLE);
        dspace.close();
        dset.close();
        file.close();
    } catch (FileIException err) {
        err.printErrorStack();
        return -1;
    } catch (DataSetIException err) {
        err.printErrorStack();
        return -1;
    } catch (DataSpaceIException err) {
        err.printErrorStack();
        return -1;
    }
    return 1;
}
*/

#endif //BAYSIS_ACCUMULATOR_HPP
