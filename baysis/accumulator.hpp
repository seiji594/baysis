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
#include "eigen3/Eigen/Dense"
//#include "H5Cpp.h"

//using namespace H5;


class SampleAccumulator {
public:
    typedef typename decltype(std::chrono::high_resolution_clock::now())::rep Timedelta;
    typedef Matrix Sample_type;

    SampleAccumulator(std::size_t nrows, std::size_t ncols, std::size_t nsamples)
    : samples(nsamples, Sample_type::Zero(nrows, ncols)), accepts() {}

    void resize(std::size_t new_sz);
    void addSample(const Sample_type &sample, std::size_t iter);
    void setAcceptances(const Vector_int& acc);
    void setDuration(Timedelta dur);
    void reset();
    Vector getSmoothedMeans(std::size_t t) const;
    Matrix getSmoothedCov(std::size_t t) const;
    const std::vector<Sample_type>& getSamples() const;
    const std::vector<int>& getAcceptances() const;
    Timedelta totalDuration() const;
//    herr_t save(const std::string& fname);

private:
//    herr_t write_matrix(const Matrix &mat, const H5::DataSpace &data_space, H5::DataSet &data_set, std::size_t pos);

    std::vector<Sample_type> samples;
    std::vector<int> accepts;
    Timedelta duration{};
};

void SampleAccumulator::addSample(const Sample_type &sample, std::size_t iter) {
    samples[iter] = sample;
}

void SampleAccumulator::setAcceptances(const Vector_int &acc) {
    for (int i = 0; i < acc.size(); ++i) {
        accepts.push_back(acc(i));
    }
}

void SampleAccumulator::setDuration(Timedelta dur) {
    duration = dur;
}

void SampleAccumulator::resize(std::size_t new_sz) {
    std::size_t ncols = samples.front().cols();
    std::size_t nrows = samples.front().rows();
    samples = std::vector<Sample_type>(new_sz, Sample_type::Zero(nrows, ncols));
}

void SampleAccumulator::reset() {
    accepts.clear();
}

Vector SampleAccumulator::getSmoothedMeans(std::size_t t) const {
    Vector retval = Vector::Zero(samples.front().rows());

    for (auto& sample: samples) {
        retval += sample.col(t);
    }

    return retval / samples.size();
}

Matrix SampleAccumulator::getSmoothedCov(std::size_t t) const {
    std::size_t nrows = samples.front().rows();
    Matrix retval(nrows, samples.size());
    Vector means = getSmoothedMeans(t);

    for (int i = 0; i < samples.size(); ++i) {
        retval.col(i) = samples[i].col(t) - means;
    }

    return retval * retval.transpose() / samples.size();
}

const std::vector<SampleAccumulator::Sample_type>& SampleAccumulator::getSamples() const {
    return samples;
}

const std::vector<int> &SampleAccumulator::getAcceptances() const {
    return accepts;
}

SampleAccumulator::Timedelta SampleAccumulator::totalDuration() const {
    return duration;
}
/*
herr_t
SampleAccumulator<Scalar>::write_matrix(const Matrix &mat, const H5::DataSpace &data_space,
                                        H5::DataSet &data_set, std::size_t pos) {
    std::size_t rows = mat.rows();
    std::size_t cols = mat.cols();
    std::size_t stride = mat.colStride();
    hsize_t fstride[3] = {1, cols, rows};
    hsize_t fcount[3] = {1, 1, 1};
    hsize_t fblock[3] = {1, cols, rows};
    hsize_t fstart[3] = {pos, 0, 0};
    hsize_t mdim[2] = {stride, cols};
    H5::DataSpace mem_space(2, mdim);
    data_space.selectHyperslab(H5S_SELECT_SET, fcount, fstart, fstride, fblock);
    data_set.write(mat.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
    return 1;
}

herr_t SampleAccumulator<Scalar>::save(const std::string& fname) {
    const H5std_string file_name("../data/"+fname+".h5");
    const H5std_string dset_name("samples");
    std::size_t dim1 = samples.size();
    std::size_t dim2 = samples.back().rows();
    std::size_t dim3 = samples.back().cols();
    hsize_t dims[3] = {dim1, dim3, dim2};

    try {
//        Exception::dontPrint();
        H5::H5File file(file_name, H5F_ACC_TRUNC);
        H5::DataSpace dspace(3, dims);
        H5::DataSet dset(file.createDataSet(dset_name, H5::PredType::NATIVE_DOUBLE, dspace));

        for (std::size_t i=0; i < samples.size(); ++i) {
            write_matrix(samples[i], dspace, dset, i);
        }

        dspace.close();
        dset.close();
        file.close();
    } catch (H5::FileIException& err) {
        H5::FileIException::printErrorStack();
        return -1;
    } catch (H5::DataSetIException& err) {
        H5::DataSetIException::printErrorStack();
        return -1;
    } catch (H5::DataSpaceIException& err) {
        H5::DataSpaceIException::printErrorStack();
        return -1;
    }
    return 1;
}
*/

#endif //BAYSIS_ACCUMULATOR_HPP
