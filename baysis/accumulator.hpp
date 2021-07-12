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


class SampleAccumulator {
public:
    typedef typename decltype(std::chrono::high_resolution_clock::now())::rep Timedelta;
    SampleAccumulator(std::size_t nrows, std::size_t ncols, std::size_t nsamples)
    : samples(std::vector<Matrix>(nsamples, Matrix(nrows, ncols))), accepts(ncols) {}

    void addSample(const Matrix &sample, std::size_t iter);
    void setAcceptances(const Vector_int& acc);
    void setDuration(Timedelta dur);
    herr_t save(const std::string& fname);

//private:
    std::vector<Matrix> samples;
    Vector_int accepts;
    Timedelta duration{};

};

void SampleAccumulator::addSample(const Matrix &sample, std::size_t iter) {
    samples[iter] = sample;
}

void SampleAccumulator::setAcceptances(const Vector_int &acc) {
    accepts = acc;
}

void SampleAccumulator::setDuration(Timedelta dur) {
    duration = dur;
}
/*
herr_t SampleAccumulator::save(const std::string& fname) {
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
