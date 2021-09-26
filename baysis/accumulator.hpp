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
#include <unordered_map>
#include <Eigen/Dense>
#include "samplingschemes.hpp"


class SampleAccumulator {
public:
    typedef typename decltype(std::chrono::high_resolution_clock::now())::rep Timedelta;
    typedef Matrix Sample_type;

    SampleAccumulator(std::size_t nrows, std::size_t ncols, std::size_t nsamples);

    void resize(std::size_t new_sz);
    template<template<class,class,class> class Scheme, typename M1, typename M2, typename Rng>
    void addSample(const Scheme<M1, M2, Rng>& sampler, std::size_t iter);
    template<typename Scheme>
    void addSample(const schemes::WithParameterUpdate<Scheme>& sampler, std::size_t iter);
    template<template<class,class,class> class Scheme, typename M1, typename M2, typename Rng>
    void setAcceptances(const Scheme<M1, M2, Rng>& sampler);
    template<typename Scheme>
    void setAcceptances(const schemes::WithParameterUpdate<Scheme>& sampler);
    void setDuration(Timedelta dur);
    void setBurnin(double cutoff);
    void reset();
    Vector getSmoothedMeans(std::size_t t) const;
    Matrix getSmoothedCov(const Vector& means, std::size_t t) const;
    const std::vector<Sample_type>& getSamples() const;
    const std::vector<Vector>& getParametersSamples() const;
    const std::vector<int>& getAcceptances() const;
    std::unordered_map<std::string, int> getParametersAcceptances() const;
    Timedelta totalDuration() const;

private:
    int offset{0};
    std::vector<Sample_type> samples;
    std::vector<Vector> par_samples;
    std::vector<int> accepts;
    std::unordered_map<std::string, int> par_accepts;
    Timedelta duration{};
};


SampleAccumulator::SampleAccumulator(std::size_t nrows, std::size_t ncols, std::size_t nsamples)
                                    : samples(nsamples, Sample_type::Zero(nrows, ncols)),
                                    par_samples(nsamples),
                                    accepts() { }

template<template<class,class,class> class Scheme, typename M1, typename M2, typename Rng>
void SampleAccumulator::addSample(const Scheme<M1, M2, Rng>& sampler, std::size_t iter) {
    samples.at(iter) = sampler.cur_sample;
}

template<typename Scheme>
void SampleAccumulator::addSample(const schemes::WithParameterUpdate<Scheme> &sampler, std::size_t iter) {
    samples.at(iter) = sampler.cur_sample;
    if (iter == 0) {
        std::size_t nsamples = samples.size();
        std::size_t nparams = sampler.numParams();
        par_samples.resize(nsamples, Vector::Zero(nparams));
    }
    par_samples.at(iter) << sampler.trm_drivers, sampler.obsm_drivers;
}

template<template<class,class,class> class Scheme, typename M1, typename M2, typename Rng>
void SampleAccumulator::setAcceptances(const Scheme<M1, M2, Rng>& sampler) {
    for (int i = 0; i < sampler.acceptances.size(); ++i) {
        accepts.push_back(sampler.acceptances(i));
    }
}

template<typename Scheme>
void SampleAccumulator::setAcceptances(const schemes::WithParameterUpdate<Scheme> &sampler) {
    for (int i = 0; i < sampler.acceptances.size(); ++i) {
        accepts.push_back(sampler.acceptances(i));
    }
    par_accepts.template emplace("trm_par_acceptances", sampler.trm_par_acceptances);
    par_accepts.template emplace("obsm_par_acceptances", sampler.obsm_par_acceptances);
}

void SampleAccumulator::setDuration(Timedelta dur) {
    duration = dur;
}

void SampleAccumulator::resize(std::size_t new_sz) {
    std::size_t ncols = samples.front().cols();
    std::size_t nrows = samples.front().rows();
    samples = std::vector<Sample_type>(new_sz, Sample_type::Zero(nrows, ncols));
    par_samples = std::vector<Vector>(new_sz);
}

void SampleAccumulator::reset() {
    accepts.clear();
}

Vector SampleAccumulator::getSmoothedMeans(std::size_t t) const {
    Vector retval = Vector::Zero(samples.front().rows());

    for (int i = offset; i < samples.size(); ++i) {
        retval += samples[i].col(t);
    }

    return retval / (samples.size() - offset);
}

Matrix SampleAccumulator::getSmoothedCov(const Vector& means, std::size_t t) const {
    std::size_t nrows = samples.front().rows();
    Matrix retval(nrows, samples.size()-offset);

    for (int i = offset; i < samples.size(); ++i) {
        retval.col(i-offset) = samples[i].col(t) - means;
    }

    return retval * retval.transpose() / (samples.size() - offset);
}

const std::vector<SampleAccumulator::Sample_type>& SampleAccumulator::getSamples() const {
    return samples;
}

const std::vector<Vector> &SampleAccumulator::getParametersSamples() const {
    return par_samples;
}

const std::vector<int> &SampleAccumulator::getAcceptances() const {
    return accepts;
}

std::unordered_map<std::string, int> SampleAccumulator::getParametersAcceptances() const {
    return par_accepts;
}

SampleAccumulator::Timedelta SampleAccumulator::totalDuration() const {
    return duration;
}

void SampleAccumulator::setBurnin(double cutoff) {
    offset = static_cast<int>(samples.size()*cutoff);
}
/*
herr_t SampleAccumulator::write_matrix(const Matrix &mat, const H5::DataSpace &data_space,
                                        H5::DataSet &data_set, std::size_t pos) const {
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

void SampleAccumulator::save(const std::string &fname, const std::unordered_map<std::string, int> &attrs) const {
    File file(fname, File::ReadWrite | File::Create | File::Overwrite);
    DataSet data_set = file.createDataSet<int>("accepts", DataSpace::From(accepts));
    DataSet data_set2 = file.createDataSet<double>("samples", DataSpace::From(samples));
    data_set.write(accepts);
    data_set2.write(samples);

    for (const auto& items: attrs) {
        Attribute a = data_set2.createAttribute<int>(items.first, DataSpace::From(items.second));
        a.write(items.second);
    }

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

}
*/

#endif //BAYSIS_ACCUMULATOR_HPP
