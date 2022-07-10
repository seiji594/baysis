//
// accumulator.cpp
// Baysis
//
// Created by Vladimir Sotskov on 20/01/2022, 21:15.
// Copyright © 2022 Vladimir Sotskov. All rights reserved.
//

#include "accumulator.hpp"


SampleAccumulator::SampleAccumulator(std::size_t nrows, std::size_t ncols, std::size_t nsamples)
: samples(nsamples, Sample_type::Zero(nrows, ncols)),
par_samples(nsamples),
accepts() { }

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
    par_accepts.clear();
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