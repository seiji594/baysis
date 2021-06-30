//
// main.cpp
// Baysis
//
// Created by Vladimir Sotskov on 25/06/2021, 17:51.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include <chrono>
#include "../baysis/probsupport.hpp"
//#include "../baysis/schemes.hpp"

using namespace std;


int main(int argc, const char * argv[]) {
/*
    RandomSample<std::mt19937, std::normal_distribution<> > rsg{1};
    RandomSample<std::mt19937, std::poisson_distribution<> > rsg2{1};
    std::uniform_int_distribution<> randint(1,10);
    RandomSample<std::mt19937, std::uniform_int_distribution<> > rsg3{randint, 1};

    // Draw 10 random numbers from standard normal
    cout << "Vector of 10 random numbers from normal:\n" << rsg.draw(10) << endl;
    cout << "Vector of 10 random numbers from poisson:\n" << rsg2.draw(10) << endl;
    cout << "Vector of 10 random numbers from U(1,10):\n" << rsg3.draw(10) << endl;

    // Calc log density and log likelihood of Normal
    size_t n(3);
    Vector norm_sample(n);
    Vector means(n);
    means << 2.1, 2.8, 1.9;
    Matrix cov(n, n), ltm(n, n);
    cov << 0.5, 0.25, 0.34, 0.25, 0.6, 0.36, 0.34, 0.36, 0.4;
    Eigen::LLT<Matrix> llt(static_cast<Eigen::Index>(n));
    llt.compute(cov);
    NumericalRcond().checkPD(llt.rcond(), "Matrix is not positive definite");
    ltm = llt.matrixL();
    norm_sample = NormalDist::sample(means, ltm, rsg);
    cout << "A multivariate sample from normal distribution:\n" << norm_sample << endl;
    cout << "Logdensity of the multivariate sample:\n" << NormalDist::logDensity(norm_sample, means, ltm) << endl;
    cout << "Low triangular matrix of Cholesky decomposition:\n" << ltm << endl;

    norm_sample = NormalDist::sample(10, 3.4, 0.5, rsg);
    cout << "A sample of 10 i.i.d. normally distributed variables N(3.4, 0.5):\n" << norm_sample << endl;
    cout << "loglikelihood of the sample:\n" << NormalDist::logLikelihood(norm_sample, 3.4, 0.5) << endl;

    // Calc log density and log likelihood of Poisson
    n = 5;
    Vector_int pois_sample(n);
    Vector lambdas(n);
    lambdas << 2.1, 2.8, 1.9, 0.5, 1.7;
    pois_sample = PoissonDist::sample(lambdas, rsg2);
    cout << "A multivariate sample from poisson distribution:\n" << pois_sample << endl;
    cout << "Logdensity of the multivariate poisson:\n" << PoissonDist::logDensity(pois_sample, lambdas.array()) << endl;

    std::poisson_distribution<> pois(3.4);
    RandomSample<std::mt19937, std::poisson_distribution<> > rsg4{pois, 1};
    pois_sample = PoissonDist::sample(n, rsg4);
    cout << "A sample of 5 i.i.d. poisson distributed variables Poi(3.4):\n" << pois_sample << endl;
    cout << "loglikelihood of the sample:\n" << PoissonDist::logLikelihood(pois_sample, 3.4) << endl;
*/

    Matrix A(128,128), L(128, 128), eye(Matrix::Identity(128, 128));
    L.setRandom();
    A = L * L.transpose();

    auto t1 = chrono::high_resolution_clock::now();
    Eigen::LLT<Matrix> llt(A);
    auto t2 = chrono::high_resolution_clock::now();
    Matrix ltri(llt.matrixL());
    ltri = ltri.inverse().eval();
    std::cout << "Inversed matrix:\n" << ltri.transpose() * ltri << std::endl;
    auto t3 = chrono::high_resolution_clock::now();
    std::cout << "Directly inversed matrix:\n" << A.inverse() << std::endl;
    auto t4 = chrono::high_resolution_clock::now();

    std::cout << "To calc llt = " << chrono::duration_cast<chrono::microseconds>(t2-t1).count() << std::endl;
    std::cout << "To calc inverse with llt = " << chrono::duration_cast<chrono::microseconds>(t3-t2).count() << std::endl;
    std::cout << "To calc inverse directly = " << chrono::duration_cast<chrono::microseconds>(t4-t3).count() << std::endl;
    std::cout << A.inverse().isApprox(ltri.transpose()*ltri);
    return 0;
}