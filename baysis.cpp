//
// baysis.cpp
// Baysis
//
// Created by Vladimir Sotskov on 27/08/2021, 19:17.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include "baysis/h5bridge.cpp"


int main(int argc, const char * argv[]) {
    File file(std::string(PATH_TO_SPEC)+"specs.h5", File::ReadOnly);
    std::cout << "Loaded model specifications." << std::endl;

    MCMCsession session(file);
    if (!session.mcmc) {
        throw LogicException("MCMC session failed to initialize.");
    }

    DataInitialiser data_initialiser;
    data_initialiser.initialise(file, session);
    data_initialiser.provideto(session);

    if (file.exist("smoother")) {
        // Run Kalman smoother on models. Only allowed on linear gaussian
        std::cout << "\nRunning Kalman smoother...\t";
        SmootherSession smoother(file);
        smoother.kalman->initialise(data_initialiser.realdata);
        smoother.kalman->run();
        std::cout << "Done" << std::endl;
        // Save results
        std::string resfname = std::string(PATH_TO_RESULTS) + "kalman_smoother_results.h5";
        File rfile(resfname, File::ReadWrite | File::Create | File::Overwrite);
        if (!saveResults<double>(rfile, "smoother/means", smoother.kalman->getMeans()) ||
                !saveResults<double>(rfile, "smoother/covariances", smoother.kalman->getCovariances())) {
            std::cout << "Skipping saving Kalman smoother results" << std::endl;
        } else {
            std::cout << "Results saved into " << resfname << std::endl;
        }
    }

    const SampleAccumulator& accumulator = session.mcmc->getStatistics();

    std:: cout << "\nRunning sampler for " << session.id << " with " << session.seeds.size() << " seeds:" << std::endl;
    for (u_long seed: session.seeds) {
        std::cout << "\trunning for seed " << seed << "...\t";
        session.mcmc->init(session.xinit, seed);
        session.mcmc->run();
        std::cout << "Done in " << accumulator.totalDuration() << "ms" << std::endl;
        // Saving results
        std::stringstream ss;
        ss << PATH_TO_RESULTS << session.id << "_results_seed" << seed << ".h5";
        std::string resfname = ss.str();
        File rfile(resfname, File::ReadWrite | File::Create | File::Overwrite);
        if (!saveResults<double>(rfile, "samples", accumulator.getSamples()) ||
                !saveResults<int>(rfile, "accepts", accumulator.getAcceptances())) {
            std::cout << "\tSkipping saving sampling results for seed " << seed << std::endl;
        } else {
            std::cout << "\tResults saved into " << resfname << std::endl;
        }
    }

    return 0;
}
