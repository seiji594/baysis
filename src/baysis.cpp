//
// src.cpp
// Baysis
//
// Created by Vladimir Sotskov on 07/05/2022, 13:36.
// Copyright © 2022 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include "h5bridge.hpp"


void Launch(const File& file);

int main(int argc, const char * argv[]) {
    if (argc == 2) {
        try {
            File file(std::string(PATH_TO_SPEC)+argv[1], File::ReadOnly);
            std::cout << "Loaded model specifications." << std::endl;
            Launch(file);
        } catch(FileException& e) {
            std::cerr << e.what() << std::endl;
            return -1;
        }
    } else {
        std::cerr << "No specifications file supplied.\nUsage: Baysis <specs_file_name>" << std::endl;
        return -1;
    }

    return 0;
}

void Launch(const File& file) {
    MCMCsession session(file);
    if (!session.mcmc) {
        throw LogicException("MCMC session failed to initialize.");
    }

    DataInitialiser data_initialiser;
    data_initialiser.initialise(file);
    data_initialiser.provideto(session);

    const SampleAccumulator& accumulator = session.mcmc->getStatistics();

    std:: cout << "\nRunning sampler for " << session.id << " with " << session.seeds.size() << " seeds:" << std::endl;
    for (u_long seed: session.seeds) {
        std::cout << "\trunning with seed " << seed << " ..." << std::endl;
        session.mcmc->reset(session.xinit, seed);
        session.mcmc->run();
        std::cout << "\tDone in " << accumulator.totalDuration() << "ms" << std::endl;
        // Saving results
        std::stringstream ss;
        ss << PATH_TO_RESULTS << session.id << "_results_seed" << seed << ".h5";
        std::string resfname = ss.str();
        File rfile(resfname, File::ReadWrite | File::Create | File::Overwrite);
        std::unordered_map<std::string, int> attributes(accumulator.getParametersAcceptances());
        attributes.emplace("duration", accumulator.totalDuration());
        if (!saveResults<double>(rfile, "samples", accumulator.getSamples(), attributes) ||
                !saveResults<int>(rfile, "accepts", accumulator.getAcceptances())) {
            std::cout << "\tskipping saving sampling results for seed " << seed << std::endl;
        } else {
            std::cout << "\tresults saved into " << resfname << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "######### All done #########" << std::endl;
}
