//
// baysis.cpp
// Baysis
//
// Created by Vladimir Sotskov on 27/08/2021, 19:17.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include "baysis/h5bridge.hpp"


int main(int argc, const char * argv[]) {
    File file(std::string(PATH_TO_SPEC)+"specs.h5", File::ReadOnly);

    MCMCsession session(file);
    if (!session.mcmc) {
        throw LogicException("MCMC session failed to initialize.");
    }

    DataInitialiser data_initialiser;
    data_initialiser.initialise(file, session);
    data_initialiser.provideto(session);

    if (file.exist("smoother")) {
        // Run Kalman smoother on models. Only allowed on linear gaussian
        SmootherSession smoother(file);
        smoother.kalman->initialise(data_initialiser.realdata);
        smoother.kalman->run();
    }

    const SampleAccumulator& accumulator = session.mcmc->getStatistics();

    for (u_long seed: session.seeds) {
        session.mcmc->init(session.xinit, seed);
        session.mcmc->run();
        std::stringstream ss;
        ss << PATH_TO_RESULTS << session.id << "_seed" << seed << ".h5";
        accumulator.save(ss.str());
    }

    return 0;
}
