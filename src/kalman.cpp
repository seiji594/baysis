//
// kalman.cpp
// Baysis
//
// Created by Vladimir Sotskov on 24/05/2022, 21:16.
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
        std::cerr << "No specifications file supplied.\nUsage: Kalman <specs_file_name>" << std::endl;
        return -1;
    }

    return 0;
}

void Launch(const File& file) {
    DataInitialiser data_initialiser;
    data_initialiser.initialise(file);

    // Run Kalman smoother on models. Only allowed on linear gaussian
    std::cout << "\nRunning Kalman smoother...\t";
    SmootherSession smoother(file);
    smoother.kalman->initialise(data_initialiser.realdata);
    smoother.kalman->run();
    std::cout << "Done" << std::endl;
    // Save results
    std::string resfname = std::string(PATH_TO_RESULTS) + smoother.id + "_results.h5";
    File rfile(resfname, File::ReadWrite | File::Create | File::Overwrite);
    if (!saveResults<double>(rfile, "smoother/means", smoother.kalman->getMeans()) ||
    !saveResults<double>(rfile, "smoother/covariances", smoother.kalman->getCovariances())) {
        std::cout << "Failed to save Kalman smoother results" << std::endl;
    } else {
        std::cout << "Results saved into " << resfname << std::endl;
    }
    std::cout << "######### All done #########" << std::endl;
}
