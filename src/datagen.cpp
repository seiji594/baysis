//
// datagen.cpp
// Baysis
//
// Created by Vladimir Sotskov on 24/05/2022, 21:15.
// Copyright © 2022 Vladimir Sotskov. All rights reserved.
//

#include <iostream>
#include "h5bridge.hpp"


int main(int argc, const char * argv[]) {
    if (argc == 2) {
        try {
            File file(std::string(PATH_TO_SPEC)+argv[1], File::ReadWrite);
            std::cout << "Loaded model specifications." << std::endl;
            DataInitialiser data_initialiser;
            data_initialiser.initialise(file);

            // Saving data
            std::string data_flag = data_initialiser.saveto(file) ? "saved" : "not saved";
            std::cout << "\tGenerated data " << data_flag << " into " << argv[1] << std::endl;
            std::cout << std::endl;
            std::cout << "######### All done #########" << std::endl;
        } catch(FileException& e) {
            std::cerr << e.what() << std::endl;
            return -1;
        }
    } else {
        std::cerr << "No specifications file supplied.\nUsage: datagen <specs_file_name>" << std::endl;
        return -1;
    }

    return 0;
}
