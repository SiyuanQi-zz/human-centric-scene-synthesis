//
// Created by siyuan on 6/20/17.
//

#ifndef CVPR2018_ARRANGER_H
#define CVPR2018_ARRANGER_H

// System header files
#include <iostream>
#include <string>
#include <algorithm>
#include <limits>

// Boost header files
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

// Lib header files
#include "lib/easylogging++.h"
#include "lib/json.hpp"

// Local header files
#include "macros.h"
#include "Room.h"
#include "helper.h"
#include "Learner.h"

using json = nlohmann::json;
namespace fs = boost::filesystem;
namespace ub = boost::numeric::ublas;

namespace FurnitureArranger{
    class Sampler {
    public:
        Sampler();
        Sampler(Learner learner, std::string suncgRoot, std::string metadataPath, std::string workspacePath);

        void arrange(int process_id);

        void run_mcmc(Room &room, size_t startFurnIndex, size_t endFurnIndex);
        void anneal(int iteration);
        void cool_down_wall_objects(Room &room);

    private:
        std::string _suncgRoot, _metadataPath, _workspacePath;
        std::vector<Room> _rooms;
        ub::vector<double> _costWeights;

        static int _iterationMax;
        double _temperature, _stepSize;
    };
}


#endif //CVPR2018_ARRANGER_H
