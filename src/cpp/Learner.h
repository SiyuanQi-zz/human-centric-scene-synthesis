//
// Created by siyuan on 1/25/17.
//

#ifndef CVPR2018_LEARNER_H
#define CVPR2018_LEARNER_H

// System header files
#include <iostream>
#include <string>
#include <algorithm>
#include <ctime>

// Boost header files
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

// Lib header files
#include "lib/easylogging++.h"

// Local header files
#include "macros.h"
#include "Room.h"
#include "helper.h"

using json = nlohmann::json;
namespace fs = boost::filesystem;
namespace ub = boost::numeric::ublas;

namespace FurnitureArranger{
    class Learner {
    public:
        Learner();
        Learner(std::string suncgRoot, std::string metadataPath, std::string workspacePath);

        json read_SUNCG_file(const fs::path &filename);
        std::vector<Room> parse_SUNCG_file(json house);
        bool load_learner();
        void save_learner();

        std::vector<Room>& get_rooms();

        void collect_stats();
        void collect_room_stats();
        void collect_furniture_stats();
        void collect_furniture_wall_stats();
        void collect_group_stats();
        void collect_support_stats();
        void collect_affordance();

        void plot_planning_heat_maps();

        ub::vector<double> get_costs_for_room(Room room);
        void collect_costs();
        void learn_cost_weights();
        void run_mcmc(Room &room, size_t startFurnIndex, size_t endFurnIndex, ub::vector<double> costWeights);

        void test_affordance_transform();

    private:
        unsigned int _costNum = 8;
        double _iterationMax = 50;
        std::string _suncgRoot, _metadataPath, _workspacePath;

        std::vector<Room> _rooms;

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version);
    };

}


#endif //CVPR2018_LEARNER_H
