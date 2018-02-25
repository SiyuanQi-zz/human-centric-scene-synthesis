//
// Created by siyuan on 2/19/17.
//

#ifndef CVPR2018_EVALUATOR_H
#define CVPR2018_EVALUATOR_H

// System header files
#include <iostream>
#include <string>
#include <algorithm>

// Boost header files
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

// Lib header files

// Local header files
#include "macros.h"
#include "Room.h"
#include "helper.h"

namespace FurnitureArranger {
    class Evaluator {
    public:
        Evaluator();
        Evaluator(std::string suncgRoot, std::string metadataPath, std::string workspacePath);

        void parse_line(const std::string &line, std::string &caption, ub::vector<double> &size, ub::vector<double> &translation, double &rotation);
        void read_sampled_txt(const fs::path &filename);
        bool load_evaluator();
        void save_evaluator();

        void collect_affordance();

    private:
        std::string _suncgRoot, _metadataPath, _workspacePath;

        std::vector<Room> _rooms;

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version);
    };
}


#endif //CVPR2018_EVALUATOR_H
