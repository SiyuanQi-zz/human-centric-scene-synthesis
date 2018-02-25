//
// Created by siyuan on 1/19/17.
//

#ifndef CVPR2018_FURNITURE_H
#define CVPR2018_FURNITURE_H

// System header files
#include <cmath>
#include <cassert>

// Boost header files
#include <boost/serialization/base_object.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include "lib/json.hpp"
#include "helper.h"
#include "SUNCGMetadata.h"
#include "Entity.h"

using json = nlohmann::json;
namespace ub = boost::numeric::ublas;
namespace bm = boost::math;

namespace FurnitureArranger{
    class Furniture : public Entity {
    public:
        Furniture();
        Furniture(json node);

        static void read_metadata(std::string metadataPath);
        void set_caption(std::string caption);

        bool blockable() const;

        void initialize_prior_distributions(json distParams, json secondDistParams, json oriParams);
        void set_affordance_map(const cv::Mat &inMap);
        double map_to_relative_coor(int gridCoor) const;
        int relative_coor_to_map(double relPos) const;
        void sample_affordance();
        std::vector<ub::vector<double>> get_human_positions() const;
        std::vector<double> get_sampled_human_probs() const;

        double compute_distance_log_prob(const double &distance) const;
        double compute_second_distance_log_prob(const double &distance) const;
        double compute_orientation_log_prob(const double &orientation) const;
        double compute_human_position_log_prob(const std::vector<ub::vector<double>> &worldPositions, std::vector<double> humanProbs) const;

    private:
        // Metadata
        static SUNCGMetadata _metadata;

        static const double _maxSupportingOjbectFloorDistance, _minBlockableHeight;
        static const int _affordanceBins, _humanSampleCount;
        static const double _affDistanceLimit;

        // Prior distributions
        bm::lognormal_distribution<double> _distanceDist, _secondDistanceDist;
        double _distanceDistLoc, _secondDistanceDistLoc;
        std::vector<std::vector<double>> _orientationDistParams;
        std::vector<double> _affordanceMap, _accumAffordanceMap;

        std::vector<ub::vector<double>> _humanPositions;
        std::vector<double> _humanProbs;

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<Entity>(*this);
        };
    };
}


#endif //CVPR2018_FURNITURE_H
