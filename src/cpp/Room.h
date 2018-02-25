//
// Created by siyuan on 1/19/17.
//

#ifndef CVPR2018_ROOM_H
#define CVPR2018_ROOM_H

// System header files
#include <fstream>
#include <unordered_map>
#include <thread>

#include <omp.h>

// Boost header files
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/thread.hpp>

// Lib header files
#include "lib/easylogging++.h"
#include "lib/json.hpp"

#include "lib/rrt/BiRRT.hpp"
#include "lib/rrt/2dplane/GridStateSpace.hpp"

#include "lib/polygon.h"

// Local header files
#include "macros.h"
#include "Entity.h"
#include "Furniture.h"

using json = nlohmann::json;
namespace fs = boost::filesystem;
namespace ub = boost::numeric::ublas;

namespace FurnitureArranger{
    class Room : public Entity {
    public:
        Room();
        Room(json node, std::string houseId);

        void add_furniture(Furniture furniture);
        void add_furniture(Furniture furniture, int supportingIndex);
        Furniture& get_furniture(size_t index);
        std::vector<Furniture>& get_furniture_list();

        void move_to_origin();
        void scale_to_meter_room(double scaleToMeters = 1);

        // Functions for collecting statistics
        static void read_group_metadata(std::string metadataPath);
        static json get_group_metadata();
        void find_groups();
        void collect_group_stats(json &groupStats);
        void set_groups(std::vector<std::pair<int, int>> groups);

        bool object_i_on_j(size_t i, size_t j) const;
        void find_supporting_objects();
        int get_supporting_index(size_t index) const;
        void get_furniture_wall_distance_angle(size_t index, double &distance, double &secondDistance, double &angle) const;

        std::vector<boost::numeric::ublas::vector<double>> get_affordance() const;

        // Helper functions for computing costs
        static void read_prior_data(std::string metadataPath);
        void initialize_prior_distributions();
        void sample_group_affordance();
        void sample_supporting_affordance();

        std::vector<std::vector<Eigen::Vector2f>> get_planning_heat_map(double &avgCost) const;
        void modify_obstacle(std::shared_ptr<RRT::GridStateSpace> stateSpace, const Furniture &furniture, bool block) const;
        std::vector<Eigen::Vector2f> plan_from_i_to_j(std::shared_ptr<RRT::GridStateSpace> stateSpace, size_t i, size_t j, double &cost) const;
        std::vector<std::vector<Eigen::Vector2f>> get_planning_trajectories(double &avgTrajCost) const;

        // Functions to compute costs
        double out_of_room_cost(ub::vector<double> vertexPos) const;
        void compute_planning_and_entropy_cost(double &avgPlanningCost, double &entropyCost) const;
        void compute_furn_wall_distance_orientation_cost(double &distCost, double &oriCost) const;
        double compute_group_cost() const;
        double compute_support_cost() const;
        double compute_affordance_cost() const;
        double compute_collision_cost() const;

        double compute_total_energy(ub::vector<double> weights) const;

        // Sampling functions
        void propose(double stepSize, size_t startFurnIndex, size_t endFurnIndex, bool proposeWallObj=true);

        // File IO
        std::string to_room_arranger_room();
        void write_to_room_arranger_file(std::string outputFilePath);
        void write_to_json(std::string outputFilePath);

    private:
        static json _groupMetadata;
        static json _furnWallDistPrior, _furnWallSecondDistPrior, _furnWallOriPrior;
        static std::unordered_map<std::string, cv::Mat> _affordanceDict;

        static const unsigned int _planningHeatMapIteration;
        static const unsigned int _RRTIteration, _gridSizeScale, _sideSegNum;
        static const double _RRTStepSize, _RRTMaxStepSize, _RRTGoalMaxDist, _noPathCost;

        std::vector<Furniture> _furnitureList;
        std::vector<std::pair<int, int>> _groups;
        std::vector<int> _supportingIndices;  // -1 stands for floor, -2 stands for wall
        std::vector<size_t> _ungroupedFurn;  // Furniture that are on the floor but not grouped

        int find_supporting_object(size_t index);

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<Entity>(*this);
            ar & _furnitureList;
            ar & _groups;
            ar & _supportingIndices;
        };
    };
}



#endif //CVPR2018_ROOM_H
