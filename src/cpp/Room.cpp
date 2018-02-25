//
// Created by siyuan on 1/19/17.
//

#include "Room.h"

namespace FurnitureArranger {
    // Metadata and prior data
    json Room::_groupMetadata;
    json Room::_furnWallDistPrior, Room::_furnWallSecondDistPrior, Room::_furnWallOriPrior;
    std::unordered_map<std::string, cv::Mat> Room::_affordanceDict;

    // Parameters for RRT planning
    const unsigned int Room::_planningHeatMapIteration = 1;
    const unsigned int Room::_RRTIteration = 200, Room::_gridSizeScale = 100, Room::_sideSegNum = 500;
    const double Room::_RRTStepSize = 0.05, Room::_RRTMaxStepSize = 0.1, Room::_RRTGoalMaxDist = 0.1, Room::_noPathCost = 10.0;

    Room::Room() {
    }

    Room::Room(json node, std::string houseId) {
        _id = node["id"];
        _modelID = node["modelId"];
        _caption = node["roomTypes"][0];
        _captionCoarse = houseId;

        _size.resize(3);
        for (size_t i = 0; i < 3; i++) {
            _size(i) = double(node["bbox"]["max"][i]) - double(node["bbox"]["min"][i]);
        }

        _transformMatrix = ub::identity_matrix<double>(4);
        for (size_t i = 0; i < 3; i++) {
            _transformMatrix(i, 3) = double(node["bbox"]["min"][i]);
        }
        transformMatrixToRT();
    }

    void Room::add_furniture(Furniture furniture) {
        _furnitureList.push_back(furniture);
    }

    void Room::add_furniture(Furniture furniture, int supportingIndex) {
        _furnitureList.push_back(furniture);
        _supportingIndices.push_back(supportingIndex);
    }

    Furniture& Room::get_furniture(size_t index) {
        return _furnitureList[index];
    }

    std::vector<Furniture>& Room::get_furniture_list() {
        return _furnitureList;
    }

    void Room::move_to_origin() {
        for (std::vector<Furniture>::iterator it = _furnitureList.begin(); it != _furnitureList.end(); ++it) {
            it->translate(-_translation);
        }

        _translation = ub::zero_vector<double>(3);
        RTTotransformMatrix();
    }

    void Room::scale_to_meter_room(double scaleToMeters) {
        scale_to_meter(scaleToMeters);
        for (std::vector<Furniture>::iterator it = _furnitureList.begin(); it != _furnitureList.end(); ++it) {
            it->scale_to_meter(scaleToMeters);
        }
    }

    void Room::read_group_metadata(std::string metadataPath) {
        read_json_file(metadataPath + "groups.json", _groupMetadata);
    }

    json Room::get_group_metadata() {
        return _groupMetadata;
    }

    void Room::find_groups() {
        for (auto &element : _groupMetadata) {
            for (json::iterator coreIt = element.begin(); coreIt != element.end(); ++coreIt) {
                for (std::string otherElement : coreIt.value()) {
                    // Loop through coreObjName, otherObjName pair: coreIt.key() and otherElement

                    // Search in the furniture for the pair
                    for (size_t coreI = 0; coreI < _furnitureList.size(); coreI++) {
                        if (_furnitureList[coreI].get_caption() == coreIt.key()) {
                            for (size_t otherI = 0; otherI < _furnitureList.size(); otherI++) {
                                if (_furnitureList[otherI].get_caption() == otherElement) {
                                    ub::vector<double> corePos = _furnitureList[coreI].get_vertex_coor(4), otherPos = _furnitureList[otherI].get_vertex_coor(4);
                                    if (norm_2(corePos - otherPos) > 2) {
                                        continue;
                                    }
                                    _groups.push_back(std::make_pair(coreI, otherI));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void Room::collect_group_stats(json &groupStats) {
        for (auto &element : _groupMetadata) {
            for (json::iterator coreIt = element.begin(); coreIt != element.end(); ++coreIt) {
                for (std::string otherElement : coreIt.value()) {
                    // Loop through coreObjName, otherObjName pair: coreIt.key() and otherElement

                    // Search in the furniture for the pair
                    for (Furniture coreFurn : _furnitureList) {
                        if (coreFurn.get_caption() == coreIt.key()) {
                            int otherOccur = 0;
                            for (Furniture otherFurn : _furnitureList) {
                                if (otherFurn.get_caption() == otherElement) {
                                    ub::vector<double> corePos = coreFurn.get_vertex_coor(4), otherPos = otherFurn.get_vertex_coor(4);
                                    if (norm_2(corePos - otherPos) > 2) {
                                        continue;
                                    }
                                    otherOccur++;
                                }
                            }
                            groupStats[coreIt.key()][otherElement].push_back(otherOccur);
                        }
                    }
                }
            }
        }
    }

    void Room::set_groups(std::vector<std::pair<int, int>> groups) {
        _groups = groups;

        std::vector<int> groupedFurn;
        for (std::pair<int, int> pair : groups){
            groupedFurn.push_back(pair.first);
            groupedFurn.push_back(pair.second);
        }

        for (size_t i = 0; i < _furnitureList.size(); i++){
            if (_supportingIndices[i] != -1){
                continue;
            }
            if (std::find(groupedFurn.begin(), groupedFurn.end(), i) != groupedFurn.end()){
                continue;
            }

            CLOG(INFO, LOGGER_NAME) << "Ungrouped furniture: " << _furnitureList[i].get_caption();
            _ungroupedFurn.push_back(i);
            _furnitureList[i].sample_affordance();
        }
    }

    bool Room::object_i_on_j(size_t i, size_t j) const {
        double distance = _furnitureList[i].get_translation()(2) -
                          (_furnitureList[j].get_translation()(2) + _furnitureList[j].get_size()(2));
        if (distance > 0.1) {
            // The vertical distance between the bottom surface of object i
            // and top surface of object j is too far
            return false;
        }

        std::vector<boost::numeric::ublas::vector<double>> poly;
        for (int vertexJ = 0; vertexJ < 4; vertexJ++) {
            poly.push_back(_furnitureList[j].get_vertex_coor(vertexJ));
        }
        for (int vertexI = 0; vertexI < 4; vertexI++) {
            if (!point_in_polygon(_furnitureList[i].get_vertex_coor(vertexI), poly)) {
                return false;
            }
        }
        return true;
    }

    int Room::find_supporting_object(size_t index) {
        // Find the supporting object index for the input furniture
        // -1 stands for floor, -2 stands for wall

        if (_furnitureList[index].get_translation()(1) < 0.2) {
            // This is a supporting object itself
            return -1;
        }

        for (size_t i = 0; i < _furnitureList.size(); i++) {
            if (i == index) {
                continue;
            }

            if (object_i_on_j(index, i)) {
                return int(i);
            }
        }

        return -2;
    }

    void Room::find_supporting_objects() {
        for (size_t i = 0; i < _furnitureList.size(); i++) {
            _supportingIndices.push_back(find_supporting_object(i));
        }
    }

    int Room::get_supporting_index(size_t index) const {
        return _supportingIndices[index];
    }

    void Room::get_furniture_wall_distance_angle(size_t index, double &distance, double &secondDistance, double &angle) const {
        double minDistance = std::numeric_limits<double>::infinity(), secondMinDistance = std::numeric_limits<double>::infinity();
        size_t nearestWallIndex = 4;
        for (int i = 0; i < 4; i++) {
            ub::vector<double> cornerPos = _furnitureList[index].get_vertex_coor(i);
            std::vector<double> distances = get_point_to_edge_distance(cornerPos);
            for (size_t iDist = 0; iDist < distances.size(); iDist++) {
                if (distances[iDist] < minDistance) {
                    if (iDist != nearestWallIndex) {
                        secondMinDistance = minDistance;
                    }
                    minDistance = distances[iDist];
                    nearestWallIndex = iDist;
                } else if (distances[iDist] < secondMinDistance && iDist != nearestWallIndex) {
                    secondMinDistance = distances[iDist];
                }
            }
        }

        distance = minDistance;
        secondDistance = secondMinDistance;
        angle = _furnitureList[index].get_rotation() - nearestWallIndex * (M_PI / 2.0);
        angle = angle - floor(angle / (2 * M_PI)) * (2 * M_PI);
    }

    std::vector<boost::numeric::ublas::vector<double>> Room::get_affordance() const {
        std::vector<boost::numeric::ublas::vector<double>> affordance;
        for (Furniture const &furniture : _furnitureList) {
            std::string furnClass = furniture.get_caption_coarse();
            if (furnClass == "chair" || furnClass == "sofa" || furnClass == "bed" || furnClass == "person") {
                ub::vector<double> center = furniture.get_vertex_coor(4);
                affordance.push_back(center);
            }
        }
        return affordance;
    }

    void Room::read_prior_data(std::string metadataPath) {
        read_json_file(metadataPath + "prior/furnWallDist.json", _furnWallDistPrior);
        read_json_file(metadataPath + "prior/furnWallSecondDist.json", _furnWallSecondDistPrior);
        read_json_file(metadataPath + "prior/furnWallOri.json", _furnWallOriPrior);

        fs::directory_iterator it(fs::path(metadataPath + "prior/affordance/")), eod;
        BOOST_FOREACH(fs::path const &affordanceMap, std::make_pair(it, eod)) {
            cv::Mat amap;
            cv::FileStorage fs(affordanceMap.string(), cv::FileStorage::READ);
            fs[affordanceMap.stem().string()] >> amap;
            _affordanceDict[affordanceMap.stem().string()] = amap;
            fs.release();
        }
    }

    void Room::initialize_prior_distributions() {
        for (Furniture &furniture : _furnitureList) {
            furniture.initialize_prior_distributions(_furnWallDistPrior[furniture.get_caption()], _furnWallSecondDistPrior[furniture.get_caption()], _furnWallOriPrior[furniture.get_caption()]);
        }

        for (Furniture &furniture : _furnitureList){
            furniture.set_affordance_map(_affordanceDict[furniture.get_caption()]);
        }
    }

    void Room::sample_group_affordance() {
        for (std::pair<int, int> group : _groups){
            _furnitureList[group.second].sample_affordance();
        }
    }

    void Room::sample_supporting_affordance() {
        for (size_t i = 0; i < _furnitureList.size(); i++){
            if (_supportingIndices[i] >= 0){
                _furnitureList[i].sample_affordance();
            }
        }
    }

    std::vector<std::vector<Eigen::Vector2f>> Room::get_planning_heat_map(double &avgCost) const {
        std::vector<std::vector<Eigen::Vector2f>> trajectories;
        double totalCost = 0;
        for (unsigned int i = 0; i < _planningHeatMapIteration; i++) {
            double avgTrajCost;
            std::vector<std::vector<Eigen::Vector2f>> newTrajectories = get_planning_trajectories(avgTrajCost);
            trajectories.insert(trajectories.end(), newTrajectories.begin(), newTrajectories.end());
            totalCost += avgTrajCost;
        }
        avgCost = totalCost / double(_planningHeatMapIteration);

        return trajectories;
    }

    // Helper functions for computing costs
    void Room::modify_obstacle(std::shared_ptr<RRT::GridStateSpace> stateSpace, const Furniture &furniture,
                               bool block = true) const {
        int discretizedWidth = stateSpace->obstacleGrid().discretizedWidth(),
                discretizedHeight = stateSpace->obstacleGrid().discretizedHeight();

        for (int vertexIndex = 0; vertexIndex < 4; vertexIndex++) {
            ub::vector<double> sideStart = furniture.get_vertex_coor(vertexIndex), sideEnd = furniture.get_vertex_coor(
                    (vertexIndex + 1) % 4);
            double startX = sideStart(0), startY = sideStart(2), endX = sideEnd(0), endY = sideEnd(2);
            double stepX = (endX - startX) / _sideSegNum, stepY = (endY - startY) / _sideSegNum;

            for (int segIndex = 0; segIndex < _sideSegNum; segIndex++) {
                // Add obstacles
                Eigen::Vector2i gridLoc = stateSpace->obstacleGrid().gridSquareForLocation(
                        Eigen::Vector2f(startX, startY));
                if (gridLoc.x() >= 0 && gridLoc.x() < discretizedWidth && gridLoc.y() >= 0 &&
                    gridLoc.y() < discretizedHeight) {
                    stateSpace->obstacleGrid().obstacleAt(gridLoc) = block;
                }
                startX += stepX;
                startY += stepY;
            }


            Eigen::Vector2i gridLoc = stateSpace->obstacleGrid().gridSquareForLocation(Eigen::Vector2f(endX, endY));
            if (gridLoc.x() >= 0 && gridLoc.x() < discretizedWidth && gridLoc.y() >= 0 &&
                gridLoc.y() < discretizedHeight) {
                stateSpace->obstacleGrid().obstacleAt(gridLoc) = block;
            }
        }
    }

    std::vector<Eigen::Vector2f> Room::plan_from_i_to_j(std::shared_ptr<RRT::GridStateSpace> stateSpace, size_t i, size_t j, double &cost) const {
        RRT::BiRRT<Eigen::Vector2f> biRRT(stateSpace);

        // Setup biRRT
        ub::vector<double> iCenter = _furnitureList[i].get_vertex_coor(4), jCenter = _furnitureList[j].get_vertex_coor(
                4);

        biRRT.setStartState(Eigen::Vector2f(iCenter(0), iCenter(2)));
        biRRT.setGoalState(Eigen::Vector2f(jCenter(0), jCenter(2)));
        biRRT.setStepSize(_RRTStepSize);
        biRRT.setMaxStepSize(_RRTMaxStepSize);
        biRRT.setGoalMaxDist(_RRTGoalMaxDist);
        biRRT.setMaxIterations(_RRTIteration);

        std::vector<Eigen::Vector2f> rrtSolution;
        modify_obstacle(stateSpace, _furnitureList[i], false);
        modify_obstacle(stateSpace, _furnitureList[j], false);

        // Grow biRRT to find solution
        if (biRRT.run()) {
            biRRT.getPath(rrtSolution);
        }

        modify_obstacle(stateSpace, _furnitureList[i], true);
        modify_obstacle(stateSpace, _furnitureList[j], true);

        if (rrtSolution.size() > 0) {
            cost = rrtSolution.size() * 0.05;
        } else {
            cost = _noPathCost;
        }
        return rrtSolution;
    }

    std::vector<std::vector<Eigen::Vector2f>> Room::get_planning_trajectories(double &avgTrajCost) const {
        avgTrajCost = 0;

        // Initialize state space
        std::shared_ptr<RRT::GridStateSpace> stateSpace = std::make_shared<RRT::GridStateSpace>(_size(0), _size(2), _size(0) * _gridSizeScale, _size(2) * _gridSizeScale);

        // Add obstacles
        for (Furniture const &furniture : _furnitureList) {
            if (!furniture.blockable()) {
                continue;
            }
            modify_obstacle(stateSpace, furniture, true);
        }

        std::vector<std::vector<Eigen::Vector2f>> trajectories;
        for (size_t i = 0; i < _furnitureList.size(); i++) {
            if (!_furnitureList[i].blockable()) {
                continue;
            }
            ub::vector<double> iPos = _furnitureList[i].get_vertex_coor(4);
            if (iPos(0) < 0 || iPos(0) >= _size(0) || iPos(2) < 0 || iPos(2) >= _size(2)) {
                // The furniture is out of the room
                continue;
            }

            for (size_t j = 0; j < _furnitureList.size(); j++) {
                if (i == j) {
                    // Same object
                    continue;
                }
                if (!_furnitureList[j].blockable()) {
                    // Blockable furniture
                    continue;
                }
                ub::vector<double> jPos = _furnitureList[j].get_vertex_coor(4);
                if (jPos(0) < 0 || jPos(0) >= _size(0) || jPos(2) < 0 || jPos(2) >= _size(2)) {
                    // The furniture is out of the room
                    continue;
                }

                double cost;
                trajectories.push_back(plan_from_i_to_j(stateSpace, i, j, cost));
                avgTrajCost += cost;
            }
        }

        avgTrajCost /= double(trajectories.size());
        return trajectories;
    }


    // ==================== Functions to compute costs ====================

    double Room::out_of_room_cost(ub::vector<double> vertexPos) const {
        double cost = 0.0;
        const double penatyScale = 4.0;
        if (vertexPos(0) < 0){
            cost += penatyScale*std::abs(vertexPos(0));
        }
        if (vertexPos(0) > _size(0)){
            cost += penatyScale*std::abs((vertexPos(0) - _size(0)));
        }
        if (vertexPos(2) < 0){
            cost += penatyScale*std::abs(vertexPos(2));
        }
        if (vertexPos(2) > _size(2)){
            cost += penatyScale*std::abs((vertexPos(2) - _size(2)));
        }
        return cost;
    }

    void Room::compute_planning_and_entropy_cost(double &avgPlanningCost, double &entropyCost) const {
        std::vector<std::vector<Eigen::Vector2f>> trajecotries = get_planning_heat_map(avgPlanningCost);
        avgPlanningCost = 0;

        entropyCost = 0;
        if (trajecotries.size() > 0){
            unsigned int scale = 1, gridPlaneWidth = unsigned(_size(0)*scale)+1, gridPlaneHeight = unsigned(_size(0)*scale)+1;
            ub::matrix<double> probMap = ub::zero_matrix<double>(gridPlaneWidth, gridPlaneHeight);

            for (std::vector<Eigen::Vector2f> trajectory : trajecotries){
                for (Eigen::Vector2f point : trajectory){
                    unsigned x = std::min(unsigned(point.x()*scale), gridPlaneWidth-1), y = std::min(unsigned(point.y()*scale), gridPlaneHeight-1);
                    probMap(x, y) += 1;
                }
            }

            probMap /= sum(prod(ub::scalar_vector<double>(probMap.size1(), 1.0), probMap));

            ub::matrix<double> logProbMap = probMap;
            for (unsigned i = 0; i < logProbMap.size1(); ++i){
                for (unsigned j = 0; j < logProbMap.size2(); ++j){
                    if (probMap(i, j) > 0){
                        logProbMap(i, j) = log(logProbMap(i, j));
                    }
                }
            }

            ub::matrix<double> entropy = element_prod(probMap, logProbMap);
            entropyCost = -sum(prod(ub::scalar_vector<double>(entropy.size1(), 1.0), entropy));
        }
    }

    void Room::compute_furn_wall_distance_orientation_cost(double &distCost, double &oriCost) const {
        distCost = 0;
        oriCost = 0;
        for (size_t i = 0; i < _furnitureList.size(); i++) {
            for (size_t iVertex = 0; iVertex < 4; iVertex++){
                ub::vector<double> vertexPos = _furnitureList[i].get_vertex_coor(iVertex);
                distCost += out_of_room_cost(vertexPos);
            }

            if (_supportingIndices[i] == -1) {
                double distance, secondDistance, angle;
                get_furniture_wall_distance_angle(i, distance, secondDistance, angle);

                // Cost is negative log probability
                distCost -= _furnitureList[i].compute_distance_log_prob(distance);
                distCost -= _furnitureList[i].compute_second_distance_log_prob(secondDistance);
                oriCost -= _furnitureList[i].compute_orientation_log_prob(angle);
            }

            if (_supportingIndices[i] == -2){
                double distance, secondDistance, angle;
                get_furniture_wall_distance_angle(i, distance, secondDistance, angle);
                distCost -= 5 * _furnitureList[i].compute_distance_log_prob(distance);
            }
        }

        distCost /= _furnitureList.size();
        oriCost /= _furnitureList.size();
    }

    double Room::compute_group_cost() const {
        double groupCost = 0.0;
        for (std::pair<int, int> group : _groups){
            std::vector<boost::numeric::ublas::vector<double>> humanPositions = _furnitureList[group.second].get_human_positions();
            groupCost -= _furnitureList[group.first].compute_human_position_log_prob(humanPositions, _furnitureList[group.second].get_sampled_human_probs());
        }

        if (_groups.size()>0){
            return groupCost/_groups.size();
        }else{
            return 0.0;
        }
    }

    double Room::compute_support_cost() const {
        double supportCost = 0.0;
        int supportedObjCount = 0;
        for (size_t i = 0; i < _furnitureList.size(); i++){
            if (_supportingIndices[i] == -2)
            {
                double minDistance = std::numeric_limits<double>::infinity();
                for (size_t iVertex = 0; iVertex < 4; iVertex++){
                    std::vector<double> distancesToWall = get_point_to_edge_distance(_furnitureList[i].get_vertex_coor(iVertex));
                    for (double d : distancesToWall){
                        if (d < minDistance){
                            minDistance = d;
                        }
                    }
                }
                supportCost += minDistance;

                // Force the wall objects to be facing the room
                double distance, secondDistance, angle;
                get_furniture_wall_distance_angle(i, distance, secondDistance, angle);
                supportCost += std::abs(angle);

                supportedObjCount++;
            }
            else if (_supportingIndices[i] >= 0)
            {
                // Check if the supporting relation holds
                if (!object_i_on_j(i, _supportingIndices[i])){
                    supportCost += 100*norm_2(_furnitureList[i].get_vertex_coor(4) - _furnitureList[_supportingIndices[i]].get_vertex_coor(4));
                }

                // Compute the affordance cost
                std::vector<boost::numeric::ublas::vector<double>> humanPositions = _furnitureList[i].get_human_positions();
                supportCost -= _furnitureList[_supportingIndices[i]].compute_human_position_log_prob(humanPositions, _furnitureList[i].get_sampled_human_probs());
                supportedObjCount++;
            }
        }

        if (supportedObjCount > 0){
            return supportCost/supportedObjCount;
        }else{
            return 0.0;
        }
    }

    double Room::compute_affordance_cost() const {
        double affCost = 0.0;
        return affCost;
        //for (size_t i : _ungroupedFurn){
        //    ub::vector<double> center = _furnitureList[i].get_vertex_coor(4);
        //
        //    for (size_t j = 0; j < _furnitureList.size(); j++){
        //        if (i == j || _supportingIndices[j] != -1){
        //            continue;
        //        }
        //
        //        std::vector<boost::numeric::ublas::vector<double>> poly;
        //        for (int iVertex = 0; iVertex < 4; iVertex++){
        //            poly.push_back(_furnitureList[j].get_vertex_coor(iVertex));
        //        }
        //        if (point_in_polygon(center, poly)){
        //            affCost += _furnitureList[j].compute_human_position_log_prob(center);
        //            break;
        //        }
        //    }
        //}
        //return affCost;
    }

    double Room::compute_collision_cost() const {
        double collisionCost = 0.0;
        for (size_t i = 0; i < _furnitureList.size(); i++){
            if (_supportingIndices[i] != -1){
                continue;
            }

            double izStart = _furnitureList[i].get_translation()(1), izEnd = izStart + _furnitureList[i].get_size()(1);

            Polygon_2 P;
            ub::vector<double> centerP = _furnitureList[i].get_vertex_coor(4);
            for (int iVertex = 0; iVertex < 4; iVertex++){
                ub::vector<double> vertex = _furnitureList[i].get_vertex_coor(iVertex);
                vertex = 1.2*(vertex-centerP) + centerP;
                P.push_back (Point_2 (vertex(0), vertex(2)));
            }

            for (size_t j = i+1; j < _furnitureList.size(); j++){
                if (_supportingIndices[j] != -1 || i == j){
                    continue;
                }
                if (i == j){
                    continue;
                }

                double jzStart = _furnitureList[j].get_translation()(1), jzEnd = jzStart + _furnitureList[j].get_size()(1);
                double zOverlap = std::min(izEnd, jzEnd) - std::max(izStart, jzStart);
                //std::cout << _furnitureList[i].get_caption() << " " << _furnitureList[j].get_caption() << " " << std::min(izEnd, jzEnd) << " " << std::max(izStart, jzStart) << " " << zOverlap << std::endl;
                if (zOverlap <= 0){
                    continue;
                }

                Polygon_2 Q;
                ub::vector<double> centerQ = _furnitureList[j].get_vertex_coor(4);
                for (int iVertex = 0; iVertex < 4; iVertex++){
                    ub::vector<double> vertex = _furnitureList[j].get_vertex_coor(iVertex);
                    vertex = 1.2*(vertex-centerQ) + centerQ;
                    Q.push_back (Point_2 (vertex(0), vertex(2)));
                }

                collisionCost += compute_intersection(P, Q)*zOverlap;
            }
        }

        //return collisionCost/(double)_furnitureList.size();
        return collisionCost;
    }

    double Room::compute_total_energy(ub::vector<double> weights) const {
        double avgPlanningCost = 0.0, entropyCost = 0.0;
        //compute_planning_and_entropy_cost(avgPlanningCost, entropyCost);

        double distCost = 0.0, oriCost = 0.0;
        compute_furn_wall_distance_orientation_cost(distCost, oriCost);

        //double groupCost = 0, supportCost = 0;
        double groupCost = compute_group_cost();
        double supportCost = compute_support_cost();
        double affordanceCost = compute_affordance_cost();
        double collisionCost = compute_collision_cost();

        //std::cout << avgPlanningCost << " " << entropyCost << " " << distCost << " " << oriCost << " " << groupCost << " " << supportCost << " " << affordanceCost << " " << collisionCost << std::endl;


        //// Multi-threading attempt 1
        //double avgPlanningCost, entropyCost;
        ////boost::thread planThread(&Room::compute_planning_and_entropy_cost, this, boost::ref(avgPlanningCost), boost::ref(entropyCost));
        //std::thread planThread(&Room::compute_planning_and_entropy_cost, this, std::ref(avgPlanningCost), std::ref(entropyCost));
        //
        //double distCost = 0.0, oriCost = 0.0;
        //compute_furn_wall_distance_orientation_cost(distCost, oriCost);
        //
        //double groupCost = compute_group_cost();
        //double supportCost = compute_support_cost();
        //double affordanceCost = compute_affordance_cost();
        //double collisionCost = compute_collision_cost();
        //
        //planThread.join();


        //// Multi-threading attempt 2
        //double avgPlanningCost, entropyCost, distCost, oriCost, groupCost, supportCost, affordanceCost, collisionCost;
        //
        //boost::thread planThread(&Room::compute_planning_and_entropy_cost, this, boost::ref(avgPlanningCost), boost::ref(entropyCost));
        //boost::thread distOriThread(&Room::compute_furn_wall_distance_orientation_cost, this, boost::ref(distCost), boost::ref(oriCost));
        //boost::thread groupThread([&]{groupCost = compute_group_cost();});
        //boost::thread supportThread([&]{supportCost = compute_support_cost();});
        //boost::thread affordanceThread([&]{affordanceCost = compute_affordance_cost();});
        //boost::thread collisionThread([&]{collisionCost = compute_collision_cost();});
        //
        //planThread.join();
        //distOriThread.join();
        //groupThread.join();
        //supportThread.join();
        //affordanceThread.join();
        //collisionThread.join();


        double totalEnergy = 0.0;
        std::vector<double> stdcosts = {avgPlanningCost, entropyCost, distCost, oriCost, groupCost, supportCost, affordanceCost, collisionCost};
        ub::vector<double> costVec(8);
        std::copy(stdcosts.begin(), stdcosts.end(), costVec.begin());
        totalEnergy = inner_prod(weights, costVec);

        return totalEnergy;
    }


    // ==================== Sampling functions ====================

    void Room::propose(double stepSize, size_t startFurnIndex, size_t endFurnIndex, bool proposeWallObj) {
        // Randomly select an object in the selected range
        size_t furnitureIndex = rand()%(endFurnIndex-startFurnIndex)+startFurnIndex;
        Furniture &furniture = _furnitureList[furnitureIndex];
        if (!proposeWallObj and get_supporting_index(furnitureIndex)==-2){
            return;
        }

        double r = (rand() / (double)RAND_MAX);
        if (r < 0.5){
            ub::vector<double> translation = ub::zero_vector<double>(3);
            translation(0) = stepSize*2*(rand()/(double)RAND_MAX - 0.5);
            translation(2) = stepSize*2*(rand()/(double)RAND_MAX - 0.5);
            furniture.translate(translation);
        }else{
            furniture.rotate(stepSize*M_PI*2*(rand()/(double)RAND_MAX - 0.5));
        }
    }


    // ==================== File IO ====================

    std::string Room::to_room_arranger_room() {
        std::stringstream ss;
        ss << "Caption;ID;Size (WxLxH);Position (X,Y,Z);Rotation;Price" << std::endl;
        ss << to_room_arranger_object() << std::endl;
        int objectID = 0;
        for (Furniture const &furniture : _furnitureList) {
            objectID++;
            ss << furniture.to_room_arranger_object(std::to_string(objectID)) << std::endl;
        }
        return ss.str();
    }

    void Room::write_to_room_arranger_file(std::string outputFilePath) {
        std::ofstream writeFile;
        writeFile.open(outputFilePath, std::ios::out | std::ios::trunc);
        writeFile << to_room_arranger_room();
        writeFile.close();
    }

    void Room::write_to_json(std::string outputFilePath) {
        json sample;
        sample["roomSize"] = json::array({_size(0), _size(1), _size(2)});

        for (Furniture const &furniture : _furnitureList){
            // List of rectangle vertices
            json rectangle;
            for (int i = 0; i < 4; i++){
                ub::vector<double> vertexCoor = furniture.get_vertex_coor(i);
                rectangle.push_back(json::array({vertexCoor(0), vertexCoor(1), vertexCoor(2)}));
            }
            sample["furnitures"][furniture.get_caption()].push_back(rectangle);
        }

        for (std::pair<int, int> pair : _groups){
            std::vector<boost::numeric::ublas::vector<double>> humanPositions = _furnitureList[pair.second].get_human_positions();
            for (ub::vector<double> humanPosition : humanPositions){
                sample["affordance"][_furnitureList[pair.second].get_caption()].push_back(json::array({humanPosition(0), humanPosition(1), humanPosition(2)}));
            }
        }

        {
            // Lists of sizes of different types of rooms
            std::ofstream ofs(outputFilePath);
            ofs << sample.dump(4);
            ofs.close();
        }
    }
};