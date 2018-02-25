//
// Created by siyuan on 1/25/17.
//

#include "Learner.h"


namespace FurnitureArranger {
    Learner::Learner(){

    }

    Learner::Learner(std::string suncgRoot, std::string metadataPath, std::string workspacePath): _suncgRoot(suncgRoot), _metadataPath(metadataPath), _workspacePath(workspacePath){
        // Important: Read metadata
        Room::read_group_metadata(metadataPath);
        Furniture::read_metadata(metadataPath);

        if (!load_learner()){
            // Read all json files
            int house_count = 0;
            fs::directory_iterator it(fs::path(suncgRoot + "suncg_data/house/")), eod;
            BOOST_FOREACH(fs::path const &houseDir, std::make_pair(it, eod)){
                std::cout << fs::path(houseDir/"house.json") << std::endl;
                json house = read_SUNCG_file(fs::path(houseDir/"house.json"));
                std::vector<Room> rooms = parse_SUNCG_file(house);

                //// Write to Room arranger files
                //for (Room room : rooms){
                //    std::string folderName = house["id"].get<std::string>();
                //    //boost::algorithm::trim_if(folderName, boost::algorithm::is_any_of("\""));
                //    //std::string outputPath = workspacePath+"tmp/roomarranger/"+folderName+"/";
                //    std::string outputPath = workspacePath+"tmp/"+folderName+"/";
                //    boost::filesystem::create_directories(outputPath);
                //    room.write_to_room_arranger_file(outputPath+room.get_id()+".txt");
                //}

                _rooms.insert(std::end(_rooms), std::begin(rooms), std::end(rooms));
                house_count++;
                //if(house_count == 200){
                //    break;
                //}
            }

            save_learner();
        }
    }

    json Learner::read_SUNCG_file(const fs::path &filename) {
        if (fs::exists(filename)) {
            std::ifstream input_file(filename.string(), std::ios::in | std::ios::binary);
            std::ostringstream contents_stream;
            contents_stream << input_file.rdbuf();
            input_file.close();
            json house = json::parse(contents_stream.str());
            //CLOG(DEBUG, LOGGER_NAME) << house << std::endl;
            return house;
        } else {
            CLOG(ERROR, LOGGER_NAME) << "Cannot find file " + filename.string() << std::endl;
        }
    }

    std::vector<Room> Learner::parse_SUNCG_file(json house) {
        std::vector <Room> rooms;
        json levels = house["levels"];
        for (json level : levels) {
            for (json node : level["nodes"]) {
                if (node["type"] == "Room") {
                    if (node["roomTypes"].size() != 1 || node["nodeIndices"].size() < 5) {
                        continue;
                    }

                    try {
                        Room room(node, house["id"]);

                        for (int nodeIndex : node["nodeIndices"]) {
                            Furniture furniture(level["nodes"][nodeIndex]);
                            room.add_furniture(furniture);
                        }

                        // Process data
                        room.move_to_origin();
                        room.scale_to_meter_room(house["scaleToMeters"]);
                        room.find_groups();
                        room.find_supporting_objects();

                        rooms.push_back(room);
                    } catch (std::domain_error& e){
                        CLOG(WARNING, LOGGER_NAME) << "Error when parsing room " << node["id"] << " of house " << house["id"] << ": " << e.what();
                        continue;
                    }
                }
            }
            //for (json::iterator it = level.begin(); it != level.end(); ++it) {
            //    std::cout << it.key() << std::endl;
            //}
        }
        return rooms;
    }

    bool Learner::load_learner() {
        if (fs::exists(_metadataPath + "stats"))
        {
            std::ifstream ifs(_metadataPath + "stats");
            if (ifs.good()){
                boost::archive::text_iarchive ia(ifs);
                ia >> *this;
                ifs.close();
                return true;
            }else{
                return false;
            }
        }else{
            return false;
        }
    }

    void Learner::save_learner() {
        std::ofstream ofs(_metadataPath + "stats");
        boost::archive::text_oarchive oa(ofs);
        oa << *this;
        ofs.close();
    }

    std::vector<Room>& Learner::get_rooms(){
        return _rooms;
    }

    void Learner::collect_stats() {
        CLOG(INFO, LOGGER_NAME) << "==================== Collecting statistics ====================";
        //collect_room_stats();
        //collect_furniture_stats();
        //collect_furniture_wall_stats();
        //collect_group_stats();
        //collect_support_stats();
        //collect_affordance();
        //plot_planning_heat_maps();
        //collect_costs();
    }

    void Learner::collect_room_stats() {
        // Classify the rooms into different categories
        std::map<std::string, std::vector<Room>> classifiedRooms;
        for (Room room : _rooms){
            if (room.get_furniture_list().size()<5){
                continue;
            }
            std::string roomName = room.get_caption();
            if (classifiedRooms.find(roomName) == classifiedRooms.end()){
                classifiedRooms[roomName] = std::vector<Room>();
            }
            classifiedRooms[roomName].push_back(room);
        }

        // Collect the stats of coarse grained furniture class in each type of rooms
        json roomStats, roomMeta, roomSizes;
        for (auto roomMapIter = classifiedRooms.begin(); roomMapIter != classifiedRooms.end(); ++roomMapIter){
            // Find the furniture types in a type of room
            std::vector<std::string> furnitureTypes;
            for (Room room : roomMapIter->second){
                for (Furniture furniture: room.get_furniture_list()){
                    std::string furnitureType = furniture.get_caption_coarse();
                    if (std::find(furnitureTypes.begin(), furnitureTypes.end(), furnitureType) == furnitureTypes.end()){
                        furnitureTypes.push_back(furnitureType);
                    }
                }
                ub::vector<double> roomSize = room.get_size();
                roomSizes[roomMapIter->first].push_back(json::array({roomSize(0), roomSize(2)}));
            }

            // Collect the number of each type of furniture as samples
            std::map<std::string, std::vector<int>> furnitureCount;
            for (std::string furnitureType: furnitureTypes){
                furnitureCount[furnitureType] = std::vector<int>();
            }
            for (Room room : roomMapIter->second){
                // Initialize the number of each furniture to be 0
                for (auto furnCountIter = furnitureCount.begin(); furnCountIter != furnitureCount.end(); furnCountIter++){
                    furnCountIter->second.push_back(0);
                }

                for (Furniture furniture: room.get_furniture_list()){
                    furnitureCount[furniture.get_caption_coarse()].back() += 1;
                }
            }

            json furnitureStats(furnitureCount);
            roomMeta[roomMapIter->first] = (roomMapIter->second).size();

            roomStats[roomMapIter->first] = furnitureStats;
        }
        roomStats["meta"] = roomMeta;

        {
            // For each type of rooms, list the number samples of each type of furniture for the room type
            std::ofstream ofs(_metadataPath + "stats/roomStats.json");
            ofs << roomStats.dump(4);
            ofs.close();
        }

        {
            // Lists of sizes of different types of rooms
            std::ofstream ofs(_metadataPath + "stats/roomSizes.json");
            ofs << roomSizes.dump(4);
            ofs.close();
        }
    }

    void Learner::collect_furniture_stats() {
        json furnitureStats, furnitureSizes;

        // Classify all the funiture in all rooms into fine grained classes
        std::map<std::string, std::vector<Furniture>> classifiedFurniture;
        for (Room room : _rooms){
            for (Furniture furniture: room.get_furniture_list()){
                if (classifiedFurniture.find(furniture.get_caption()) == classifiedFurniture.end()){
                    classifiedFurniture[furniture.get_caption()] = std::vector<Furniture>();
                }
                classifiedFurniture[furniture.get_caption()].push_back(furniture);

                ub::vector<double> furnitureSize = furniture.get_size();
                furnitureSizes[furniture.get_caption()].push_back(json::array({furnitureSize(0), furnitureSize(1), furnitureSize(2)}));
            }
        }

        // Count the total number of coarse grained classes
        std::map<std::string, int> coarseClassCount;
        for (auto furnMapIter = classifiedFurniture.begin(); furnMapIter != classifiedFurniture.end(); ++furnMapIter){
            std::string coarseCaption = (furnMapIter->second).begin()->get_caption_coarse();
            if (coarseClassCount.find(coarseCaption) == coarseClassCount.end()){
                coarseClassCount[coarseCaption] = 0;
            }
            coarseClassCount[coarseCaption] += (furnMapIter->second).size();
        }

        // Compute the probability of each fine grained class in its coarse grained class
        for (auto furnMapIter = classifiedFurniture.begin(); furnMapIter != classifiedFurniture.end(); ++furnMapIter){
            std::string coarseCaption = (furnMapIter->second).begin()->get_caption_coarse();
            furnitureStats[coarseCaption][furnMapIter->first] = double((furnMapIter->second).size())/double(coarseClassCount[coarseCaption]);
        }

        {
            // The probability of each fine grained class in its coarse grained class of furniture
            std::ofstream ofs(_metadataPath + "stats/furnitureStats.json");
            ofs << furnitureStats.dump(4);
            ofs.close();
        }

        {
            // Lists of sizes of different fine grained class of furniture
            std::ofstream ofs(_metadataPath + "stats/furnitureSizes.json");
            ofs << furnitureSizes.dump(4);
            ofs.close();
        }
    }

    void Learner::collect_furniture_wall_stats() {
        json furnitureWallStats;
        for (Room room : _rooms){
            for (size_t i=0; i < room.get_furniture_list().size(); i++){
                double distance, secondDistance, angle;
                room.get_furniture_wall_distance_angle(i, distance, secondDistance, angle);
                furnitureWallStats[room.get_furniture(i).get_caption()].push_back(json::array({distance, secondDistance, angle}));
            }
        }

        {
            // Lists of distances and angles to wall of fine grained class of furniture
            std::ofstream ofs(_metadataPath + "stats/furnitureWallStats.json");
            ofs << furnitureWallStats.dump(4);
            ofs.close();
        }
    }

    void Learner::collect_group_stats() {
        json groupStats;

        for (Room room : _rooms) {
            room.collect_group_stats(groupStats);
        }

        {
            // Lists of distances and angles to wall of fine grained class of furniture
            std::ofstream ofs(_metadataPath + "stats/groupStats.json");
            ofs << groupStats.dump(4);
            ofs.close();
        }
    }

    void Learner::collect_support_stats() {
        json supportingStats;
        for (Room room : _rooms){
            for (size_t i=0; i < room.get_furniture_list().size(); i++){
                int supporting_index = room.get_supporting_index(i);
                std::string supportingObjectName, supportedObjectName = room.get_furniture(i).get_caption();
                if (supporting_index >= 0){
                    supportingObjectName = room.get_furniture(supporting_index).get_caption();
                }else if (supporting_index == -1){
                    supportingObjectName = "floor";
                }else if (supporting_index == -2){
                    supportingObjectName = "wall";
                }

                if (supportingStats[supportedObjectName].find(supportingObjectName) != supportingStats[supportedObjectName].end()) {
                    supportingStats[supportedObjectName][supportingObjectName] = int(supportingStats[supportedObjectName][supportingObjectName]) + 1;
                }else{
                    supportingStats[supportedObjectName][supportingObjectName] = 1;
                }
            }
        }

        // Normalize the statistics
        for (auto& supportedElement : supportingStats) {
            // Compute the total number of supporting objects
            double totalNum = 0;
            for (json::iterator it = supportedElement.begin(); it != supportedElement.end(); ++it) {
                totalNum += double(it.value());
            }

            // Normalize
            for (json::iterator it = supportedElement.begin(); it != supportedElement.end(); ++it) {
                it.value() = double(it.value())/totalNum;
            }
        }

        {
            // Lists of supporting objects for fine grained class of furniture
            std::ofstream ofs(_metadataPath + "stats/supportingStats.json");
            ofs << supportingStats.dump(4);
            ofs.close();
        }
    }

    void Learner::collect_affordance() {
        json furnitureAffordance;
        for (Room room : _rooms){
            std::vector<boost::numeric::ublas::vector<double>> affordance = room.get_affordance();

            for (Furniture const &furniture : room.get_furniture_list()){
                for (ub::vector<double> pos : affordance){
                    ub::vector<double> relPos = furniture.compute_relative_pos(pos);
                    furnitureAffordance[furniture.get_caption()].push_back(json::array({relPos(0), relPos(2)}));
                }
            }
        }

        {
            // Lists of affordance positions of fine grained class of furniture
            std::ofstream ofs(_metadataPath + "stats/furnitureAffordance.json");
            ofs << furnitureAffordance.dump(4);
            ofs.close();
        }
    }

    void Learner::plot_planning_heat_maps() {
        for (Room room : _rooms){
            // Plot planning heat map
            double avgCost = 0;
            std::vector<std::vector<Eigen::Vector2f>> trajectories = room.get_planning_heat_map(avgCost);

            // ===== Save the information for python plot =====
            std::string outputPath = _workspacePath + "tmp/heatmaps/";
            boost::filesystem::create_directories(outputPath);
            // Save trajectories to json
            json paths, heatMapInfo;
            for (std::vector<Eigen::Vector2f> trajectory : trajectories){
                json path;
                for (Eigen::Vector2f pathPoint : trajectory){
                    path.push_back(json::array({pathPoint.x(), pathPoint.y()}));
                }
                paths.push_back(path);
            }

            // Add room and furniture information
            heatMapInfo["trajectories"] = paths;
            heatMapInfo["roomSize"] = json::array({room.get_size()(0), room.get_size()(2)});
            for (Furniture const &furniture : room.get_furniture_list()){
                if (!furniture.blockable()){
                    continue;
                }

                // List of rectangle vertices
                json rectangle;
                for (int i = 0; i < 4; i++){
                    ub::vector<double> vertexCoor = furniture.get_vertex_coor(i);
                    rectangle.push_back(json::array({vertexCoor(0), vertexCoor(2)}));
                }
                heatMapInfo["furnitures"].push_back(rectangle);
            }

            // Write to file
            {
                // Lists of affordance positions of fine grained class of furniture
                std::ofstream ofs(outputPath+room.get_caption_coarse()+"_"+room.get_id()+".json");
                ofs << heatMapInfo.dump(4);
                ofs.close();
            }
        }
    }

    ub::vector<double> Learner::get_costs_for_room(Room room){
        ub::vector<double> costs;
        costs.resize(_costNum);
        double avgPlanningCost, entropyCost;
        room.compute_planning_and_entropy_cost(avgPlanningCost, entropyCost);
        costs(0) = avgPlanningCost;
        costs(1) = entropyCost;

        double distCost, oriCost;
        room.compute_furn_wall_distance_orientation_cost(distCost, oriCost);
        costs(2) = distCost;
        costs(3) = oriCost;

        double groupCost = room.compute_group_cost();
        double supportCost = room.compute_support_cost();
        costs(4) = groupCost;
        costs(5) = supportCost;

        double affordanceCost = room.compute_affordance_cost();
        double collisionCost = room.compute_collision_cost();
        costs(6) = affordanceCost;
        costs(7) = collisionCost;

        return costs;
    }

    void Learner::collect_costs(){
        Room::read_prior_data(_metadataPath);

        int roomCount = 0;
        //std::vector<double> avgPlanningCostVec, entropyCostVec, distCostVec, oriCostVec, groupCostVec, supportCostVec;
        std::vector<std::vector<double>> costVecs;
        for (int i =0; i < _costNum; i++){
            std::vector<double> cost_vec;
            costVecs.push_back(cost_vec);
        }
        for (Room room : _rooms){
            room.initialize_prior_distributions();
            room.sample_group_affordance();
            room.sample_supporting_affordance();

            ub::vector<double> costs = get_costs_for_room(room);
            for (unsigned int i =0; i < _costNum; i++){
                costVecs[i].push_back(costs(i));
            }

            roomCount++;
            //if (roomCount == 10){
            //    break;
            //}
        }
        std::cout << roomCount << std::endl;

        json avgCosts;
        for (std::vector<double> costVec : costVecs){
            double average = accumulate(costVec.begin(), costVec.end(), 0.0)/costVec.size();
            avgCosts.push_back(average);
            std::cout << average << std::endl;
        }

        {
            // Lists of affordance positions of fine grained class of furniture
            std::ofstream ofs(_metadataPath + "stats/avgCosts.json");
            ofs << avgCosts.dump(4);
            ofs.close();
        }
    }

    void Learner::learn_cost_weights() {
        // Learn weights for cost functions
        json avgCostsJson;
        read_json_file(_metadataPath + "stats/avgCosts.json", avgCostsJson);
        ub::vector<double> avgCosts = json_to_ub_vector(avgCostsJson);
        Room::read_group_metadata(_metadataPath);
        Room::read_prior_data(_metadataPath);
        Furniture::read_metadata(_metadataPath);

        double eta = 0.01, epsilon = 0.1, sampleRoomNum = 1;
        ub::vector<double> weights(_costNum), gradient(_costNum);
        for (unsigned int i = 0; i < _costNum; i++){
            weights(i) = 1;
            gradient(i) = 1;
        }
        std::cout << weights << std::endl;

        // Gradient descent by contrastive divergence
        ub::vector<double> sampleAvgCosts;
        while(norm_2(gradient) > epsilon){
            ub::vector<double> costs(_costNum);
            for (unsigned int i = 0; i < _costNum; i++){
                costs(i) = 0;
            }

            for (int i = 0; i < sampleRoomNum; i++){
                int roomIndex = int(rand() * _rooms.size() / (double)RAND_MAX);
                Room room = _rooms[roomIndex];
                room.initialize_prior_distributions();
                room.sample_group_affordance();
                room.sample_supporting_affordance();
                run_mcmc(room, 0, room.get_furniture_list().size(), weights);

                costs = costs + get_costs_for_room(room);
            }
            sampleAvgCosts = costs/sampleRoomNum;

            gradient = sampleAvgCosts - avgCosts;
            weights = weights + eta * gradient;
            //std::cout << avgCosts << sampleAvgCosts << gradient << std::endl;
            std::cout << norm_2(gradient) << " " << weights << std::endl;
        }

        json weightJson(weights);
        {
            // Lists of affordance positions of fine grained class of furniture
            std::ofstream ofs(_metadataPath + "costWeight.json");
            ofs << weightJson.dump(4);
            ofs.close();
        }
    }


    void Learner::run_mcmc(Room &room, size_t startFurnIndex, size_t endFurnIndex, ub::vector<double> costWeights) {
        //std::cout << "compute_total_energy" << std::endl;
        double _stepSize = 0.1;
        double currentEnergy = room.compute_total_energy(costWeights), proposedEnergy;

        for (int i = 0; i < _iterationMax; i++) {

            Room proposedRoom = room;
            proposedRoom.propose(_stepSize, startFurnIndex, endFurnIndex);
            proposedEnergy = proposedRoom.compute_total_energy(costWeights);

            //double r = (rand() / (double)RAND_MAX), acceptanceProb = std::min(1.0, exp(  (currentEnergy - proposedEnergy)/_temperature) );
            double r = (rand() / (double)RAND_MAX), acceptanceProb = std::min(1.0, exp(  currentEnergy - proposedEnergy));
            if (r < acceptanceProb){
                //std::cout << "Accepted: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
                room = proposedRoom;
                currentEnergy = proposedEnergy;
            }else{
                //std::cout << "Rejected: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
            }

            //// Output every iteration result to file for visualization
            //std::stringstream outputFilename;
            //outputFilename << "sample_";
            //outputFilename << std::setfill('0') << std::setw(4) << i;
            //outputFilename << ".json";
            //room.write_to_json(_workspacePath+"tmp/samples/json/"+outputFilename.str());
        }
    }

    void Learner::test_affordance_transform() {
        for (Room room : _rooms){
            for (Furniture const &furniture : room.get_furniture_list()){
                ub::vector<double> pos = furniture.get_vertex_coor(4);
                ub::vector<double> relPos, worldPos, newRelPos;
                relPos = room.get_furniture(0).compute_relative_pos(pos);
                worldPos = room.get_furniture(0).compute_world_pos(relPos);
                newRelPos = room.get_furniture(0).compute_relative_pos(worldPos);
                std::cout << furniture.get_vertex_coor(4) << " " << relPos << " " << worldPos << " " << newRelPos << std::endl;
            }
            break;
        }
    }

    template<class Archive>
    void Learner::serialize(Archive & ar, const unsigned int version)
    {
        ar & _suncgRoot;
        ar & _metadataPath;
        ar & _workspacePath;
        ar & _rooms;
    }
}