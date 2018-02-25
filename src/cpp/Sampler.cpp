//
// Created by siyuan on 6/20/17.
//

#include "Sampler.h"


namespace FurnitureArranger{
    int Sampler::_iterationMax = 5000;

    Sampler::Sampler() {

    }

    Sampler::Sampler(Learner learner, std::string suncgRoot, std::string metadataPath, std::string workspacePath): _suncgRoot(suncgRoot), _metadataPath(metadataPath), _workspacePath(workspacePath){
        _rooms = learner.get_rooms();
        std::random_shuffle(_rooms.begin(), _rooms.end());
        Room::read_group_metadata(metadataPath);
        Room::read_prior_data(metadataPath);
        Furniture::read_metadata(metadataPath);

        json weight;
        read_json_file(_metadataPath + "costWeight.json", weight);
        _costWeights = json_to_ub_vector(weight);
    }

    void Sampler::arrange(int process_id) {
        std::cout << process_id << std::endl;
        int sample_interval = 1000, sample_index = sample_interval * process_id;

        std::vector<Room>::const_iterator first = _rooms.begin() + sample_index;
        std::vector<Room>::const_iterator last = _rooms.begin() + sample_index + sample_interval + 1;
        std::vector<Room> new_rooms(first, last);
        for(Room room : new_rooms){
            std::cout << sample_index << " " << room.get_caption() << std::endl;
            //std::cout << room.get_id() << std::endl;
            room.initialize_prior_distributions();
            room.sample_group_affordance();
            room.sample_supporting_affordance();
            //cool_down_wall_objects(room);
            run_mcmc(room, 0, room.get_furniture_list().size());

            std::string jsonOutputFolder = _workspacePath+"tmp/samples/"+room.get_caption()+"/json/", txtOutputFolder = _workspacePath+"tmp/samples/"+room.get_caption()+"/txt/";
            boost::filesystem::create_directories(jsonOutputFolder);
            boost::filesystem::create_directories(txtOutputFolder);

            std::stringstream outputFilename;
            outputFilename << "sample_";
            outputFilename << std::setfill('0') << std::setw(10) << sample_index;
            room.write_to_room_arranger_file(txtOutputFolder+outputFilename.str()+".txt");
            room.write_to_json(jsonOutputFolder+outputFilename.str()+".json");
            sample_index++;
        }
    }

    void Sampler::run_mcmc(Room &room, size_t startFurnIndex, size_t endFurnIndex) {
        _iterationMax = 5000 * room.get_furniture_list().size();
        //std::cout << "compute_total_energy" << std::endl;
        double currentEnergy = room.compute_total_energy(_costWeights), proposedEnergy;

        _temperature = 10;
        for (int i = 0; i < _iterationMax; i++) {
            //std::cout << "MCMC iteration: " << i << ", temperature: " << _temperature << std::endl;
            anneal(i);

            Room proposedRoom = room;
            proposedRoom.propose(_stepSize, startFurnIndex, endFurnIndex, false);
            proposedEnergy = proposedRoom.compute_total_energy(_costWeights);

            double r = (rand() / (double)RAND_MAX), acceptanceProb = std::min(1.0, exp(  (currentEnergy - proposedEnergy)/_temperature) );
            if (r < acceptanceProb){
                //std::cout << "Accepted: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
                room = proposedRoom;
                currentEnergy = proposedEnergy;
            }else{
                //std::cout << "Rejected: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
            }

            //if (i % 100 == 0){
            //    // Output every result to file for visualization
            //    std::stringstream outputFilename;
            //    outputFilename << "sample_";
            //    outputFilename << std::setfill('0') << std::setw(8) << i;
            //    room.write_to_room_arranger_file(_workspacePath+"tmp/samples/mcmc/txt/"+outputFilename.str()+".txt");
            //    room.write_to_json(_workspacePath+"tmp/samples/mcmc/json/"+outputFilename.str()+".json");
            //}
        }
    }

    void Sampler::anneal(int iteration) {
        if (iteration % 100 == 0){
            _temperature *= 0.9;
        }
        _stepSize = (_iterationMax-iteration)/double(_iterationMax) + 0.01;
    }

    void Sampler::cool_down_wall_objects(Room &room){
        _iterationMax = 5000;
        for (unsigned int iFurn = 0; iFurn < room.get_furniture_list().size(); iFurn++){
            if (room.get_supporting_index(iFurn) == -2){
                double currentEnergy = room.compute_total_energy(_costWeights), proposedEnergy;

                _temperature = 0.1;
                for (unsigned int i = 0; i < _iterationMax; i++) {
                    //std::cout << "Cool down iteration: " << i << ", temperature: " << _temperature << std::endl;
                    anneal(i);
                    _stepSize = 0.05*(_iterationMax-i)/double(_iterationMax) + 0.01;

                    Room proposedRoom = room;
                    proposedRoom.propose(_stepSize, iFurn, iFurn+1);
                    proposedEnergy = proposedRoom.compute_total_energy(_costWeights);

                    double r = (rand() / (double)RAND_MAX), acceptanceProb = std::min(1.0, exp(  (currentEnergy - proposedEnergy)/_temperature) );
                    if (r < acceptanceProb){
                        //std::cout << "Accepted: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
                        room = proposedRoom;
                        currentEnergy = proposedEnergy;
                    }else{
                        //std::cout << "Rejected: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
                    }

                    //if (i % 100 == 0) {
                    //    // Output iteration results to file for visualization
                    //    std::stringstream outputFilename;
                    //    outputFilename << "sample_";
                    //    outputFilename << std::setfill('0') << std::setw(8) << total_iteration;
                    //    room.write_to_room_arranger_file(
                    //            _workspacePath + "tmp/samples/mcmc/txt/" + outputFilename.str() + ".txt");
                    //    room.write_to_json(_workspacePath + "tmp/samples/mcmc/json/" + outputFilename.str() + ".json");
                    //}
                }
            }
        }
    }
}
