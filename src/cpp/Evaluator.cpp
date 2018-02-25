//
// Created by siyuan on 2/19/17.
//

#include "Evaluator.h"

namespace FurnitureArranger {
    Evaluator::Evaluator(){

    }

    Evaluator::Evaluator(std::string suncgRoot, std::string metadataPath, std::string workspacePath): _suncgRoot(suncgRoot), _metadataPath(metadataPath), _workspacePath(workspacePath){
        // Important: Read metadata
        Room::read_group_metadata(metadataPath);
        Furniture::read_metadata(metadataPath);

        if (!load_evaluator()){
            // Read all json files
            int room_count = 0;
            fs::directory_iterator it(fs::path(_workspacePath + "tmp/samples/")), eod;
            BOOST_FOREACH(fs::path const &roomType, std::make_pair(it, eod)){
                //std::cout << fs::path(roomType) << std::endl;
                if (roomType.stem() != "Storage"){
                    continue;
                }
                fs::directory_iterator it2(fs::path(roomType/"txt")), eod2;
                BOOST_FOREACH(fs::path const &sampleTxt, std::make_pair(it2, eod2)) {
                    //std::cout << fs::path(sampleTxt) << std::endl;
                    read_sampled_txt(sampleTxt);
                    room_count++;
                    //break;
                }
                //break;
            }

            save_evaluator();
        }
    }

    void Evaluator::parse_line(const std::string &line, std::string &caption, ub::vector<double> &size, ub::vector<double> &translation, double &rotation) {
        std::vector<std::string> strvec, strvec2;
        boost::algorithm::split(strvec, line, boost::algorithm::is_any_of(";"), boost::algorithm::token_compress_off);

        caption = strvec[0];

        boost::algorithm::split(strvec2, strvec[2], boost::algorithm::is_any_of("x"), boost::algorithm::token_compress_on);
        size.resize(3);
        size[0] = atof(strvec2[0].c_str());
        size[1] = atof(strvec2[2].c_str());
        size[2] = atof(strvec2[1].c_str());
        strvec2.clear();

        boost::algorithm::split(strvec2, strvec[3], boost::algorithm::is_any_of(","), boost::algorithm::token_compress_on);
        translation.resize(3);
        translation[0] = atof(strvec2[0].c_str());
        translation[1] = atof(strvec2[2].c_str());
        translation[2] = atof(strvec2[1].c_str());
        strvec2.clear();

        rotation = atof(strvec[4].c_str());
    }

    void Evaluator::read_sampled_txt(const fs::path &filename) {
        if (fs::exists(filename)) {
            std::ifstream input_file(filename.string(), std::ios::in);
            Room room;

            std::string line;
            int line_no = 0;
            while(std::getline(input_file, line)){
                line_no++;
                if (line_no == 1){
                    continue;
                }

                std::string caption;
                ub::vector<double> size, translation;
                double rotation;
                parse_line(line, caption, size, translation, rotation);

                if (line_no == 2){
                    room.set_caption(caption);
                    room.set_size(size);
                    room.set_translation(translation);
                    room.set_rotation(rotation);
                }else{
                    Furniture furniture;
                    furniture.set_caption(caption);
                    furniture.set_size(size);
                    furniture.set_translation(translation);
                    furniture.set_rotation(rotation);
                    room.add_furniture(furniture);
                }
            }
            _rooms.push_back(room);
        }
    }


    bool Evaluator::load_evaluator() {
        return false;
        if (fs::exists(_workspacePath + "tmp/evaluator"))
        {
            std::ifstream ifs(_workspacePath + "tmp/evaluator");
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

    void Evaluator::save_evaluator() {
        std::ofstream ofs(_workspacePath + "tmp/evaluator");
        boost::archive::text_oarchive oa(ofs);
        oa << *this;
        ofs.close();
    }

    void Evaluator::collect_affordance() {
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
            std::ofstream ofs(_metadataPath + "stats/furnitureAffordanceSynthesized.json");
            ofs << furnitureAffordance.dump(4);
            ofs.close();
        }
    }

    template<class Archive>
    void Evaluator::serialize(Archive & ar, const unsigned int version)
    {
        ar & _suncgRoot;
        ar & _metadataPath;
        ar & _workspacePath;
        ar & _rooms;
    }
}