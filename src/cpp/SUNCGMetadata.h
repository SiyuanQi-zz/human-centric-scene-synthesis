//
// Created by siyuan on 1/20/17.
//

#ifndef CVPR2018_SUNCGMETADATA_H
#define CVPR2018_SUNCGMETADATA_H

#include <fstream>
#include <vector>
#include <algorithm>

#include <boost/tokenizer.hpp>

#include "helper.h"

namespace ub = boost::numeric::ublas;

namespace FurnitureArranger {
    class SUNCGMetadata {
    public:
        SUNCGMetadata();

        void read_metadata(std::string metadataPath);
        void read_models(std::string metadataPath);
        void read_model_category_mapping(std::string metadataPath);

        ub::vector<double> get_min_piont(std::string modelID) const;
        ub::vector<double> get_dims(std::string modelID) const;
        std::string get_category(std::string modelID) const;
        std::string get_coarse_category(std::string modelID) const;
        std::string get_coarse_category_from_fine(std::string fineCategory) const;

        static ub::vector<double> string_to_vector(std::string inputString, std::string delim);

    private:
        std::vector<std::string> modelsId;
        std::vector<ub::vector<double>> modelsMinPoint;
        std::vector<ub::vector<double>> modelsMaxPoint;
        std::vector<ub::vector<double>> modelsAlignedDims;

        std::vector<std::string> categoryModelId;
        std::vector<std::string> categoryFineGrainedClass;
        std::vector<std::string> categoryCoarseGrainedClass;

        int find_model_index(std::string modelID) const;
        int find_category_index(std::string modelID) const;
    };
}


#endif //CVPR2018_SUNCGMETADATA_H
