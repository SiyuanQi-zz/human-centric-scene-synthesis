//
// Created by siyuan on 1/20/17.
//

#include "SUNCGMetadata.h"

namespace FurnitureArranger{
    SUNCGMetadata::SUNCGMetadata() {
        modelsId = std::vector<std::string>();
        modelsMinPoint = std::vector<boost::numeric::ublas::vector<double>>();
        modelsMaxPoint = std::vector<boost::numeric::ublas::vector<double>>();
        modelsAlignedDims = std::vector<boost::numeric::ublas::vector<double>>();
    }

    void SUNCGMetadata::read_metadata(std::string metadataPath) {
        read_models(metadataPath);
        read_model_category_mapping(metadataPath);
    }

    void SUNCGMetadata::read_models(std::string metadataPath){
        // File format: id,front,nmaterials,minPoint,maxPoint,aligned.dims,index,variantIds
        std::ifstream fin(metadataPath+"models.csv", std::ifstream::in);
        std::string line;

        std::getline(fin, line);

        while(std::getline(fin, line)) {
            boost::tokenizer<boost::escaped_list_separator<char> > tk(
                    line, boost::escaped_list_separator<char>('\\', ',', '\"'));
            boost::tokenizer<boost::escaped_list_separator<char> >::iterator it(tk.begin());

            modelsId.push_back(*it);
            std::advance(it, 3);
            modelsMinPoint.push_back(string_to_vector(*it, ", "));
            std::advance(it, 1);
            modelsMaxPoint.push_back(string_to_vector(*it, ", "));
            std::advance(it, 1);
            modelsAlignedDims.push_back(string_to_vector(*it, ", ")/100.0);
        }

        //// Check for inconsistency between min, max and dimension
        //for(unsigned int i=0; i < modelsAlignedDims.size(); i++) {
        //    if (norm_2((modelsMaxPoint[i] - modelsMinPoint[i]) * 100 - modelsAlignedDims[i]) > 0.1){
        //        std::cout << "Norm too large." << std::endl;
        //        std::cout << i << " " << modelsMinPoint[i] << " " << modelsMinPoint[i] << " " << modelsAlignedDims[i] << " " << norm_2((modelsMaxPoint[i] - modelsMinPoint[i]) * 100 - modelsAlignedDims[i]) << std::endl;
        //    }
        //}
    }

    void SUNCGMetadata::read_model_category_mapping(std::string metadataPath) {
        // File format: index,model_id,fine_grained_class,coarse_grained_class,empty_struct_obj,nyuv2_40class,wnsynsetid,wnsynsetkey
        std::ifstream fin(metadataPath+"ModelCategoryMapping.csv", std::ifstream::in);
        std::string line;

        std::getline(fin, line);

        while(std::getline(fin, line)) {
            boost::tokenizer<boost::escaped_list_separator<char> > tk(
                    line, boost::escaped_list_separator<char>('\\', ',', '\"'));
            boost::tokenizer<boost::escaped_list_separator<char> >::iterator it(tk.begin());

            std::advance(it, 1);
            categoryModelId.push_back(*it);
            std::advance(it, 1);
            categoryFineGrainedClass.push_back(*it);
            std::advance(it, 1);
            categoryCoarseGrainedClass.push_back(*it);
            std::advance(it, 1);
        }
    }

    ub::vector<double> SUNCGMetadata::get_min_piont(std::string modelID) const{
        return modelsMinPoint[find_model_index(modelID)];
    }

    ub::vector<double> SUNCGMetadata::get_dims(std::string modelID) const{
        return modelsAlignedDims[find_model_index(modelID)];
    }

    std::string SUNCGMetadata::get_category(std::string modelID) const{
        return categoryFineGrainedClass[find_category_index(modelID)];
    }

    std::string SUNCGMetadata::get_coarse_category(std::string modelID) const{
        return categoryCoarseGrainedClass[find_category_index(modelID)];
    }

    std::string SUNCGMetadata::get_coarse_category_from_fine(std::string fineCategory) const {
        auto it = std::find(categoryFineGrainedClass.begin(), categoryFineGrainedClass.end(), fineCategory);
        auto index = std::distance(categoryFineGrainedClass.begin(), it);
        return categoryCoarseGrainedClass[index];
    }

    ub::vector<double> SUNCGMetadata::string_to_vector(std::string inputString, std::string delim){
        std::vector<std::string> strvec;
        boost::algorithm::trim_if(inputString, boost::algorithm::is_any_of(delim));
        boost::algorithm::split(strvec, inputString, boost::algorithm::is_any_of(delim), boost::algorithm::token_compress_on);

        ub::vector<double> vec(strvec.size());
        for( unsigned int i=0; i<strvec.size(); i++) {
            vec[i] = boost::lexical_cast<double>(strvec[i]);
        }
        return vec;
    }

    int SUNCGMetadata::find_model_index(std::string modelID) const{
        auto it = std::find(modelsId.begin(), modelsId.end(), modelID);
        auto index = std::distance(modelsId.begin(), it);
        return index;
    }

    int SUNCGMetadata::find_category_index(std::string modelID) const{
        auto it = std::find(categoryModelId.begin(), categoryModelId.end(), modelID);
        auto index = std::distance(categoryModelId.begin(), it);
        return index;
    }
}