//
// Created by siyuan on 1/24/17.
//

#ifndef CVPR2018_HELPER_H
#define CVPR2018_HELPER_H

// System header files
#include <fstream>

// Boost header files

#include <boost/filesystem.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>

// Lib header files
#include "lib/json.hpp"

// OpenCV header files
#include <opencv2/core/core.hpp>

using json = nlohmann::json;
namespace fs = boost::filesystem;
namespace ub = boost::numeric::ublas;

namespace FurnitureArranger {
    void read_json_file(std::string filename, json &data);

    ub::vector<double> json_to_ub_vector(json jsonVec);
    ub::vector<double> string_to_vector(std::string inputString, std::string delim);

    std::string cv_type2str(int type);

    ub::matrix<double> make_rotation_matrix(const double &rotation);
    ub::matrix<double> make_transformation_matrix(const ub::vector<double> &translation, const double &rotation);
    int determinant_sign(const ub::permutation_matrix<std ::size_t>& pm);
    double determinant( ub::matrix<double>& m );
    bool point_in_polygon(const ub::vector<double> &point, const std::vector<boost::numeric::ublas::vector<double>> &poly);
}

#endif //CVPR2018_HELPER_H
