//
// Created by siyuan on 1/24/17.
//

#include "helper.h"

namespace FurnitureArranger {
    void read_json_file(std::string filename, json &data) {
        {
            if (fs::exists(filename)) {
                std::ifstream input_file(filename, std::ios::in | std::ios::binary);
                std::ostringstream contents_stream;
                contents_stream << input_file.rdbuf();
                input_file.close();
                data = json::parse(contents_stream.str());
            }
        }
    }

    ub::vector<double> json_to_ub_vector(json jsonVec){
        ub::vector<double> ubVec;
        ubVec.resize(jsonVec.size());
        for (int i = 0; i < jsonVec.size(); i++){
            ubVec(i) = jsonVec[i];
        }
        return ubVec;
    }

    ub::vector<double> string_to_vector(std::string inputString, std::string delim) {
        std::vector <std::string> strvec;
        boost::algorithm::trim_if(inputString, boost::algorithm::is_any_of(delim));
        boost::algorithm::split(strvec, inputString, boost::algorithm::is_any_of(delim),
                                boost::algorithm::token_compress_on);

        ub::vector<double> vec(strvec.size());
        for (unsigned int i = 0; i < strvec.size(); i++) {
            vec(i) = boost::lexical_cast<double>(strvec[i]);
        }
        return vec;
    }

    // For opencv type checking
    std::string cv_type2str(int type) {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }

    ub::matrix<double> make_rotation_matrix(const double &rotation){
        ub::matrix<double> rot = ub::identity_matrix<double>(3);
        rot(0, 0) = cos(rotation);
        rot(2, 2) = cos(rotation);
        rot(0, 2) = -sin(rotation);
        rot(2, 0) = sin(rotation);
        return rot;
    }

    ub::matrix<double> make_transformation_matrix(const ub::vector<double> &translation, const double &rotation){
        ub::matrix<double> transformMatrix = ub::identity_matrix<double>(4);
        for (int i = 0; i < 3; i++){
            transformMatrix(i, 3) = translation(i);
        }
        transformMatrix(0, 0) = cos(rotation);
        transformMatrix(2, 2) = cos(rotation);
        transformMatrix(0, 2) = -sin(rotation);
        transformMatrix(2, 0) = sin(rotation);
        return transformMatrix;
    }


    int determinant_sign(const ub::permutation_matrix<std ::size_t>& pm)
    {
        int pm_sign=1;
        std::size_t size = pm.size();
        for (std::size_t i = 0; i < size; ++i)
            if (i != pm(i))
                pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
        return pm_sign;
    }

    double determinant( ub::matrix<double>& m ) {
        ub::permutation_matrix<std::size_t> pm(m.size1());
        double det = 1.0;
        if( ub::lu_factorize(m,pm) ) {
            det = 0.0;
        } else {
            for(int i = 0; i < m.size1(); i++)
                det *= m(i,i); // multiply by elements on diagonal
            det = det * determinant_sign( pm );
        }
        return det;
    }

    bool point_in_polygon(const ub::vector<double> &point, const std::vector<boost::numeric::ublas::vector<double>> &poly) {
        // point is a 3d vector, poly is a array of 3d vectors of length n
        // Use cross product to check if the point is at the same side of
        // all the sides of the polygon
        if (poly.size() < 3) {
            return false;
        }

        double lastCrossProductY = 0;
        for (int i = 0; i < poly.size(); i++){
            ub::vector<double> P0P = point - poly[i], P0P1 = poly[(i+1)%poly.size()] - poly[i];
            double crossProductY = P0P(2)*P0P1(0) - P0P1(2)*P0P(0);
            if (i > 0){
                if (lastCrossProductY * crossProductY < 0){
                    return false;
                }
            }
            lastCrossProductY = crossProductY;
        }

        return true;
    }
}