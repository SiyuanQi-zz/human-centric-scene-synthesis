//
// Created by siyuan on 1/19/17.
//

#ifndef CVPR2018_ENTITY_H
#define CVPR2018_ENTITY_H

#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <limits>

// Boost header files
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

// OpenCV header files
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lib/json.hpp"

#include "helper.h"

using json = nlohmann::json;
namespace ub = boost::numeric::ublas;

namespace FurnitureArranger {
    class Entity {
    public:
        Entity();

        void scale_to_meter(double scaleToMeters = 1);

        std::string get_id() const;

        std::string get_caption() const;
        virtual void set_caption(std::string caption);

        std::string get_caption_coarse() const;
        void set_caption_coarse(std::string captionCoarse);

        ub::vector<double> get_size() const;
        void set_size(const ub::vector<double> &size);

        ub::vector<double> get_translation() const;
        void set_translation(const ub::vector<double> &translation);

        double get_rotation() const;
        void set_rotation(const double &rotation);

        void translate(const ub::vector<double> &translation);
        void rotate(const double &rotation);

        void normalize_rotation();

        void compute_vertex_coor();
        ub::vector<double> get_vertex_coor(size_t index) const;
        std::vector<double> get_point_to_edge_distance(ub::vector<double> point) const;

        const ub::vector<double> compute_relative_pos(const ub::vector<double> &worldPos) const;
        const ub::vector<double> compute_world_pos(const ub::vector<double> &relativePos) const;

        std::string to_room_arranger_object(std::string objectID="") const;
        friend std::ostream &operator<<(std::ostream &os, const Entity &Entity);

    protected:
        std::string _id;
        std::string _modelID;
        std::string _caption;
        std::string _captionCoarse;

        ub::vector<double> _size;
        ub::matrix<double> _transformMatrix;
        ub::vector<double> _translation;
        double _rotation;

        ub::matrix<double> _vertices;

        void read_transformation_matrix(json arr);
        void transformMatrixToRT();
        void RTTotransformMatrix();

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & _id;
            ar & _modelID;
            ar & _caption;
            ar & _captionCoarse;

            ar & _size;
            ar & _transformMatrix;
            ar & _translation;
            ar & _rotation;

            ar & _vertices;
        };
    };
}

#endif //CVPR2018_ENTITY_H
