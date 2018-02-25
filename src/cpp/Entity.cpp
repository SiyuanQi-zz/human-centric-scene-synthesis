//
// Created by siyuan on 1/19/17.
//

#include "Entity.h"

namespace FurnitureArranger {
    Entity::Entity() {
        _translation.resize(3);
        _transformMatrix.resize(4, 4);
        _vertices.resize(4, 5);
    }

    void Entity::scale_to_meter(double scaleToMeters) {
        _size *= scaleToMeters;
        _translation *= scaleToMeters;
        RTTotransformMatrix();
    }

    std::string Entity::get_id() const { return _id; }

    std::string Entity::get_caption() const { return _caption; }
    void Entity::set_caption(std::string caption) {
        _caption = caption;
    }

    std::string Entity::get_caption_coarse() const { return _captionCoarse; }
    void Entity::set_caption_coarse(std::string captionCoarse) {
        _captionCoarse = captionCoarse;
    }

    ub::vector<double> Entity::get_size() const { return _size; }
    void Entity::set_size(const ub::vector<double> &size) {
        _size = size;
    }

    ub::vector<double> Entity::get_translation() const { return _translation; }
    void Entity::set_translation(const ub::vector<double> &translation) {
        _translation = translation;
        RTTotransformMatrix();
    }

    double Entity::get_rotation() const { return _rotation; }
    void Entity::set_rotation(const double &rotation) {
        _rotation = rotation;
        RTTotransformMatrix();
    }

    void Entity::translate(const ub::vector<double> &translation) {
        _translation += translation;
        RTTotransformMatrix();
    }

    void Entity::rotate(const double &rotation) {
        _rotation += rotation;
        RTTotransformMatrix();
    }

    void Entity::normalize_rotation() {
        _rotation = _rotation - floor(_rotation / (2 * M_PI)) * (2 * M_PI);
    }

    void Entity::compute_vertex_coor() {
        for (size_t i = 0; i < 5; i++) {
            ub::vector<double> v(_size);
            v.resize(4);
            column(_vertices, i) = v;
        }
        row(_vertices, 1) = ub::zero_vector<double>(5);
        row(_vertices, 3) = ub::scalar_vector<double>(5, 1);
        _vertices(0, 0) = 0.0;
        _vertices(2, 0) = 0.0;
        _vertices(2, 1) = 0.0;
        _vertices(0, 3) = 0.0;
        _vertices(0, 4) /= 2.0;
        _vertices(2, 4) /= 2.0;

        _vertices = prod(_transformMatrix, _vertices);
    }

    ub::vector<double> Entity::get_vertex_coor(size_t index) const {
        ub::vector<double> vertexCoor(3);
        vertexCoor(0) = _vertices(0, index);
        vertexCoor(1) = 0;
        vertexCoor(2) = _vertices(2, index);
        return vertexCoor;
    }

    std::vector<double> Entity::get_point_to_edge_distance(ub::vector<double> point) const {
        std::vector<boost::numeric::ublas::vector<double>> corners;
        for (size_t i = 0; i < 4; i++) {
            corners.push_back(get_vertex_coor(i));
        }

        std::vector<double> distances;
        for (int i = 0; i < 4; i++) {
            ub::vector<double> ab = corners[(i + 1) % 4] - corners[i], ac = point - corners[i];
            double distance = sqrt(pow(norm_2(ac), 2.0) - pow(inner_prod(ab, ac) / norm_2(ab), 2.0));
            distances.push_back(distance);
        }

        return distances;
    }

    const ub::vector<double> Entity::compute_relative_pos(const ub::vector<double> &worldPos) const {
        ub::matrix<double> coorTransformMatrix = make_transformation_matrix(
                -prod(make_rotation_matrix(-get_rotation()), get_translation()), -get_rotation());
        ub::vector<double> relPos(4);
        relPos(0) = worldPos(0);
        relPos(1) = 0;
        relPos(2) = worldPos(2);
        relPos(3) = 1;
        relPos = prod(coorTransformMatrix, relPos);
        relPos.resize(3);
        relPos -= get_size() / 2;
        relPos(1) = 0;
        return relPos;
    }

    const ub::vector<double> Entity::compute_world_pos(const ub::vector<double> &relativePos) const {
        ub::vector<double> worldPos(4);
        subrange(worldPos, 0, 3) = relativePos + _size/2;
        worldPos(1) = 0;
        worldPos(3) = 1;
        worldPos = prod(_transformMatrix, worldPos);
        worldPos(1) = 0;
        worldPos.resize(3);
        return worldPos;
    }

    std::string Entity::to_room_arranger_object(std::string objectID) const {
        // File format: Caption;ID;Size (WxLxH);Position (X,Y,Z);Rotation;Price
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << _caption << ";" << objectID << ";";
        //ss << _modelID << ";;";
        ss << _size(0) * 100 << "x" << _size(2) * 100 << "x" << _size(1) * 100 << ";";
        ss << _transformMatrix(0, 3) * 100 << "," << _transformMatrix(2, 3) * 100 << "," <<
        _transformMatrix(1, 3) * 100 << ";";
        ss << _rotation * 180.0 / M_PI << ";" << 0;
        return ss.str();
    }

    std::ostream &operator<<(std::ostream &os, const Entity &entity) {
        os << entity._caption << " ";
        os << " _size: " << entity._size << " ";
        os << " _translation: " << entity._translation << " ";
        os << " _rotation: " << entity._rotation << " ";

        return os;
    }

    void Entity::read_transformation_matrix(json arr) {
        for (auto i = 0; i < _transformMatrix.size1(); i++) {
            for (auto j = 0; j < _transformMatrix.size2(); j++) {
                _transformMatrix(i, j) = arr[j * 4 + i];
            }
        }

        transformMatrixToRT();
    }

    void Entity::transformMatrixToRT() {
        ub::matrix_column <ub::matrix<double>> mc(_transformMatrix, 3);
        std::copy(mc.begin(), mc.end() - 1, _translation.begin());

        ub::matrix<double> R = subrange(_transformMatrix, 0, 3, 0, 3);
        if (std::abs(determinant(R) - 1) > 0.001) {
            _rotation = 0.0;
        } else {
            if (_transformMatrix(0, 2) < 0) {
                _rotation = acos(_transformMatrix(0, 0));
            } else {
                _rotation = -acos(_transformMatrix(0, 0));
            }

            if (std::isnan(_rotation)) {
                _rotation = 0.0;
            } else {
                normalize_rotation();
            }
        }
        compute_vertex_coor();
    }

    void Entity::RTTotransformMatrix() {
        _transformMatrix = make_transformation_matrix(_translation, _rotation);
        compute_vertex_coor();
    }
}