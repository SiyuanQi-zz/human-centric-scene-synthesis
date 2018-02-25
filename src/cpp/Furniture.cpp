//
// Created by siyuan on 1/19/17.
//

#include "Furniture.h"

namespace FurnitureArranger {
    SUNCGMetadata Furniture::_metadata = SUNCGMetadata();
    const double Furniture::_maxSupportingOjbectFloorDistance = 0.2, Furniture::_minBlockableHeight = 0.3;
    const int Furniture::_affordanceBins = 20, Furniture::_humanSampleCount = 100;
    const double Furniture::_affDistanceLimit = 2.0;

    Furniture::Furniture() {
    }

    Furniture::Furniture(json node) {
        _id = node["id"];
        _modelID = node["modelId"];
        _caption = _metadata.get_category(_modelID);
        _captionCoarse = _metadata.get_coarse_category(_modelID);

        _size = _metadata.get_dims(_modelID);
        read_transformation_matrix(node["transform"]);
        _translation += prod(make_rotation_matrix(get_rotation()), _metadata.get_min_piont(_modelID));
        RTTotransformMatrix();
    }

    void Furniture::read_metadata(std::string metadataPath) {
        _metadata.read_metadata(metadataPath);
    }

    void Furniture::set_caption(std::string caption) {
        _caption = caption;
        _captionCoarse = _metadata.get_coarse_category_from_fine(caption);
        std::cout << _caption << " " << _captionCoarse << std::endl;
    }

    bool Furniture::blockable() const {
        return (_translation(1) < _maxSupportingOjbectFloorDistance) && (_size(1) > _minBlockableHeight);
    }

    void Furniture::initialize_prior_distributions(json distParams, json secondDistParams, json oriParams) {
        _distanceDist = bm::lognormal_distribution<double>(log(double(distParams[2])), double(distParams[0]));
        _distanceDistLoc = double(distParams[1]);

        _secondDistanceDist = bm::lognormal_distribution<double>(log(double(secondDistParams[2])),
                                                                 double(secondDistParams[0]));
        _secondDistanceDistLoc = double(secondDistParams[1]);

        for (auto &p : oriParams) {
            std::vector<double> params = {double(p[0]), double(p[1]), double(p[2])};
            _orientationDistParams.push_back(params);
        }
    }

    void Furniture::set_affordance_map(const cv::Mat &inMap) {
        // Note: assert() does not work in release mode
        assert(_affordanceBins == inMap.rows);
        assert(_affordanceBins == inMap.cols);

        if (_affordanceBins != inMap.rows || _affordanceBins != inMap.cols){
            for (int i = 0; i < _affordanceBins; i++){
                for (int j = 0; j < _affordanceBins; j++) {
                    // Invalid input affordance map
                    _affordanceMap.push_back(1.0);
                    _accumAffordanceMap.push_back(0.0);
                }
            }
        }else{
            for (int i = 0; i < inMap.rows; i++){
                for (int j = 0; j < inMap.cols; j++) {
                    _affordanceMap.push_back(inMap.at<double>(i, j));
                    if (_accumAffordanceMap.size() == 0){
                        _accumAffordanceMap.push_back(_affordanceMap.back());
                    }else{
                        _accumAffordanceMap.push_back(_accumAffordanceMap.back()+_affordanceMap.back());
                    }
                }
            }
        }
    }

    double Furniture::map_to_relative_coor(int gridCoor) const {
        return (gridCoor/double(_affordanceBins))*2*_affDistanceLimit - _affDistanceLimit + 2*_affDistanceLimit/_affordanceBins*(rand() / (double)RAND_MAX);
    }

    int Furniture::relative_coor_to_map(double relPos) const {
        return int(((relPos+_affDistanceLimit)/(2*_affDistanceLimit))*_affordanceBins);
    }

    void Furniture::sample_affordance() {
        // Sample human position and save the relative coordinate
        _humanPositions.clear();
        for (int i = 0; i < _humanSampleCount; i++){
            ub::vector<double> _humanPosition = ub::zero_vector<double>(3);

            if (_accumAffordanceMap.back() == 0){
                _humanPosition(2) = _size(2);
                _humanPositions.push_back(_humanPosition);
                _humanProbs.push_back(1.0);
            }else{
                double r = (rand() / (double)RAND_MAX);
                int lowerBound = std::distance(_accumAffordanceMap.begin(), std::lower_bound(_accumAffordanceMap.begin(), _accumAffordanceMap.end(), r));
                int x = lowerBound%_affordanceBins, y = lowerBound/_affordanceBins;

                // Compute the relative coordinate from the affordance map position
                _humanPosition(0) = map_to_relative_coor(x);
                _humanPosition(1) = 0.0;
                _humanPosition(2) = map_to_relative_coor(y);

                _humanPositions.push_back(_humanPosition);
                _humanProbs.push_back(_affordanceMap[y*_affordanceBins+x]);
            }
        }
    }

    std::vector<boost::numeric::ublas::vector<double>> Furniture::get_human_positions() const {
        std::vector<boost::numeric::ublas::vector<double>> worldHumanPositions;
        for (ub::vector<double> const &humanPosition : _humanPositions){
            worldHumanPositions.push_back(compute_world_pos(humanPosition));
        }

        return worldHumanPositions;
    }

    std::vector<double> Furniture::get_sampled_human_probs() const {
        return _humanProbs;
    }

    double Furniture::compute_distance_log_prob(const double &distance) const {
        if (distance < _distanceDistLoc){
            return log(0.001);
        }else{
            return log(pdf(_distanceDist, distance - _distanceDistLoc));
        }
    }

    double Furniture::compute_second_distance_log_prob(const double &distance) const {
        if (distance < _secondDistanceDistLoc){
            return log(0.001);
        }else{
            return log(pdf(_secondDistanceDist, distance - _secondDistanceDistLoc));
        }
    }

    double Furniture::compute_orientation_log_prob(const double &orientation) const {
        double prob = 0;
        for (size_t i = 0; i < 4; i++) {
            double weight = _orientationDistParams[i][0], kappa = _orientationDistParams[i][1], mu = _orientationDistParams[i][2];
            double bessel = bm::cyl_bessel_i(0, kappa);
            prob += weight * exp(kappa * cos(orientation - mu)) / (2 * M_PI * bessel);
        }
        return log(prob);
    }

    double Furniture::compute_human_position_log_prob(const std::vector<boost::numeric::ublas::vector<double>> &worldPositions, std::vector<double> humanProbs) const {
        double logProb, maxLogProb = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < worldPositions.size(); i++){
            ub::vector<double> worldPos = worldPositions[i];
            ub::vector<double> relativePos = compute_relative_pos(worldPos);
            if (std::abs(relativePos(0))>2 || std::abs(relativePos(2))>2){
                logProb = log(0.001)-norm_2(relativePos);
            }else{
                int x = relative_coor_to_map(relativePos(0)), y = relative_coor_to_map(relativePos(2));
                logProb = log(_affordanceMap[y*_affordanceBins+x]*humanProbs[i]+0.001);
            }

            if (logProb > maxLogProb){
                maxLogProb = logProb;
            }
        }
        return maxLogProb;
    }
}
