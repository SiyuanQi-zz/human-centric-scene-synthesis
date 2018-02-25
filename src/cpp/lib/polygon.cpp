//
// Created by siyuan on 2/8/17.
//

#include "polygon.h"

// Compute the intersection of P and Q.
double compute_intersection(Polygon_2 P, Polygon_2 Q)
{
    Pwh_list_2                  intR;
    Pwh_list_2::const_iterator  it;
    double                      total_area = 0.0;

    CGAL::intersection (P, Q, std::back_inserter(intR));
    for (it = intR.begin(); it != intR.end(); ++it) {
        total_area += CGAL::to_double(it->outer_boundary().area());
    }

    return total_area;
}