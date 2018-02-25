//
// Created by siyuan on 2/8/17.
//

#ifndef CVPR2018_POLYGON_H
#define CVPR2018_POLYGON_H


#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Boolean_set_operations_2.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel       K;
typedef K::Point_2                                              Point_2;
typedef CGAL::Polygon_2<K>                                      Polygon_2;
typedef CGAL::Polygon_with_holes_2<K>                           Polygon_with_holes_2;
typedef std::list<Polygon_with_holes_2>                         Pwh_list_2;


// Compute the intersection of P and Q.
double compute_intersection(Polygon_2 P, Polygon_2 Q);

#endif //CVPR2018_POLYGON_H
