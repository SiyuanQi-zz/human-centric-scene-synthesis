"""
Created on Jan 31, 2017

@author: Siyuan Qi

Description of the file.

"""

# System imports
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import cv2

import config


# ===================== Learn Furniture Wall Relationships =====================
def learn_log_normal(distances, furniture, paths):
    # TODO: investigate if the distribution is fitted correctly: the bin number affects the pdf, does it make sense to compare the predicted pdf with the histogram?
    shape, loc, scale = scipy.stats.lognorm.fit(distances)
    x = np.linspace(np.min(distances), np.max(distances), 100)
    pdf_fitted = scipy.stats.lognorm.pdf(x, shape, loc, scale)

    # Plot histogram and fitted distribution
    fig = plt.figure()
    plt.hist(distances, bins=20, normed=1)
    plt.plot(x, pdf_fitted, 'r-')

    ax = fig.gca()
    ax.set_title(furniture)
    # plt.show()
    fig.savefig(os.path.join(paths.tmp_root, 'distributions', furniture + '_distance.png'))
    plt.close(fig)
    return shape, loc, scale


def learn_mix_von_mises(orientations, furniture, paths):
    orientations[abs(orientations - 2 * np.pi) < np.pi / 4] -= 2 * np.pi

    params = list()
    fig = plt.figure()
    for i in range(4):
        partial_data = orientations[abs(orientations - i * np.pi / 2) < np.pi / 4]
        if partial_data.shape[0] == 0:
            params.append([0, 0, 0, 0])
            continue

        shape, loc, scale = scipy.stats.vonmises.fit(partial_data, fscale=1)
        params.append([partial_data.shape[0] / float(orientations.shape[0]), shape, loc, scale])

    # Plot fitted distribution
    x = np.linspace(-np.pi / 4, 2 * np.pi - np.pi / 4, 100)
    pdf_fitted = np.zeros(x.shape)
    for i in range(4):
        if params[i][0] == 0:
            continue
        pdf_fitted += params[i][0] * scipy.stats.vonmises.pdf(x, params[i][1], params[i][2], params[i][3])
    plt.plot(x, pdf_fitted, 'r-')

    # Plot histogram
    plt.hist(orientations, bins=20, normed=1)
    ax = fig.gca()
    ax.set_title(furniture)
    # plt.show()
    fig.savefig(os.path.join(paths.tmp_root, 'distributions', furniture + '_orientation.png'))
    plt.close(fig)

    return params


def learn_furniture_wall_relationships(paths):
    plot_folder = os.path.join(paths.tmp_root, 'distributions')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    with open(os.path.join(paths.metadata_root, 'stats', 'furnitureWallStats.json')) as f:
        furn_wall_stats = json.load(f)

    furn_wall_first_dist_prior = dict()
    furn_wall_second_dist_prior = dict()
    furn_wall_ori_prior = dict()
    for furniture, rel_list in furn_wall_stats.items():
        print furniture
        rel_list = np.array(rel_list)
        distances = rel_list[:, 0]
        second_distances = rel_list[:, 1]
        orientations = rel_list[:, 2]

        shape, loc, scale = learn_log_normal(distances, furniture + '_first', paths)
        furn_wall_first_dist_prior[furniture] = shape, loc, scale

        shape, loc, scale = learn_log_normal(second_distances, furniture + '_second', paths)
        furn_wall_second_dist_prior[furniture] = shape, loc, scale

        params = learn_mix_von_mises(orientations, furniture, paths)
        furn_wall_ori_prior[furniture] = params

    with open(os.path.join(paths.metadata_root, 'prior', 'furnWallDist.json'), 'w') as output_file:
        json.dump(furn_wall_first_dist_prior, output_file, indent=4)

    with open(os.path.join(paths.metadata_root, 'prior', 'furnWallSecondDist.json'), 'w') as output_file:
        json.dump(furn_wall_second_dist_prior, output_file, indent=4)

    with open(os.path.join(paths.metadata_root, 'prior', 'furnWallOri.json'), 'w') as output_file:
        json.dump(furn_wall_ori_prior, output_file, indent=4)


# ============================= Learn Affordance =============================
def filter_afforance(aff):
    distance_limit = 2
    aff_filter = np.logical_and(np.abs(aff[:, 0]) < distance_limit, np.abs(aff[:, 1]) < distance_limit)
    return aff[aff_filter, :]


affordance_bins = 20
distance_limit = 2


def learn_affordance_distributions(paths):
    with open(os.path.join(paths.metadata_root, 'stats', 'furnitureAffordance.json')) as f:
        affordance_stats = json.load(f)

    prior_folder = os.path.join(paths.metadata_root, 'prior', 'affordance')
    if not os.path.exists(prior_folder):
        os.makedirs(prior_folder)

    for furniture, affordance_list in affordance_stats.items():
        print furniture
        aff = filter_afforance(np.array(affordance_list))
        heatmap, xedges, yedges = np.histogram2d(aff[:, 0], aff[:, 1], bins=[affordance_bins, affordance_bins], range=[[-distance_limit, distance_limit], [-distance_limit, distance_limit]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Save the affordance map as a probability map
        if np.sum(heatmap) > 0:
            heatmap = heatmap/np.sum(heatmap)
        cv2.cv.Save(os.path.join(prior_folder, furniture+'.xml'), cv2.cv.fromarray(heatmap))

        # fig = plt.figure()
        # ax = fig.gca()
        # im = matplotlib.image.NonUniformImage(ax, interpolation='bilinear')
        # xcenters = (xedges[:-1] + xedges[1:]) / 2
        # ycenters = (yedges[:-1] + yedges[1:]) / 2
        # im.set_data(xcenters, ycenters, heatmap)
        # ax.images.append(im)
        # ax.set_xlim(xedges[0], xedges[-1])
        # ax.set_ylim(yedges[0], yedges[-1])
        # plt.show()

        # fig = plt.figure()
        # plt.clf()
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        # plt.show()
        # plt.close(fig)


def main():
    paths = config.Paths()
    # learn_furniture_wall_relationships(paths)
    learn_affordance_distributions(paths)


if __name__ == '__main__':
    main()
