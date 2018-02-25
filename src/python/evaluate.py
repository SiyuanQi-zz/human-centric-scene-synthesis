"""
Created on Feb 19, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil
import json

import tabulate
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import cv2

import config
import learn_distribution

# ============================= Human Evaluation =============================
def prepare_evaluation(paths):
    for room_type in os.listdir(os.path.join(paths.tmp_root, 'samples')):
        exp_folder = os.path.join(paths.tmp_root, 'experiment', room_type)
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        sample_folder = os.path.join(paths.tmp_root, 'samples', room_type)
        with open(os.path.join(sample_folder, 'scores.json')) as f:
            scores_dict = json.load(f)
        print room_type
        scores = [score for index, score in scores_dict.items()]
        indices= [index for index, score in scores_dict.items()]
        scores = np.array(scores)
        sorted_order = np.argsort(scores)

        rank_count = 3
        sample_count = 8
        for rank in range(rank_count):
            for i in range(scores.shape[0]*rank/rank_count, scores.shape[0]*rank/rank_count+sample_count):
                src = os.path.join(sample_folder, 'figures', 'sample_{:04d}.png'.format(int(indices[sorted_order[i]])))
                dest = os.path.join(exp_folder, '{:d}_sample_{:04d}.png'.format(i, int(indices[sorted_order[i]])))
                shutil.copyfile(src, dest)

        # plt.plot(scores)
        # plt.show()
        # break


def print_latex_table(data, row_labels, col_labels):
    row_labels = np.array(row_labels)
    row_labels = np.reshape(row_labels, [row_labels.shape[0], 1])
    data = np.hstack((row_labels, data))
    print
    print(tabulate.tabulate(data, tablefmt="latex", floatfmt=".2f", numalign="center", headers=col_labels))


def evaluate(paths):
    methods = ['random', 'object', 'ours']
    rank_count = len(methods)

    rated = ['Bathroom', 'Bedroom', 'Dining_Room', 'Garage', 'Guest_Room', 'Gym', 'Kitchen', 'Living_Room', 'Office', 'Storage']

    room_types = list()
    data = list()
    for room_type in rated:
        room_types.append(room_type.replace('_', ' '))
        data.append(list())

        with open(os.path.join(paths.tmp_root, 'experiment', room_type, 'rating.json')) as f:
            ratings = np.array(json.load(f))
            # Different criteria
            for q in ratings:
                data[-1].append(list())
                # Different methods
                for method in q:
                    mu, std = scipy.stats.norm.fit(method)
                    data[-1][-1].append('{:.2f} pm {:.2f}'.format(mu, std))

    for q in range(2):
        print_latex_table(np.array(data)[:, q, :].T, methods, room_types)


def qualitative_result(paths):
    for room_type in os.listdir(os.path.join(paths.tmp_root, 'samples')):
        res_folder = os.path.join(paths.tmp_root, 'qualitative', room_type)
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        sample_folder = os.path.join(paths.tmp_root, 'samples', room_type)
        if room_type in selected_dict:
            for sample_index in selected_dict.get(room_type):
                # Sample figure
                src = os.path.join(sample_folder, 'figures', 'sample_{:04d}.png'.format(sample_index))
                dest = os.path.join(res_folder, 'sample_{:04d}.png'.format(sample_index))
                shutil.copyfile(src, dest)

                # Sample txt
                src = os.path.join(sample_folder, 'txt', 'sample_{:04d}.txt'.format(sample_index))
                dest = os.path.join(res_folder, 'sample_{:04d}.txt'.format(sample_index))
                shutil.copyfile(src, dest)


def print_figure_latex():
    objects = ['desk', 'coffee_table', 'dining_table', 'books', 'laptop', 'stand',
               'fruit_bowl', 'vase', 'floor_lamp', 'wall_lamp', 'fireplace', 'ceiling_fan']
    for o in objects:
        print '\\subfloat[{}]{{ \includegraphics[width=0.15\\textwidth]{{../fig/raw/affordance/{}.png}} }}'.format(o.replace('_', ' '), o)


def show_heatmaps(paths):
    figure_folder = os.path.join(paths.tmp_root, 'heatmaps', 'figures')
    for figure in sorted(os.listdir(figure_folder)):
        print figure
        img = cv2.imread(os.path.join(figure_folder, figure))
        cv2.imshow('image', img)
        cv2.waitKey(0)


# ============================= Afforance Evaluation =============================
affordance_bins = learn_distribution.affordance_bins
distance_limit = learn_distribution.distance_limit


def compute_affordance_kl_divergence(paths):
    prior_folder = os.path.join(paths.metadata_root, 'prior', 'affordance')

    with open(os.path.join(paths.metadata_root, 'stats', 'furnitureAffordanceSynthesized.json')) as f:
        syn_affordance = json.load(f)

    avg_total_variation = 0
    avg_h_distance = 0
    avg_kl_divergence = 0
    valid_furniture_count = 0
    for furniture, affordance_list in syn_affordance.items():
        aff = learn_distribution.filter_afforance(np.array(affordance_list))
        heatmap, xedges, yedges = np.histogram2d(aff[:, 0], aff[:, 1], bins=[affordance_bins, affordance_bins], range=[[-distance_limit, distance_limit], [-distance_limit, distance_limit]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Save the affordance map as a probability map
        if np.sum(heatmap) > 0:
            heatmap = heatmap/np.sum(heatmap)
            valid_furniture_count += 1
        else:
            continue

        heatmap_gt = cv2.cv.Load(os.path.join(prior_folder, furniture+'.xml'))
        heatmap_gt = np.asarray(heatmap_gt)

        threshold = 0.00
        heatmap[heatmap < threshold] = 0
        heatmap = heatmap/np.sum(heatmap)
        heatmap_gt[heatmap_gt < threshold] = 0
        heatmap_gt = heatmap_gt/np.sum(heatmap_gt)

        # Total variation distance
        total_variation = np.sum(np.abs(heatmap-heatmap_gt))/2
        avg_total_variation += total_variation
        # print furniture, total_variation

        # Helligenger distance
        h_distance = 0
        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                h_distance += (np.sqrt(heatmap_gt[x, y]) - np.sqrt(heatmap[x, y]))**2
        h_distance = np.sqrt(h_distance/2)
        avg_h_distance += h_distance
        # print furniture, h_distance

        # # KL-divergence
        # kl_divergence = 0
        # for x in range(heatmap.shape[0]):
        #     for y in range(heatmap.shape[1]):
        #         # if heatmap_gt[x, y] != 0:
        #         #     kl_divergence += heatmap_gt[x, y] * np.log(heatmap_gt[x, y]+0.01/heatmap[x, y]+0.01)
        #         if heatmap[x, y] != 0:
        #             kl_divergence += heatmap[x, y] * np.log(heatmap[x, y]/heatmap_gt[x, y])
        # avg_kl_divergence += kl_divergence

        # fig1 = plt.figure()
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        # fig2 = plt.figure()
        # plt.imshow(heatmap_gt.T, extent=extent, origin='lower')
        # plt.show()
        # break
    print avg_total_variation/valid_furniture_count
    print avg_h_distance/valid_furniture_count
    print avg_kl_divergence/valid_furniture_count


def test():
    length = 400
    mu, sigma = 0, 100
    gt = np.array([float(a) for a in range(1, length+1)])
    gt_noise = np.abs(gt + np.random.normal(mu, sigma, length))

    gt /= np.sum(gt)
    gt_noise /= np.sum(gt_noise)

    print np.dot(gt, np.log(gt/gt_noise))


def main():
    paths = config.Paths()
    # prepare_evaluation(paths)
    # evaluate(paths)
    qualitative_result(paths)
    # print_figure_latex()
    # show_heatmaps(paths)
    # compute_affordance_kl_divergence(paths)
    # test()


if __name__ == '__main__':
    main()
