"""
Created onJan 26, 2017

@author: Siyuan Qi

This file contains the basic configuration for the furniture arranging project.

"""

import errno
import logging
import os


class Paths(object):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        self.data_root = '/home/siyuan/data/SUNCG/'
        self.project_root = '/home/siyuan/projects/release/cvpr2018/'
        self.metadata_root = os.path.join(self.project_root, 'src/metadata')
        self.tmp_root = os.path.join(self.project_root, 'tmp')


def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


