"""utils.py"""

import argparse
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
import os

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

    def get_images_set(self, len_of_images):
        assert len_of_images == len(self.data["images"])
        original_set = []
        recon_set = []
        for i in range(len_of_images):
            if i%2 == 0:
                original_set.extend(self.data["images"][i])
            elif i%2 == 1:
                recon_set.extend(self.data["images"][i])
        return original_set, recon_set

    def write_out(self, file_name):
        with open(os.path.join(file_name, 'record.txt'), "a") as gather_file:
            for key in self.data.keys():
                if str(key) != "images":
                    gather_file.write("{}-{} ".format(key,self.data[key]))
            gather_file.write("\n")

def plot_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax